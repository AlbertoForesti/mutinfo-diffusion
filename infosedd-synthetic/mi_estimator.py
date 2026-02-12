import pytorch_lightning as pl
import minde_utils
import infosedd_utils
import fdime_utils
from model import *
import model_minde
import graph_lib
import noise_lib
import sde_lib
import time
import psutil
import torch
import os
import glob

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

# Check if CPU energy monitoring is available (Intel RAPL on Linux)
CPU_ENERGY_AVAILABLE = os.path.exists('/sys/class/powercap/intel-rapl')

class EmbeddingLayer(LightningModule):
    def __init__(self, dim, vocab_dim):
        """
        Mode arg: 0 -> use a learned layer, 1 -> use eigenvectors, 
        2-> add in eigenvectors, 3 -> use pretrained embedding matrix
        """
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]

class MIEstimator(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.estimator = config.estimator
        self.results = {"mi_history": [], "runtime_train": [], "runtime_val": [], "memory": [], 
                       "inference_runtime": [], "energy_dissipated": [], "cpu_utilization": [],
                       "gpu_utilization": [], "cpu_energy": []}
        self.train_epoch_start_time = None
        self.val_epoch_start_time = None
        self.val_step_times = []
        self.val_step_energy_start = []
        self.val_step_cpu_energy_start = []
        self.process = psutil.Process()
        
        # Initialize GPU energy monitoring if available
        self.gpu_handle = None
        if PYNVML_AVAILABLE and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except:
                self.gpu_handle = None
        
        # Initialize CPU energy monitoring (Intel RAPL on Linux)
        self.cpu_energy_paths = []
        if CPU_ENERGY_AVAILABLE:
            try:
                # Find all RAPL energy files for package (CPU) energy
                rapl_base = '/sys/class/powercap/intel-rapl'
                for rapl_dir in glob.glob(f'{rapl_base}/intel-rapl:*'):
                    name_file = os.path.join(rapl_dir, 'name')
                    if os.path.exists(name_file):
                        with open(name_file, 'r') as f:
                            name = f.read().strip()
                        if name == 'package-0' or name.startswith('package'):
                            energy_file = os.path.join(rapl_dir, 'energy_uj')
                            if os.path.exists(energy_file):
                                self.cpu_energy_paths.append(energy_file)
            except Exception as e:
                print(f"Warning: Could not initialize CPU energy monitoring: {e}")
                self.cpu_energy_paths = []
        
        if 'infosedd' in self.estimator:
            graph = graph_lib.get_graph(config)
            noise = noise_lib.get_noise(config)
            self._loss = infosedd_utils.get_loss_fn(config, noise=noise, graph=graph, train=True, sampling_eps=config.sampling_eps)
            self._mutinfo_fn = infosedd_utils.get_mutinfo_step_fn(config, graph, noise)
            self.is_parametric_marginal = config.is_parametric_marginal
            self.alphabet_size = config.alphabet_size
            if config.graph == "absorb":
                self.alphabet_size += 1
            self.backbone = UnetMLP(config)
        elif 'minde' in self.estimator:
            self.use_embeddings = config.minde_use_embeddings
            self.alphabet_size = config.alphabet_size
            self.embedding_layer = EmbeddingLayer(config.init_dim, self.alphabet_size)
            self.sde = sde_lib.VP_SDE(config)
            self._mutinfo_fn = minde_utils.get_mutinfo_step_fn(self.sde, config.importance_sampling, config.minde_type, config.sampling_eps)
            self.backbone = model_minde.UnetMLP(config)
        else:
            self.divergence = config.divergence
            self.alpha = config.alpha
            self.backbone = CombinedNet(
                UnetMLP(config), # Net(length_of_x_plus_y, 1),
                config.divergence
            )

    def loss(self, x, y):
        if 'infosedd' in self.estimator:
            loss = self._loss(self.backbone, x, y)
        elif 'minde' in self.estimator:
            backbone_score_forward = partial(minde_utils.score_forward, self.backbone)
            if self.use_embeddings:
                x = self.embedding_layer(x).reshape(x.shape[0], -1)
                y = self.embedding_layer(y).reshape(y.shape[0], -1)
            loss = self.sde.train_step(x, y, backbone_score_forward).mean()
        else:
            y_shuffle = torch.index_select(y, 0, fdime_utils.derangement(list(range(y.shape[0])), self.device))
            xy_shuffle = torch.hstack((x, y_shuffle))
            xy = torch.hstack((x, y))

            D_value_1, D_value_2 = self.backbone(xy, xy_shuffle)
            loss, _ = fdime_utils.compute_loss_ratio(self.divergence, "deranged", D_value_1=D_value_1,
                                             D_value_2=D_value_2,
                                             scores=None, buffer=None, alpha=self.alpha, device=self.device)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def on_train_epoch_start(self):
        self.train_epoch_start_time = time.time()
    
    def on_train_epoch_end(self):
        if self.train_epoch_start_time is not None:
            train_epoch_time = time.time() - self.train_epoch_start_time
            self.results["runtime_train"].append(train_epoch_time)
            
            # Log memory consumption
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024  # Resident Set Size
            vms_mb = memory_info.vms / 1024 / 1024  # Virtual Memory Size
            
            gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
            gpu_max_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            
            # Get system virtual memory
            vm = psutil.virtual_memory()
            
            self.results["memory"].append({
                "cpu_rss_mb": memory_mb, 
                "cpu_vms_mb": vms_mb,
                "system_vm_total_mb": vm.total / 1024 / 1024,
                "system_vm_available_mb": vm.available / 1024 / 1024,
                "system_vm_percent": vm.percent,
                "gpu_mb": gpu_memory_mb, 
                "gpu_max_mb": gpu_max_memory_mb
            })
            
            # Log CPU and GPU utilization
            cpu_util = self._get_cpu_utilization()
            gpu_util = self._get_gpu_utilization()
            self.results["cpu_utilization"].append(cpu_util)
            self.results["gpu_utilization"].append(gpu_util)

    def on_validation_epoch_start(self):
        self.val_epoch_start_time = time.time()
        self.mi_values = []
        self.val_step_times = []
        self.val_step_energy_start = []
        self.val_step_cpu_energy_start = []

    def on_validation_epoch_end(self):
        val_epoch_time = time.time() - self.val_epoch_start_time
        self.results["runtime_val"].append(val_epoch_time)
        
        # Calculate average inference time per forward pass
        if self.val_step_times:
            avg_inference_time = sum(self.val_step_times) / len(self.val_step_times)
            self.results["inference_runtime"].append(avg_inference_time)
        
        # Calculate average GPU energy dissipated per forward pass
        if self.val_step_energy_start:
            avg_energy = sum(self.val_step_energy_start) / len(self.val_step_energy_start)
            self.results["energy_dissipated"].append(avg_energy)
        
        # Calculate average CPU energy dissipated per forward pass
        if self.val_step_cpu_energy_start:
            avg_cpu_energy = sum(self.val_step_cpu_energy_start) / len(self.val_step_cpu_energy_start)
            self.results["cpu_energy"].append(avg_cpu_energy)
        
        if self.mi_values and len(self.mi_values) > 0:
            self.results["mi_history"].append({'epoch': self.current_epoch,\
                                                'mean_mi': torch.mean(torch.tensor(self.mi_values)).item(),
                                                'std_mi': torch.std(torch.tensor(self.mi_values)).item()})
    
    def _get_gpu_energy(self):
        """Get GPU energy in millijoules. Returns 0 if not available."""
        if self.gpu_handle is not None:
            try:
                # Get total energy consumed since driver load in millijoules
                return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.gpu_handle)
            except:
                return 0.0
        return 0.0
    
    def _get_cpu_energy(self):
        """Get CPU energy in microjoules. Returns 0 if not available."""
        if not self.cpu_energy_paths:
            return 0.0
        try:
            total_energy = 0.0
            for energy_file in self.cpu_energy_paths:
                with open(energy_file, 'r') as f:
                    # RAPL reports energy in microjoules
                    total_energy += int(f.read().strip())
            return total_energy
        except:
            return 0.0
    
    def _get_gpu_utilization(self):
        """Get GPU utilization percentage. Returns 0 if not available."""
        if self.gpu_handle is not None:
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                return utilization.gpu
            except:
                return 0.0
        return 0.0
    
    def _get_cpu_utilization(self):
        """Get CPU utilization percentage for this process."""
        try:
            # Get CPU utilization for this specific process
            # interval=None uses the value since last call
            return self.process.cpu_percent(interval=None)
        except:
            return 0.0

    def validation_step(self, batch, batch_idx):
        # Record start time and energy
        step_start_time = time.time()
        step_start_energy = self._get_gpu_energy()
        step_start_cpu_energy = self._get_cpu_energy()
        
        x, y = batch
        if 'infosedd' in self.estimator:
            mi = self._mutinfo_fn(self.backbone, x, y)
        elif 'minde' in self.estimator:
            if self.use_embeddings:
                x = self.embedding_layer(x).reshape(x.shape[0], -1)
                y = self.embedding_layer(y).reshape(y.shape[0], -1)
            mi = self._mutinfo_fn(self.backbone, x, y)
        else:
            y_shuffle = torch.index_select(y, 0, fdime_utils.derangement(list(range(y.shape[0])), self.device))
            xy_shuffle = torch.hstack((x, y_shuffle))
            xy = torch.hstack((x, y))
            D_value_1, D_value_2 = self.backbone(xy, xy_shuffle)
            _, R = fdime_utils.compute_loss_ratio(self.divergence, "deranged", D_value_1=D_value_1,
                                             D_value_2=D_value_2,
                                             scores=None, buffer=None, alpha=self.alpha, device=self.device)
            mi = torch.mean(torch.log(R))
        
        # Record end time and energy
        step_end_time = time.time()
        step_end_energy = self._get_gpu_energy()
        step_end_cpu_energy = self._get_cpu_energy()
        
        # Store metrics
        self.val_step_times.append(step_end_time - step_start_time)
        
        # GPU energy (in millijoules)
        if step_end_energy > 0 and step_start_energy > 0:
            self.val_step_energy_start.append(step_end_energy - step_start_energy)
        
        # CPU energy (in microjoules, convert to millijoules for consistency)
        if step_end_cpu_energy > 0 and step_start_cpu_energy > 0:
            cpu_energy_mj = (step_end_cpu_energy - step_start_cpu_energy) / 1000.0
            self.val_step_cpu_energy_start.append(cpu_energy_mj)
        
        self.log("MI", mi, on_step=False, on_epoch=True, prog_bar=True)
        self.mi_values.append(mi)

        return mi

    def configure_optimizers(self):
        if "minde" in self.estimator and self.use_embeddings:
            params = list(self.backbone.parameters()) + list(self.embedding_layer.parameters())
        else:
            params = self.backbone.parameters()
        optimizer = torch.optim.Adam(params, lr=0.002)
        return optimizer
    
    def __del__(self):
        """Cleanup pynvml on deletion."""
        if self.gpu_handle is not None:
            try:
                pynvml.nvmlShutdown()
            except:
                pass