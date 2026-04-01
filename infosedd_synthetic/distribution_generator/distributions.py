try:
    from .evolution_lib import *
except ImportError:
    from distribution_generator.evolution_lib import *
from scipy.stats import rv_discrete
from scipy.stats._multivariate import multi_rv_frozen

from dataclasses import dataclass

import itertools

class DistributionManager:
    def __init__(self):
        self.distribution_config = None
        self.evolution_task = None
        self.distribution = None

    def __call__(self, 
                 mutual_information: float,
                 dim_x: int,
                 dim_y: int,
                 seq_length_x: int,
                 seq_length_y: int,
                 scale: float = 1.0,
                 loc: float = 0.0,
                 mean: np.array = None,
                 cov: np.array = None,
                 strategy='comma',
                 mu=25,
                 population_size=50,
                 n_generations=100,
                 min_val=1e-3,
                 noise_dimensions=0,
                 force_retrain=False,
                 xy_map=None,
                 noise_rv_x=None,
                 noise_rv_y=None,
                 fast=True) -> None:

        self.fast = fast
        if not force_retrain and self.distribution_config is not None and self.distribution_config == DistributionConfig(mutual_information, dim_x, dim_y, seq_length_x, seq_length_y, min_val):
            return self.rv
        self.distribution_config = DistributionConfig(mutual_information, dim_x, dim_y, seq_length_x, seq_length_y, min_val, xy_map, noise_rv_x, noise_rv_y, fast)
        self.train_tasks(scale, loc, mean, cov, strategy, mu, population_size, min_val, n_generations)
        if not fast:
            self.rv = JointDiscrete(self.distribution, vocabulary_x=self.possible_x_sequences, vocabulary_y=self.possible_y_sequences, noise_dimensions=noise_dimensions)
        else:
            self.rv = FastJointDiscrete(self.distributions, noise_dimensions=noise_dimensions, xy_map=xy_map, noise_rv_x=noise_rv_x, noise_rv_y=noise_rv_y)

        return self.rv

    def train_tasks(self, scale, loc, mean, cov, strategy, mu, population_size, min_val, n_generations):
        """
        Function to train many tasks to reach the desired mutual information by stacking random variables
        It only generates distributions with pairwise dependencies between tokens
        """

        seq_length_x = self.distribution_config.seq_length_x
        seq_length_y = self.distribution_config.seq_length_y
        mutual_information = self.distribution_config.mutual_information
        dim_x = self.distribution_config.dim_x
        dim_y = self.distribution_config.dim_y

        assert self.distribution_config.seq_length_x == self.distribution_config.seq_length_y, "Currently only supports same sequence length for x and y"

        relevant_dims = min(seq_length_x, seq_length_y)
        unit_mutinfo = mutual_information / relevant_dims
        self.evolution_tasks = []
        self.distributions = []
        for dim in range(relevant_dims):
            task = EvolutionTask(unit_mutinfo, dim_x, dim_y, scale, loc, mean, cov, strategy, mu, population_size, min_val)
            task.train(n_generations)
            self.distributions.append(task.best_agent.distribution)
        
        if not self.fast:
        
            dist = np.zeros((dim_x**seq_length_x, dim_y**seq_length_y))

            self.possible_x_sequences = np.array(list(itertools.product(range(dim_x), repeat=relevant_dims)))
            self.possible_y_sequences = np.array(list(itertools.product(range(dim_y), repeat=relevant_dims)))
            
            for idx_x, x in enumerate(self.possible_x_sequences):
                for idx_y, y in enumerate(self.possible_y_sequences):
                    joint_prob = 1
                    for i in range(relevant_dims):
                        joint_prob *= self.distributions[i][x[i], y[i]]
                    dist[idx_x, idx_y] = joint_prob
            self.distribution = dist

@dataclass
class DistributionConfig:
    mutual_information: float
    dim_x: int
    dim_y: int
    seq_length_x: int
    seq_length_y: int
    min_val: float
    xy_map: callable = None
    noise_rv_x: callable = None
    noise_rv_y: callable = None
    fast: bool = True

class FastJointDiscrete(multi_rv_frozen):
    
    def __init__(self, distributions, *args, noise_dimensions=0, xy_map=None, noise_rv_x=None, noise_rv_y=None, **kwargs):
        self.distributions = distributions
        self.noise_dimensions = noise_dimensions
        self.xy_map = xy_map
        self.noise_rv_x = noise_rv_x
        self.noise_rv_y = noise_rv_y
        self._hidden_values = [np.arange(len(d.flatten())) for d in distributions]
        self._hidden_univariates = [rv_discrete(name=f"hidden_univariate_{i}", values=(self._hidden_values[i], self.distributions[i].flatten())) for i in range(len(distributions))]
    
    def cantor_map(self, x, y):
        res = (x + y)*(x + y + 1)/2 + y
        res = np.array(res, dtype=int)
        return res
    
    def rank_array(self, arr):
        """Map each unique value in arr to a contiguous rank starting from 0."""
        flat = arr.ravel()                          # flatten, doesn't copy
        unique, inv = np.unique(flat, return_inverse=True)
        return inv.reshape(arr.shape)     

    def rvs(self, *args, **kwargs):
        if len(args) > 1:
            raise NotImplementedError("Different sizes not implemented, pass keyworkd argument size instead")
        elif len(args) == 1:
            kwargs['size'] = args[0]
            args = []
        samples = [h_rv.rvs(*args, **kwargs) for h_rv in self._hidden_univariates]
        samples = [np.unravel_index(s, d.shape) for s, d in zip(samples, self.distributions)]
        samples = [np.stack(s, axis=1) for s in samples]
        X = [s[:,0].reshape(-1,1) for s in samples]
        X = np.concatenate(X, axis=1)
        Y = [s[:,1].reshape(-1,1) for s in samples]
        Y = np.concatenate(Y, axis=1)

        if self.noise_rv_x is not None: # This increases the size of the alphabet
            X_noise = self.noise_rv_x.rvs(size=X.shape)
            X_noise = X_noise - np.min(X_noise)
            X = X - np.min(X)

            X = self.cantor_map(X, X_noise)
            X = self.rank_array(X)
        
        if self.noise_rv_y is not None: # This increases the size of the alphabet
            Y_noise = self.noise_rv_y.rvs(size=Y.shape)
            Y_noise = Y_noise - np.min(Y_noise)
            Y = Y - np.min(Y)

            Y = self.cantor_map(Y, Y_noise)
            Y = self.rank_array(Y)
        
        if self.xy_map is not None: # This increases the length of the sequence
            X = X.reshape(*X.shape, 1)
            X = np.apply_along_axis(self.xy_map, -1, X)
            X = X.reshape(X.shape[0], -1)

            Y = Y.reshape(*Y.shape, 1)
            Y = np.apply_along_axis(self.xy_map, -1, Y)
            Y = Y.reshape(Y.shape[0], -1)
        
        if self.noise_dimensions > 0:
            dim_x = np.max(X)
            dim_y = np.max(Y)
            
            X_noise = np.random.randint(low=0, high=dim_x, size=(X.shape[0], self.noise_dimensions))
            Y_noise = np.random.randint(low=0, high=dim_y, size=(Y.shape[0], self.noise_dimensions))

            X = np.concatenate((X, X_noise), axis=1)
            Y = np.concatenate((Y, Y_noise), axis=1)

        return X, Y
    
    @property
    def entropy(self):
        return -np.sum([d.flatten() * np.log(d.flatten()) for d in self.distributions])
    
    @property
    def mutual_information(self):
        return np.sum([self._mutual_information(d) for d in self.distributions])

    def _mutual_information(self, joint_dist):
        marginal_x = np.sum(joint_dist, axis=1)
        marginal_y = np.sum(joint_dist, axis=0)
        return np.sum(joint_dist * np.log(joint_dist / np.outer(marginal_x, marginal_y)))

class JointDiscrete(multi_rv_frozen):

    def __init__(self, joint_dist, *args, vocabulary_x=None, vocabulary_y=None, noise_dimensions=0, **kwargs):
        self.joint_dist = joint_dist
        self.vocab_x = vocabulary_x
        self.vocab_y = vocabulary_y
        self.noise_dimensions = noise_dimensions
        pmf = joint_dist.flatten()
        values = np.arange(len(pmf))
        self._hidden_univariate = rv_discrete(name="hidden_univariate", values=(values, pmf))
        super().__init__(*args, **kwargs)
    
    def rvs(self, *args, **kwargs):
        if len(args) > 1:
            raise NotImplementedError("Different sizes not implemented, pass keyworkd argument size instead")
        elif len(args) == 1:
            kwargs['size'] = args[0]
            args = []
        samples = self._hidden_univariate.rvs(*args, **kwargs)
        samples = np.unravel_index(samples, self.joint_dist.shape)
        samples = np.stack(samples, axis=1)
        
        X = samples[:,0].reshape(-1,1)
        if self.vocab_x is not None:
            X = self.vocab_x[X]
        
        Y = samples[:,1].reshape(-1,1)
        if self.vocab_y is not None:
            Y = self.vocab_y[Y]
        
        if self.noise_dimensions > 0:
            dim_x = np.max(self.vocab_x)
            dim_y = np.max(self.vocab_y)
            
            X_noise = np.random.randint(low=0, high=dim_x, size=(X.shape[0], self.noise_dimensions))
            Y_noise = np.random.randint(low=0, high=dim_y, size=(Y.shape[0], self.noise_dimensions))

            X = np.concatenate((X, X_noise), axis=1)
            Y = np.concatenate((Y, Y_noise), axis=1)
        
        return X, Y
    
    @property
    def entropy(self):
        return -np.sum(self.joint_dist * np.log(self.joint_dist))
    
    @property
    def mutual_information(self):
        marginal_x = np.sum(self.joint_dist, axis=1)
        marginal_y = np.sum(self.joint_dist, axis=0)
        return np.sum(self.joint_dist * np.log(self.joint_dist / np.outer(marginal_x, marginal_y)))

distribution_manager = DistributionManager()

def get_rv(mutual_information: float,
                     dim: int = None,
                     dim_x: int = None,
                     dim_y: int = None,
                     seq_length_x: int = None,
                     seq_length_y: int = None,
                     seq_length: int = None,
                     scale: float = 1.0,
                     loc: float = 0.0,
                     mean: np.array = None,
                     cov: np.array = None,
                     strategy: str ='comma',
                     mu: int = 25,
                     population_size: int = 50,
                     n_generations: int = 100,
                     min_val: float = 0,
                     noise_dimensions: int = 0,
                     force_retrain: bool = False,
                     xy_map=None,
                     noise_rv=None,
                     noise_rv_x=None,
                     noise_rv_y=None,
                     fast: bool = True,
                     seed: int = 42) -> None:
    
    if dim is not None:
        assert dim_x == dim_y == dim or (dim_x is None and dim_y is None), "If dim is passed, dim_x and dim_y must be equal to dim or set to None"
        dim_x = dim_y = dim
    
    if seq_length is not None:
        assert seq_length_x == seq_length_y == seq_length or (seq_length_x is None or seq_length_y is None), "If seq_length is passed, seq_length_x and seq_length_y must be equal to seq_length or set to None"
        seq_length_x = seq_length_y = seq_length
    
    if noise_rv is not None:
        assert noise_rv_x is None and noise_rv_y is None, "If noise_rv is passed, noise_rv_x and noise_rv_y must be set to None"
        noise_rv_x = noise_rv_y = noise_rv
    
    assert mutual_information <= min(seq_length_x*np.log(dim_x), seq_length_y*np.log(dim_y)), f"Mutual information is too high for the given dimensions, max is {min(seq_length_x*np.log(dim_x), seq_length_y*np.log(dim_y))} nats"
    assert mutual_information >= 0, "Mutual information must be non-negative"

    np.random.seed(seed)
    custom_rv = distribution_manager(mutual_information,
                                dim_x,
                                dim_y,
                                seq_length_x,
                                seq_length_y,
                                scale,
                                loc,
                                mean,
                                cov,
                                strategy,
                                mu,
                                population_size,
                                n_generations,
                                min_val,
                                noise_dimensions,
                                force_retrain,
                                xy_map,
                                noise_rv_x,
                                noise_rv_y,
                                fast)
    return custom_rv