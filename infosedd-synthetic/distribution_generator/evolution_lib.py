import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Optional, Callable, Union
from itertools import chain, combinations
from functools import reduce
from tqdm import tqdm
from copy import deepcopy
import scipy

class Agent:

    def __init__(self, dim_x: int, dim_y: int, params: Optional[np.array] = None, min_val=0) -> None:
        """
        num_rows: number of rows of the game
        params: initial parameters
        dim_x: alphabet size of first marginal
        dim_y: alphabet size of second marginal
        """
        if params is None:
            self._params = np.random.normal(size=(dim_x, dim_y))
        else:
            self._params = params
        self._fitness: float = -np.inf
        self.min_val = min_val
    
    def __lt__(self, other: 'Agent'):
        return self.fitness < other.fitness
    
    @property
    def distribution(self):
        dist = self.params - np.min(self.params)
        dist = dist/np.sum(dist)
        dist = dist * (1 - self.min_val) + self.min_val
        dist = dist/np.sum(dist)

        # print(np.sum(dist))

        """dist = np.exp(self.params)
        dist /= np.sum(dist)"""
        if np.any(dist < 0):
            print(np.min(self.params))
            print(self.params + np.min(self.params))
            raise ValueError("Distribution is negative:", dist)
        return dist
    
    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, p):
        self._params = p
    
    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, fitness: float):
        self._fitness = fitness
    
    def reset(self):
        self._fitness = -np.inf
    
    def __iadd__(self, other) -> None:
        self._params += other

class DistributionalAgent(Agent):

    def __init__(self, dist_name_x: str, dist_name_y: str, bins_x: Optional[int] = None, bins_y: Optional[int] = None, params: Optional[np.array] = None) -> None:
        self.dist_name_x = dist_name_x
        self.dist_name_y = dist_name_y
        self.bins_x = bins_x
        self.bins_y = bins_y

        if hasattr(scipy.stats, dist_name_x) and hasattr(scipy.stats, dist_name_y):
            self.dist_x = getattr(scipy.stats, dist_name_x)
            self.dist_y = getattr(scipy.stats, dist_name_y)
        else:
            raise NotImplementedError(f"One or both distributions {dist_name_x}, {dist_name_y} not implemented in scipy.stats")

        if params is None:
            self._params = np.random.normal(size=(self.dist_x.numargs + self.dist_y.numargs,))
        else:
            self._params = params
        self._fitness: float = -np.inf

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, p):
        self._params = p

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, fitness: float):
        self._fitness = fitness

    def reset(self):
        self._fitness = -np.inf

    def __iadd__(self, other) -> None:
        self._params += other

    @property
    def distribution(self):
        # Generate samples from the distributions using the parameters
        params_x = self._params[:self.dist_x.numargs]
        params_y = self._params[self.dist_x.numargs:]
        samples_x = self.dist_x.rvs(*params_x, size=1000)
        samples_y = self.dist_y.rvs(*params_y, size=1000)
        # Create a joint distribution
        joint_samples = np.vstack([samples_x, samples_y]).T
        hist, xedges, yedges = np.histogram2d(samples_x, samples_y, bins=(self.bins_x, self.bins_y))
        hist = hist / hist.sum()  # Normalize to create a probability distribution
        return hist

class EvolutionTask:

    """
    Evolutionary task to optimize the mutual information of a given distribution
    """

    def __init__(self, mutual_information: float, dim_x: int, dim_y: int, scale: float = 1.0, loc: float = 0.0, mean: np.array = None, cov: np.array = None, strategy='comma', mu=25, population_size=50, min_val=0) -> None:
        self.scale = scale
        self.loc = loc
        self.mu = mu
        self.strategy = strategy
        self.population_size = population_size
        self.mutual_information = mutual_information
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.mean = mean
        self.cov = cov
        self.min_val = min_val
    
    def mutate(self, agent: Agent) -> None:
        """
        Mutates according to Gaussian mutation
        """
        new_agent = deepcopy(agent)
        if self.mean is not None and self.cov is not None:
            delta = np.random.multivariate_normal(self.mean, self.cov).reshape(self.dim_x, self.dim_y)
            new_agent.params += delta
        new_agent.params += np.random.normal(size = agent.params.shape, loc=self.loc, scale=self.scale)
        return new_agent

    def calculate_mutual_information(self, pxy: np.array) -> float:
        """
        Calculates the mutual information of a given distribution
        """
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        if np.any(px == 0) or np.any(py == 0):
            return np.nan
        return np.sum(pxy*np.log(pxy/(px[:,None]*py[None,:])))

    def fitness(self, agent: Agent) -> float:
        m_info = self.calculate_mutual_information(agent.distribution)
        if m_info == np.nan:
            print("Mutinfo is nan", m_info)
            return -np.inf
        return -np.abs(m_info-self.mutual_information)
    
    def exploration(self) -> None:
        """
        Performs tournament to calculate fitness and select parents
        """
        mu = self.mu
        parents = np.partition(self.agents, self.population_size-mu)[self.population_size-mu:] # takes the top half parents in terms of fitness
        children = []
        if self.strategy == 'comma':
            num_children = self.population_size
        else:
            num_children = self.population_size-mu
        for _ in range(num_children):
            children.append(self.crossover(np.random.choice(parents), np.random.choice(parents)))
    
        if self.strategy == 'comma':
            self.agents = np.array(children)
        else:
            num_children = self.population_size-mu
        self.agents = np.concatenate((parents,children))
        self.agents = self.agents[np.argsort(self.agents)]
        if len(self.agents) > self.population_size:
            self.agents = self.agents[:self.population_size]
    
    def crossover(self, a1: Agent, a2: Agent) -> Agent:
        """
        Given two agents it randomly selects the parameters between the two
        """
        new_params: List[float] = []
        for i in range(len(a1.params)):
            if np.random.normal() > 0:
                new_params.append(a1.params[i])
            else:
                new_params.append(a2.params[i])
        return Agent(self.dim_x, self.dim_y, np.array(new_params), min_val=self.min_val)

    def compute_fitness(self) -> None:
        for agent in self.agents:
            agent.fitness = self.fitness(agent)

    def exploitation(self) -> None:
        """
        Mutates parameters of the agents with Gaussian mutation
        """
        mu = self.mu
        parents = np.partition(self.agents, self.population_size-mu)[self.population_size-mu:] # takes the top half parents in terms of fitness
        children = []
        if self.strategy == 'comma':
            num_children = self.population_size
        else:
            num_children = self.population_size-mu
        for _ in range(num_children):
            children.append(self.mutate(np.random.choice(parents)))
        if self.strategy == 'comma':
            self.agents = np.array(children)
        else:
            self.agents = np.concatenate((parents,children))
    
    def reset_fitness(self) -> None:
        for agent in self.agents:
            agent.reset()

    def train(self, n_generations=100, temperature = 0.5, exploitation_only = False):
        """
        Training loop, the temperature defines the transition from an exploration prevalent strategy to an exploitation prevalent strategy.
        In particular exploitation is performed with probability P[X>(generation/tot_generations)^t], while exploitation is performed with P[X<(generation/tot_generations)^t], where X is uniformly distributed between 0 and 1
        """
        self.agents: List[Agent] = [Agent(self.dim_x, self.dim_y, min_val=self.min_val) for _ in range(self.population_size)]
        self.best_agent = None
        pbar = tqdm(range(n_generations))
        self.compute_fitness()
        for gen in pbar:            
            self.best_agent = max(self.agents, key = lambda x: x.fitness)
            pbar.set_description(f"Best fitness: {self.best_agent.fitness}")
            if np.random.uniform(0,1) > np.power(gen/n_generations, temperature) or exploitation_only:
                #exploitation
                self.exploitation()
                self.compute_fitness()
            if np.random.uniform(0,1) < np.power(gen/n_generations, temperature) and not exploitation_only:
                #exploration
                self.exploration()
                self.compute_fitness()
            best_agent = max(self.agents, key = lambda x: x.fitness)
            if best_agent.fitness > self.best_agent.fitness:
                self.best_agent = best_agent
            # print(self.best_agent.fitness)
    
    @property
    def best_agent(self) -> Agent:
        return self._best_agent

    @best_agent.setter
    def best_agent(self, a: Optional[Agent]) -> Agent:
        self._best_agent = a