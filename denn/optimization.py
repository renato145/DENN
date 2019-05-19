from .imports import *
from .utils import *

__all__ = ['Individual', 'Population', 'Optimization']

@dataclass
class Individual:
    dimensions:int
    lower_limit:float
    upper_limit:float

    @classmethod
    def new_random(cls, dimensions=10, lower_limit=-5, upper_limit=5):
        res = cls(dimensions, lower_limit, upper_limit)
        res.func = partial(np.random.uniform, low=lower_limit, high=upper_limit, size=dimensions)
        res.refresh()
        return res

    def refresh(self, *args, **kwargs):
        self.data = self.func(*args, **kwargs)

@dataclass
class Population:
    n:int
    dimension:int

    def __getitem__(self, idx): return self.individuals[idx]
    def __setitem__(self, idx, val): self.individuals[idx] = val
    def __len__(self): return len(self.individuals)
    def __iter__(self): return iter(self.individuals)

    @classmethod
    def new_random(cls, n=20, dimension=10, lower_limit=-5, upper_limit=5):
        res = cls(n, dimension)
        res.new_individual = partial(Individual.new_random, dimension, lower_limit, upper_limit)
        res.refresh()
        return res

    def refresh(self, *args, **kwargs):
        self.individuals = [self.new_individual(*args, **kwargs) for _ in range(self.n)]

    def __call__(self, func, pbar=None):
        return [func(individual) for individual in progress_bar(self.individuals, parent=pbar)]

@dataclass
class Optimization:
    population:Population
    get_fitness:Callable
    CR:float=0.3
    betamin:float=0.2
    betamax:float=0.8
    max_eval:Optional[int]=None

    def __post_init__(self):
        self.gen = 0
        self.evals = 0
        self.bests = []
        self.eval_fitness()

    @property
    def best(self): return min(self.bests)

    def eval_fitness(self, pbar=None):
        for indiv in progress_bar(self.population, parent=pbar):
            if self.max_eval is not None:
                if self.evals >= self.max_eval: break

            self.get_fitness(indiv)
            indiv.gen = self.gen
            self.evals += 1
        self.bests.append(self.get_best().fitness_value)

    def evolve_one(self, idx):
        res = 0
        indiv = self.population[idx]
        dims = indiv.dimensions
        jrand = np.random.randint(dims)
        crs = np.argwhere(np.random.rand(dims) < self.CR)[:,0].tolist()
        picked_dims = get_unique(crs + [jrand])

        if len(picked_dims) > 0:
            picked = [e.data[picked_dims] for e in pick_n_but(3, idx, self.population)]
            F = np.random.uniform(self.betamin, self.betamax, size=len(picked_dims))
            new_data = picked[0] + F*picked[1] - picked[2]
            indiv.data[picked_dims] = new_data.clip(indiv.lower_limit, indiv.upper_limit)

    def evolve_all(self):
        for i in range(len(self.population)): self.evolve_one(i)

    def get_best(self):
        idx = np.argmin([e.fitness_value for e in self.population])
        return self.population[idx]

    def run_one_gen(self, pbar=None):
        self.gen += 1
        self.evolve_all()
        self.eval_fitness(pbar)

    def run(self, generations=100):
        pbar = master_bar(range(generations))
        for gen in pbar: self.run_one_gen(pbar=pbar)

