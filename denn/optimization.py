from .imports import *
from .callbacks import *
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
    get_constraint:Callable=None
    constraint_param:float=None
    CR:float=0.3
    betamin:float=0.2
    betamax:float=0.8
    max_evals:Optional[int]=None
    metrics:Collection[str]=('fitness',)
    callbacks:Collection[Callback]=None

    def __post_init__(self):
        self.have_constraint = self.get_constraint is not None
        if self.have_constraint:
            if constraint_param is None: raise Exception('You need to especify a `constraint_param`.')

        self.callbacks = [Recorder(self)] + [e(self) for e in listify(self.callbacks)]
        self.cb_handler = CallbackHandler(self, self.callbacks)

    def eval_fitness(self, pbar=None):
        try:
            self.cb_handler.on_fitness_all_begin()
            for indiv in progress_bar(self.population, parent=pbar):
                self.cb_handler.on_fitness_one_begin(indiv=indiv)
                self.get_fitness(indiv)
                self.cb_handler.on_fitness_one_end(indiv=indiv)

        except CancelFitnessException: self.cb_handler.on_fitness_cancel()
        finally: self.cb_handler.on_fitness_all_end(best=self.get_best().fitness_value)

    def eval_constraint(self, pbar=None):
        pass

    def evolve_one(self, idx):
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

    def evolve_all(self, pbar=None):
        try:
            self.cb_handler.on_evol_all_begin()
            for i in progress_bar(range(len(self.population)), parent=pbar):
                self.cb_handler.on_evol_one_begin()
                self.evolve_one(i)
                self.cb_handler.on_evol_one_end()

        except CancelEvolException: self.cb_handler.on_cancel_evol()
        finally: self.cb_handler.on_evol_all_end()


    def get_best(self):
        idx = np.argmin([e.fitness_value for e in self.population])
        return self.population[idx]

    def run_one_gen(self, pbar=None):
        self.evolve_all(pbar)
        self.eval_fitness(pbar)
        self.eval_constraint(pbar)

    def run(self, generations=100, show_graph=True, update_each=10):
        pbar = master_bar(range(generations))
        try:
            self.cb_handler.on_run_begin(generations, pbar, self.metrics, self.max_evals)
            for gen in pbar:
                self.cb_handler.on_gen_begin()
                self.run_one_gen(pbar=pbar)
                self.cb_handler.on_gen_end(update_each=update_each, show_graph=show_graph)
        except CancelRunException: self.cb_handler.on_cancel_run()
        finally: self.cb_handler.on_run_end()
