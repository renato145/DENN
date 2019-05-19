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
    CR:float=0.3
    betamin:float=0.2
    betamax:float=0.8
    max_eval:Optional[int]=None
    metrics:Collection[str]=('fitness',)
    callbacks:Collection[Callback]=None

    def __post_init__(self):
        self.gen = 0
        self.evals = 0
        self.bests = []
        self.callbacks = [e(self) for e in listify(self.callbacks)]
        self.cb_handler = CallbackHandler(self, self.callbacks)
        self.eval_fitness()

    @property
    def best(self): return min(self.bests)

    @property
    def best_with_idx(self):
        idx = np.argmin(self.bests)
        return (idx, self.bests[idx])

    def eval_fitness(self, pbar=None):
        try:
            self.cb_handler.on_fitness_all_begin()
            for indiv in progress_bar(self.population, parent=pbar):
                self.cb_handler.on_fitness_one_begin()
                if self.max_eval is not None: # remove later
                    if self.evals >= self.max_eval: break

                self.get_fitness(indiv)
                self.cb_handler.on_fitness_one_end()
                indiv.gen = self.gen # needs modification ex: self.cb_handler.state_dict['gen']
                self.evals += 1 # remove later

            self.bests.append(self.get_best().fitness_value) # should be on recorder
        except CancelFitnessException: self.cb_handler.on_fitness_cancel()
        finally: self.cb_handler.on_fitness_all_end()

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
        try:
            self.cb_handler.on_evol_all_begin()
            for i in range(len(self.population)):
                self.cb_handler.on_evol_one_begin()
                self.evolve_one(i)
                self.cb_handler.on_evol_one_end()

        except CancelEvolException: self.cb_handler.on_cancel_evol()
        finally: self.cb_handler.on_evol_all_end()


    def get_best(self):
        idx = np.argmin([e.fitness_value for e in self.population])
        return self.population[idx]

    def run_one_gen(self, pbar=None):
        self.gen += 1 # add to recorder
        self.evolve_all()
        self.eval_fitness(pbar)

    def run(self, generations=100, show_graph=True, update_each=10, **kwargs):
        pbar = master_bar(range(generations))
        pbar.names = self.metrics
        try:
            self.cb_handler.on_run_begin(generations)
            for gen in pbar:
                self.cb_handler.on_gen_begin()
                self.run_one_gen(pbar=pbar)
                if show_graph and (gen+1)%update_each==0: pbar.update_graph(self._get_plot_data(), **kwargs)
                self.cb_handler.on_gen_end()
            else:
                if show_graph: pbar.update_graph(self._get_plot_data(), **kwargs)
        except CancelRunException: self.cb_handler.on_cancel_run()
        finally: self.cb_handler.on_run_end()

    def _get_plot_data(self): # should be on recorder
        x = np.arange(len(self.bests))
        return [[x,self.bests]]

    def plot(self, ax=None, alpha=0.75, size=100, color='green', figsize=(8,5)): # should be on recorder
        if ax is None: fig,ax = plt.subplots(1, 1, figsize=figsize)
        for g,n in zip(self._get_plot_data(),self.metrics): ax.plot(*g, alpha=alpha, label=n)
        idx,best = self.best_with_idx
        ax.scatter(idx, best, s=size, c=color)
        ax.legend(loc='upper right')
        ax.set_title(f'Best fitness value: {best:.2f}\nGeneration: {idx}')

