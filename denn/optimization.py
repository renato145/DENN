from .imports import *
from .callbacks import *
from .utils import *

__all__ = ['Individual', 'Population', 'Optimization']

@dataclass
class Individual:
    dimensions:int
    lower_limit:float
    upper_limit:float
    idx:Optional[int]=None
    gen:Optional[int]=None
    fitness_value:Optional[float]=None
    constraints:Optional[Collection[float]]=None
    constraints_sum:Optional[float]=None
    is_feasible:Optional[bool]=False

    @classmethod
    def new_random(cls, dimensions=10, lower_limit=-5, upper_limit=5):
        res = cls(dimensions, lower_limit, upper_limit)
        res.func = partial(np.random.uniform, low=lower_limit, high=upper_limit, size=dimensions)
        res.refresh()
        return res

    def refresh(self, *args, **kwargs):
        self.data = self.func(*args, **kwargs)

    def assign_idx(self, idx):
        self.idx = idx
        return self

    def clip_limits(self):
        self.data = self.data.clip(self.lower_limit, self.upper_limit)

    def clone(self):
        return self.__class__(dimensions=self.dimensions, lower_limit=self.lower_limit, upper_limit=self.upper_limit,
                              idx=self.idx, gen=self.gen, fitness_value=self.fitness_value, constraints=self.constraints,
                              constraints_sum=self.constraints_sum, is_feasible=self.is_feasible)

    def copy_from(self, indiv):
        self.dimensions = indiv.dimensions
        self.lower_limit = indiv.lower_limit
        self.upper_limit = indiv.upper_limit
        self.idx = indiv.idx
        self.gen = indiv.gen
        self.fitness_value = indiv.fitness_value
        self.constraints = indiv.constraints
        self.constraints_sum = indiv.constraints_sum
        self.is_feasible = indiv.is_feasible

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
        self.individuals = [self.new_individual(*args, **kwargs).assign_idx(i) for i in range(self.n)]

    def __call__(self, func, pbar=None):
        return [func(individual) for individual in progress_bar(self.individuals, parent=pbar)]

@dataclass
class Optimization:
    population:Population
    get_fitness:Callable
    get_constraints:Union[Callable,Collection[Callable]]=None
    constraint_params:Union[Any,Collection[Any]]=None
    time_params:Union[Any,Collection[Any]]=None
    frequency:Optional[int]=None
    CR:float=0.3
    beta_min:float=0.2
    beta_max:float=0.8
    max_evals:Optional[int]=None
    metrics:Collection[str]=('fitness',)
    callbacks:Collection[Callback]=None

    def __post_init__(self):
        self.get_constraints = listify(self.get_constraints)
        self.have_constraints = len(self.get_constraints)>0
        if self.have_constraints:
            if self.constraint_params is None: raise Exception('You need to specify `constraint_params` when giving `get_constraints`.')
            self.constraint_params = listify(self.constraint_params)
            assert len(self.get_constraints) == len(self.constraint_params)

        self.time_params = listify(self.time_params)
        self.max_times = len(self.time_params)
        self.have_time = self.max_times > 0
        if self.have_time:
            if self.frequency is None: raise Exception('You need to specify `frequency` when giving `time_params`.')

        self.callbacks = [Recorder(self)] + [e(self) for e in listify(self.callbacks)]
        self.cb_handler = CallbackHandler(self, self.callbacks)

    @property
    def state_dict(self): return self.cb_handler.state_dict
    @property
    def best(self): return self.state_dict['best']

    def _get_best(self, indiv1, indiv2):
        return indiv1 if indiv1.fitness_value <= indiv2.fitness_value else indiv2

    def _get_lowest_constraint(self, indiv1, indiv2):
        return indiv1 if indiv1.constraints_sum <= indiv2.constraints_sum else indiv2

    def _get_best_constraints(self, indiv1, indiv2):
        'Feasibility rules method.'
        if indiv1.is_feasible:
            if indiv2.is_feasible: return self._get_best(indiv1, indiv2)              # Both are feasible
            else                 : return indiv1                                      # Only indiv1 is feasible
        else:
            if indiv2.is_feasible: return indiv2                                      # Only indiv2 is feasible
            else                 : return self._get_lowest_constraint(indiv1, indiv2) # Both are unfeasible

    def get_best(self, indiv1, indiv2):
        fn = self._get_best_constraints if self.have_constraints else self._get_best
        return fn(indiv1, indiv2)

    def get_sum_constraints(self, indiv):
        constraints = np.array(indiv.constraints)
        constraints[constraints<0] = 0
        return sum(constraints)

    def eval_feasibility(self, indiv):
        return indiv.constraints_sum == 0

    def _evolve(self, indiv):
        dims = indiv.dimensions
        jrand = np.random.randint(dims)
        crs = np.argwhere(np.random.rand(dims) < self.CR)[:,0].tolist()
        picked_dims = get_unique(crs + [jrand])

        if len(picked_dims) > 0:
            picked = [e.data[picked_dims] for e in pick_n_but(3, indiv.idx, self.population)]
            F = np.random.uniform(self.beta_min, self.beta_max, size=len(picked_dims))
            new_data = picked[0] + F*picked[1] - picked[2]
            indiv.data[picked_dims] = new_data
            indiv.clip_limits()

    def evolve(self, indiv):
        try:
            self.cb_handler.on_evolve_begin()
            self._evolve(indiv)
        except CancelEvolveException: self.cb_handler.on_cancel_evolve()
        finally: self.cb_handler.on_evolve_end()

    def eval_fitness(self, indiv):
        fitness = None
        try:
            self.cb_handler.on_fitness_begin()
            fitness = self.get_fitness(indiv)

        except CancelFitnessException: self.cb_handler.on_fitness_cancel()
        finally:
            evals = self.cb_handler.on_fitness_end(fitness=fitness)
            self.change_time(evals)

    def change_time(self, evals):
        if self.have_time:
            if evals % self.frequency == 0:
                self.cb_handler.on_time_change()

    def get_each_constraint(self, indiv):
        try:
            for fn,b in zip(self.get_constraints,self.constraint_params):
                b = self.cb_handler.on_each_constraint_begin(b=b)
                constraint = fn(indiv, b)
                self.cb_handler.on_each_constraint_end(constraint=constraint)
        except CancelEachConstraintException: self.cb_handler.on_cancel_each_constraint()

    def eval_constraints(self, indiv):
        try:
            self.cb_handler.on_constraints_begin()
            self.get_each_constraint(indiv)     

        except CancelConstraintsException: self.cb_handler.on_cancel_constraints()
        finally: self.cb_handler.on_constraints_end()

    def process_individual(self, indiv):
        try:
            self.cb_handler.on_individual_begin(indiv=indiv)
            self.evolve(indiv)
            self.eval_fitness(indiv)
            if self.have_constraints: self.eval_constraints(indiv)

        except CancelGenException: self.cb_handler.on_cancel_individual()
        finally:
            new_indiv = self.cb_handler.on_individual_end()
            indiv.copy_from(new_indiv)

    def run_one_gen(self, pbar=None):
        try:
            self.cb_handler.on_gen_begin()
            for indiv in progress_bar(self.population, parent=pbar): self.process_individual(indiv)

        except CancelGenException: self.cb_handler.on_cancel_gen()
        finally: self.cb_handler.on_gen_end()

    def run(self, generations=100, show_graph=True, update_each=10):
        pbar = master_bar(range(generations))
        try:
            self.cb_handler.on_run_begin(generations, pbar, self.metrics, self.max_evals, self.max_times, self.frequency,
                                         show_graph=show_graph, update_each=update_each)
            for gen in pbar: self.run_one_gen(pbar=pbar)
                
        except CancelRunException: self.cb_handler.on_cancel_run()
        finally: self.cb_handler.on_run_end()

