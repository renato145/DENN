from .imports import *
from .callbacks import *
from .utils import *

__all__ = ['Individual', 'Population', 'Optimization', 'Runs']

@dataclass
class Individual:
    dimensions:int
    lower_limit:float
    upper_limit:float
    idx:Optional[int]=None
    gen:Optional[int]=None
    time:Optional[int]=None
    fitness_value:Optional[float]=None
    constraints:Optional[Collection[float]]=None
    constraints_sum:Optional[float]=None
    is_feasible:Optional[bool]=False
    data:Optional[np.ndarray]=None

    def __eq__(self, other:'Individual')->bool:
        if not all(self.data == other.data): return False
        if not (self.fitness_value == other.fitness_value): return False
        if not all(i==j for i,j in zip(self.constraints,other.constraints)): return False
        return True

    @classmethod
    def new_random(cls, dimensions:int=10, lower_limit:float=-5, upper_limit:float=5)->'Individual':
        res = cls(dimensions, lower_limit, upper_limit)
        res.func = partial(np.random.uniform, low=lower_limit, high=upper_limit, size=dimensions)
        res.refresh()
        return res

    def zero_data(self)->None:
        self.data[:] = 0

    def refresh(self, *args:Any, **kwargs:Any)->None:
        self.data = self.func(*args, **kwargs)
        self.fitness_value = None
        self.constraints = None
        self.constraints_sum = None
        self.is_feasible = False

    def assign_idx(self, idx:int)->'Individual':
        self.idx = idx
        return self

    def clip_limits(self)->None:
        self.data = self.data.clip(self.lower_limit, self.upper_limit)

    def clone(self)->'Individual':
        return self.__class__(dimensions=self.dimensions, lower_limit=self.lower_limit, upper_limit=self.upper_limit,
                              idx=self.idx, gen=self.gen, time=self.time, fitness_value=self.fitness_value, constraints=self.constraints,
                              constraints_sum=self.constraints_sum, is_feasible=self.is_feasible, data=self.data.copy())

    def copy_from(self, indiv:'Individual')->None:
        self.dimensions = indiv.dimensions
        self.lower_limit = indiv.lower_limit
        self.upper_limit = indiv.upper_limit
        self.idx = indiv.idx
        self.gen = indiv.gen
        self.time = indiv.time
        self.fitness_value = indiv.fitness_value
        self.constraints = indiv.constraints
        self.constraints_sum = indiv.constraints_sum
        self.is_feasible = indiv.is_feasible
        self.data = indiv.data.copy()

@dataclass
class Population:
    n:int
    dimension:int

    def __getitem__(self, idx:int)->Individual: return self.individuals[idx]
    def __setitem__(self, idx:int, val:Individual)->None: self.individuals[idx] = val
    def __len__(self)->int: return len(self.individuals)
    def __iter__(self): return iter(self.individuals)

    @classmethod
    def new_random(cls, n:int=20, dimension:int=10, lower_limit:float=-5, upper_limit:float=5)->Collection[Individual]:
        res = cls(n, dimension)
        res.new_individual = partial(Individual.new_random, dimension, lower_limit, upper_limit)
        res.individuals = [res.new_individual().assign_idx(i) for i in range(n)]
        return res

    def refresh(self)->None:
        for indiv in self.individuals: indiv.refresh()

    def __call__(self, func:Callable, pbar:Optional[PBar]=None)->Collection[Any]:
        return [func(individual) for individual in progress_bar(self.individuals, parent=pbar)]

@dataclass
class Optimization:
    population:Population
    get_fitness:Callable
    get_constraints:Union[Callable,Collection[Callable]]=None
    fitness_params:Optional[Any]=None
    constraint_params:Optional[Any]=None
    max_times:Optional[int]=None
    frequency:Optional[int]=None
    CR:float=0.3
    beta_min:float=0.2
    beta_max:float=0.8
    max_evals:Optional[int]=None
    time_change_detect:bool=True
    time_change_pcts:Collection[float]=(0.0,0.5)
    callbacks:Collection[Callback]=None

    def __post_init__(self):
        self.get_constraints = listify(self.get_constraints)
        self.have_constraints = len(self.get_constraints)>0
        if self.have_constraints:
            if self.constraint_params is None: raise Exception('You need to specify `constraint_params` when giving `get_constraints`.')
            assert len(self.get_constraints) == len(self.constraint_params)

        self.have_time = self.max_times is not None
        if self.have_time:
            if self.frequency is None: raise Exception('You need to specify `frequency` when giving `max_times`.')
            self.time_change_checks = [int(self.population.n*e) for e in self.time_change_pcts]

        self.callbacks = [Recorder(self)] + [e(self) for e in listify(self.callbacks)]
        self.cb_handler = CallbackHandler(self, self.callbacks)

    @property
    def state_dict(self)->dict: return self.cb_handler.state_dict
    @property
    def best(self)->Individual: return self.state_dict['best']
    @property
    def time(self)->int: return self.state_dict['time']

    def _get_best(self, indiv1:Individual, indiv2:Individual)->Individual:
        return indiv1 if indiv1.fitness_value <= indiv2.fitness_value else indiv2

    def _get_lowest_constraint(self, indiv1:Individual, indiv2:Individual)->Individual:
        return indiv1 if indiv1.constraints_sum <= indiv2.constraints_sum else indiv2

    def _get_best_constraints(self, indiv1:Individual, indiv2:Individual)->Individual:
        'Feasibility rules method.'
        if indiv1.is_feasible:
            if indiv2.is_feasible: return self._get_best(indiv1, indiv2)              # Both are feasible
            else                 : return indiv1                                      # Only indiv1 is feasible
        else:
            if indiv2.is_feasible: return indiv2                                      # Only indiv2 is feasible
            else                 : return self._get_lowest_constraint(indiv1, indiv2) # Both are unfeasible

    def get_best(self, indiv1:Individual, indiv2:Individual)->Individual:
        fn = self._get_best_constraints if self.have_constraints else self._get_best
        return fn(indiv1, indiv2)

    def get_sum_constraints(self, indiv:Individual)->float:
        constraints = np.array(indiv.constraints)
        constraints[constraints<0] = 0
        return sum(constraints)

    def eval_feasibility(self, indiv:Individual)->bool:
        return indiv.constraints_sum == 0

    def _evolve(self, indiv:Individual)->None:
        dims = indiv.dimensions
        jrand = np.random.randint(dims)
        crs = np.argwhere(np.random.rand(dims) < self.CR)[:,0].tolist()
        picked_dims = get_unique(crs + [jrand])

        if len(picked_dims) > 0:
            picked = [self.population[i].data[picked_dims] for i in pick_n_but(3, indiv.idx, len(self.population))]
            F = np.random.uniform(self.beta_min, self.beta_max, size=len(picked_dims))
            new_data = picked[0] + F*picked[1] - picked[2]
            indiv.data[picked_dims] = new_data
            indiv.clip_limits()

    def evolve(self, indiv:Individual)->None:
        try:
            self.cb_handler.on_evolve_begin()
            self._evolve(indiv)
        except CancelEvolveException as exception: self.cb_handler.on_cancel_evolve(exception)
        finally: self.cb_handler.on_evolve_end()

    def eval_fitness(self, indiv:Individual)->None:
        fitness = None
        try:
            self.cb_handler.on_fitness_begin()
            fitness = self.get_fitness(indiv, self.fitness_params, self.time)
        except CancelFitnessException as exception: self.cb_handler.on_fitness_cancel(exception)
        finally:
            evals = self.cb_handler.on_fitness_end(fitness=fitness)
            self.change_time(evals)

    def change_time(self, evals:int)->None:
        if self.have_time:
            if evals % self.frequency == 0:
                self.cb_handler.on_time_change()

    def detect_change(self, indiv:Individual)->None:
        try:
            self.cb_handler.on_detect_change_begin()
            self.eval_fitness(indiv)
            if self.have_constraints: self.eval_constraints(indiv)

        except CancelDetectChangeException as exception: self.cb_handler.on_cancel_detect_change(exception)
        finally:
            new_indiv,changed = self.cb_handler.on_detect_change_end()
            if changed: indiv.copy_from(new_indiv)

    def get_each_constraint(self, indiv:Individual)->None:
        try:
            for fn,b in zip(self.get_constraints,self.constraint_params):
                self.cb_handler.on_each_constraint_begin()
                constraint = fn(indiv, b, self.time)
                self.cb_handler.on_each_constraint_end(constraint=constraint)

        except CancelEachConstraintException as exception: self.cb_handler.on_cancel_each_constraint(exception)

    def eval_constraints(self, indiv:Individual)->None:
        try:
            self.cb_handler.on_constraints_begin()
            self.get_each_constraint(indiv)

        except CancelConstraintsException as exception: self.cb_handler.on_cancel_constraints(exception)
        finally: self.cb_handler.on_constraints_end()

    def process_individual(self, indiv:Individual)->None:
        try:
            self.cb_handler.on_individual_begin(indiv=indiv)
            if self.time_change_detect and self.have_time and (indiv.idx in self.time_change_checks): self.detect_change(indiv)
            self.evolve(indiv)
            self.eval_fitness(indiv)
            if self.have_constraints: self.eval_constraints(indiv)

        except CancelGenException as exception: self.cb_handler.on_cancel_individual(exception)
        finally:
            new_indiv,changed = self.cb_handler.on_individual_end()
            if changed: indiv.copy_from(new_indiv)

    def process_individuals(self)->None:
        for indiv in self.population: self.process_individual(indiv)

    def run_one_gen(self)->None:
        try:
            self.cb_handler.on_gen_begin()
            self.process_individuals()

        except CancelGenException as exception: self.cb_handler.on_cancel_gen(exception)
        finally: self.cb_handler.on_gen_end()

    def run(self, generations:int, show_graph:bool=True, update_each:int=10, show_report:bool=True, silent:bool=False)->None:
        pbar = master_bar(range(1))
        try:
            self.cb_handler.on_run_begin(generations, pbar, self.max_evals, self.max_times, self.frequency,
                                         show_graph=show_graph, update_each=update_each, show_report=show_report, silent=silent)
            for _ in pbar:
                for gen in progress_bar(range(generations), parent=pbar): self.run_one_gen()

        except CancelRunException as exception: self.cb_handler.on_cancel_run(exception)
        finally: self.cb_handler.on_run_end()

    def clone(self)->'Optimization': return deepcopy(self)

    def create_multiple_runs(self, n_runs:int, **kwargs:Any)->'Runs':
        runs = Runs.from_template([self.clone() for _ in range(n_runs)], **kwargs)
        runs.reset_population()
        return runs

def _optimization_run(opt:Optimization, i:int, generations:int)->Optimization:
    opt.run(generations, silent=True)
    return opt

@dataclass
class Runs:
    optimizations:Collection[Optimization]
    parallel:bool=True
    n_cpus:Optional[int]=None

    def __post_init__(self):
        self.n_cpus = ifnone(self.n_cpus, cpu_count())

    def reset_population(self)->None:
        for opt in self.optimizations: opt.population.refresh()

    def run(self, generations:int)->None:
        self.optimizations = parallel(partial(_optimization_run, generations=generations), self.optimizations, max_workers=self.n_cpus)
        self.times_data = self.get_times_data()

    @classmethod
    def from_template(cls, optimizations:Collection[Optimization], **kwargs:Any)->'Runs':
        return cls(optimizations, **kwargs)

    def get_times_data(self)->pd.DataFrame:
        times_data = np.stack([e.recorder.best_times_fitness for e in self.optimizations])
        df = pd.DataFrame(times_data.T)
        df.columns = [f'run_{i+1}' for i in range(df.shape[1])]
        return df
