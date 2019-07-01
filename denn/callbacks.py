from .imports import *
from .utils import *

__all__ = ['Callback', 'CallbackHandler', 'DynamicConstraint', 'Recorder', 'OnChangeRestartPopulation',
           'CancelDetectChangeException', 'CancelEvolveException', 'CancelFitnessException', 'CancelEachConstraintException',
           'CancelConstraintsException', 'CancelGenException', 'CancelRunException']

class Callback():
    _order = 0

    def __init__(self, optim):
        self.optim = optim
        setattr(self.optim, self.cb_name, self)

    @property
    def cb_name(self): return camel2snake(self.__class__.__name__)
    def on_run_begin(self, **kwargs:Any)->None: pass
    def on_gen_begin(self, **kwargs:Any)->None: pass
    def on_individual_begin(self, **kwargs:Any)->None: pass
    def on_detect_change_begin(self, **kwargs:Any)->None: pass
    def on_detect_change_end(self, **kwargs:Any)->None: pass
    def on_evolve_begin(self, **kwargs:Any)->None: pass
    def on_evolve_end(self, **kwargs:Any)->None: pass
    def on_fitness_begin(self, **kwargs:Any)->None: pass
    def on_fitness_end(self, **kwargs:Any)->None: pass
    def on_time_change(self, **kwargs:Any)->None: pass
    def on_constraints_begin(self, **kwargs:Any)->None: pass
    def on_each_constraint_begin(self, **kwargs:Any)->None: pass
    def on_each_constraint_end(self, **kwargs:Any)->None: pass
    def on_constraints_end(self, **kwargs:Any)->None: pass
    def on_individual_end(self, **kwargs:Any)->None: pass
    def on_gen_end(self, **kwargs:Any)->None: pass
    def on_run_end(self, **kwargs:Any)->None: pass
    def on_cancel_detect_change(self, **kwargs:Any)->None: pass
    def on_cancel_evolve(self, **kwargs:Any)->None: pass
    def on_cancel_fitness(self, **kwargs:Any)->None: pass
    def on_cancel_each_constraint(self, **kwargs:Any)->None: pass
    def on_cancel_constraints(self, **kwargs:Any)->None: pass
    def on_cancel_gen(self, **kwargs:Any)->None: pass
    def on_cancel_run(self, **kwargs:Any)->None: pass

def _get_init_state(): return dict(gen=0, evals=0, time=0, best=None)

@dataclass
class CallbackHandler():
    optim:Any=None
    callbacks:Collection[Callback]=None

    def __post_init__(self):
        self.callbacks = listify(self.callbacks)
        self.callbacks = sorted(self.callbacks, key=lambda o: getattr(o, '_order', 0))
        self.state_dict = _get_init_state()

    def _call_and_update(self, cb, cb_name, **kwargs):
        "Call `cb_name` on `cb` and update the inner state."
        new = ifnone(getattr(cb, f'on_{cb_name}')(**self.state_dict, **kwargs), dict())
        for k,v in new.items():
            if k not in self.state_dict:
                raise Exception(f"{k} isn't a valid key in the state of the callbacks.")
            else: self.state_dict[k] = v

    def __call__(self, cb_name, **kwargs):
        for cb in self.callbacks: self._call_and_update(cb, cb_name, **kwargs)

    def on_run_begin(self, generations:int, pbar:PBar, metrics:Collection[str], max_evals:Optional[int], max_times:Optional[int],
                     frequency:Optional[int], show_graph:bool, update_each:int, show_report:bool, silent:bool)->None:
        gen_end = generations + self.state_dict['gen'] - 1
        update_each = min(update_each, generations)
        if silent: show_graph = show_report = False
        self.state_dict.update(dict(run_gens=generations, gen_end=gen_end, pbar=pbar, metrics=metrics, max_evals=max_evals, max_times=max_times,
                                    frequency=frequency, show_graph=show_graph, update_each=update_each, show_report=show_report, silent=silent))
        self('run_begin')

    def on_gen_begin(self, **kwargs:Any)->None:
        self.state_dict['is_final_gen'] = self.state_dict['gen']==self.state_dict['gen_end']
        self('gen_begin')

    def on_individual_begin(self, indiv, **kwargs:Any)->None:
        self.state_dict['last_indiv'] = indiv
        if indiv.fitness_value is None:
            self.optim.eval_fitness(indiv)
            if self.optim.have_constraints: self.optim.eval_constraints(indiv)

        self.state_dict['indiv_bkup'] = indiv.clone()
        self('individual_begin')

    def on_detect_change_begin(self, **kwargs:Any)->None:
        self.state_dict['skip_detect_change'] = False
        self.state_dict['change_detected'] = False
        if self.state_dict['gen'] == 0:
            self.state_dict['skip_detect_change'] = True
            raise CancelDetectChangeException('Skipping first generation.')
        self('detect_change_begin')

    def on_detect_change_end(self, **kwargs:Any)->None:
        if not self.state_dict['skip_detect_change']:
            indiv = self.state_dict['last_indiv']
            indiv_bkup = self.state_dict['indiv_bkup']
            self.state_dict['change_detected'] = indiv != indiv_bkup
            self('detect_change_end')
            indiv = self.state_dict['last_indiv']

            if self.state_dict['change_detected']:
                population = self.optim.population
                for i,ind in enumerate([self.state_dict['best']] + population.individuals):
                    self.state_dict['last_indiv'] = ind
                    self.optim.eval_fitness(ind)
                    if self.optim.have_constraints: self.optim.eval_constraints(ind)
                    ind.gen = self.state_dict['gen']
                    ind.time = self.state_dict['time']
                    if i > 0: self.state_dict['best'] = self.optim.get_best(ind, self.state_dict['best'])

                indiv = population[indiv.idx]
                self.state_dict['last_indiv'] = indiv
                self.state_dict['indiv_bkup'] = indiv.clone()

        return self.state_dict['last_indiv'],self.state_dict['change_detected']

    def on_evolve_begin(self, **kwargs:Any)->None:
        self('evolve_begin')

    def on_evolve_end(self, **kwargs:Any)->None:
        self('evolve_end')

    def on_fitness_begin(self, **kwargs:Any)->None:
        self('fitness_begin')

    def on_fitness_end(self, fitness:float, **kwargs:Any)->None:
        indiv = self.state_dict['last_indiv']
        self.state_dict['last_fitness'] = fitness
        self('fitness_end')
        indiv.fitness_value = self.state_dict['last_fitness']

        # eval step
        self.state_dict['evals'] += 1
        if self.state_dict['max_evals'] is not None:
            if self.state_dict['max_evals'] <= self.state_dict['evals']: raise CancelRunException('`max_evals` reached.')

        # return evals to check time step
        return self.state_dict['evals']

    def on_time_change(self, **kwargs:Any)->None:
        self.state_dict['time'] += 1
        self('time_change')
        if self.state_dict['max_times'] is not None:
            if self.state_dict['max_times'] <= self.state_dict['time']: raise CancelRunException('`max_time` reached.')

    def on_constraints_begin(self, **kwargs:Any)->None:
        indiv = self.state_dict['last_indiv']
        indiv.constraints = []
        self('constraints_begin')

    def on_each_constraint_begin(self, b, **kwargs:Any)->Any:
        self.state_dict['last_constraint_param'] = b
        self('each_constraint_begin')
        return self.state_dict['last_constraint_param']

    def on_each_constraint_end(self, constraint, **kwargs:Any)->None:
        indiv = self.state_dict['last_indiv']
        self.state_dict['last_each_constraint'] = constraint
        self('each_constraint_end')
        indiv.constraints.append(self.state_dict['last_each_constraint'])

    def on_constraints_end(self, **kwargs:Any)->None:
        indiv = self.state_dict['last_indiv']
        self.state_dict['last_constraints'] = indiv.constraints
        self('constraints_end')
        indiv.constraints = self.state_dict['last_constraints']
        indiv.constraints_sum = self.optim.get_sum_constraints(indiv)
        indiv.is_feasible = self.optim.eval_feasibility(indiv)

    def on_individual_end(self, **kwargs:Any)->Tuple['Individual',bool]:
        indiv = self.state_dict['last_indiv']
        indiv_bkup = self.state_dict['indiv_bkup']
        new_indiv = self.optim.get_best(indiv, indiv_bkup)
        change = new_indiv != indiv
        new_indiv.gen = self.state_dict['gen']
        new_indiv.time = self.state_dict['time']

        if self.state_dict['best'] is None: self.state_dict['best'] = new_indiv
        else                              : self.state_dict['best'] = self.optim.get_best(new_indiv, self.state_dict['best'])

        self.state_dict['new_indiv'] = new_indiv
        self.state_dict['change_indiv'] = change
        self('individual_end')
        return self.state_dict['new_indiv'],self.state_dict['change_indiv']

    def on_gen_end(self, **kwargs:Any)->None:
        self('gen_end')
        self.state_dict['gen'] += 1

    def on_run_end(self, **kwargs:Any)->None:
        self('run_end')

    def on_cancel_detect_change(self, exception, **kwargs:Any)->None:
        if self.state_dict['skip_detect_change']: return False
        print(f'Detect change cancelled: {exception}')
        self('cancel_detect_change')

    def on_cancel_evolve(self, exception, **kwargs:Any)->None:
        print(f'Evolve cancelled: {exception}')
        self('cancel_evolve')

    def on_cancel_fitness(self, exception, **kwargs:Any)->None:
        print(f'Fitness cancelled: {exception}')
        self('cancel_fitness')

    def on_cancel_each_constraint(self, exception, **kwargs:Any)->None:
        print(f'Each_constraint cancelled: {exception}')
        self('cancel_each_constraint')

    def on_cancel_constraints(self, exception, **kwargs:Any)->None:
        print(f'Constraints cancelled: {exception}')
        self('cancel_constraints')

    def on_cancel_gen(self, exception, **kwargs:Any)->None:
        print(f'Gen cancelled: {exception}')
        self('cancel_gen')

    def on_cancel_run(self, exception, **kwargs:Any)->None:
        print(f'Run cancelled: {exception}')
        self('cancel_run')

class DynamicConstraint(Callback):
    def on_each_constraint_begin(self, last_constraint_param:Optional[Collection[float]], time:int, **kwargs:Any)->Dict:
        return {'last_constraint_param': last_constraint_param[time]}

class OnChangeRestartPopulation(Callback):
    def on_detect_change_end(self, change_detected:bool, **kwargs:Any)->Optional[Dict]:
        if change_detected: self.optim.population.refresh()

class Recorder(Callback):
    _order = 99

    def __init__(self, optim):
        super().__init__(optim)
        self.bests = []
        self.best_times = []

    def on_run_begin(self, pbar, metrics, **kwargs):
        self.pbar = pbar
        self.metrics = metrics
        self.pbar.names = metrics
        self.start_time = get_time()

    def on_gen_end(self, best:'Individual', **kwargs):
        self.bests.append(best.fitness_value)

    def on_time_change(self, best:'Individual', time:int, show_graph:bool, update_each:int, **kwargs:Any):
        self.best_times.append(best.clone())
        if show_graph and (time+1)%update_each==0: self._pbar_plot()

    def on_run_end(self, show_graph, show_report, silent, **kwargs):
        self.elapsed = format_time(get_time() - self.start_time)
        if show_graph: self._pbar_plot()
        if not silent: clear_output()
        if show_report: self.show_report()

    @property
    def best(self): return min(self.bests)

    @property
    def best_with_idx(self):
        idx = np.argmin(self.bests)
        return (idx, self.bests[idx])

    @property
    def best_times_fitness(self): return np.asarray([e.fitness_value for e in self.best_times])

    def show_report(self):
        print('A proper report should be shown here :)')
        print(f'Total time: {self.elapsed}')

    def _pbar_plot(self):
        self.pbar.update_graph(self.get_plot_data(), x_bounds=(0,self.optim.max_times))

    def get_plot_data(self):
        x = np.arange(len(self.best_times_fitness))
        return [[x,self.best_times_fitness]]

    def plot_times(self, ax=None, figsize=(8,5), **kwargs):
        if ax is None: fig,ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(self.best_times_fitness, **kwargs)
        ax.set_xlabel('times')
        ax.set_ylabel('fitness value')

    def plot_generations(self, ax=None, alpha=0.75, size=100, color='green', figsize=(8,5)):
        if ax is None: fig,ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(self.bests, alpha=alpha)
        idx,best = self.best_with_idx
        ax.scatter(idx, best, s=size, c=color)
        ax.set_title(f'Best fitness value: {best:.2f}\nGeneration: {idx}')
        ax.set_xlabel('generations')
        ax.set_ylabel('fitness value')

class CancelDetectChangeException(Exception): pass
class CancelEvolveException(Exception): pass
class CancelFitnessException(Exception): pass
class CancelEachConstraintException(Exception): pass
class CancelConstraintsException(Exception): pass
class CancelGenException(Exception): pass
class CancelRunException(Exception): pass
