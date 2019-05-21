from .imports import *
from .utils import *

__all__ = ['Callback', 'CallbackHandler', 'Recorder', 'CancelEvolveException', 'CancelFitnessException', 'CancelEachConstraintException',
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
    def on_evolve_begin(self, **kwargs:Any)->None: pass
    def on_evolve_end(self, **kwargs:Any)->None: pass
    def on_fitness_begin(self, **kwargs:Any)->None: pass
    def on_fitness_end(self, **kwargs:Any)->None: pass
    def on_constraints_begin(self, **kwargs:Any)->None: pass
    def on_each_constraint_begin(self, **kwargs:Any)->None: pass
    def on_each_constraint_end(self, **kwargs:Any)->None: pass
    def on_constraints_end(self, **kwargs:Any)->None: pass
    def on_individual_end(self, **kwargs:Any)->None: pass
    def on_gen_end(self, **kwargs:Any)->None: pass
    def on_run_end(self, **kwargs:Any)->None: pass
    def on_cancel_evolve(self, **kwargs:Any)->None: pass
    def on_cancel_fitness(self, **kwargs:Any)->None: pass
    def on_cancel_each_constraint(self, **kwargs:Any)->None: pass
    def on_cancel_constraints(self, **kwargs:Any)->None: pass
    def on_cancel_gen(self, **kwargs:Any)->None: pass
    def on_cancel_run(self, **kwargs:Any)->None: pass

def _get_init_state(): return dict(gen=0, evals=0, best=None)

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

    def on_run_begin(self, generations:int, pbar:PBar, metrics:Collection[str], max_evals:Optional[int], show_graph:bool, update_each:int)->None:
        gen_end = generations + self.state_dict['gen'] - 1
        self.state_dict.update(dict(run_gens=generations, gen_end=gen_end, pbar=pbar, metrics=metrics, max_evals=max_evals,
                                    show_graph=show_graph, update_each=update_each))
        self('run_begin')

    def on_gen_begin(self, **kwargs:Any)->None:
        self.state_dict['is_final_gen'] = self.state_dict['gen']==self.state_dict['gen_end']
        self('gen_begin')

    def on_individual_begin(self, indiv, **kwargs:Any)->None:
        self.state_dict['last_indiv'] = indiv
        self('individual_begin')

    def on_evolve_begin(self, **kwargs:Any)->None:
        indiv = self.state_dict['last_indiv']
        if indiv.fitness_value is None:
            self.optim.eval_fitness(indiv)
            if self.optim.have_constraints: self.optim.eval_constraints(indiv)
        self('evolve_begin')

    def on_evolve_end(self, **kwargs:Any)->None:
        self('evolve_end')

    def on_fitness_begin(self, **kwargs:Any)->None:
        self('fitness_begin')

    def on_fitness_end(self, fitness, **kwargs:Any)->None:
        indiv = self.state_dict['last_indiv']
        indiv.gen = self.state_dict['gen']
        self.state_dict['evals'] += 1
        self.state_dict['last_fitness'] = fitness
        self('fitness_end')
        indiv.fitness_value = self.state_dict['last_fitness']
        if self.state_dict['max_evals'] is not None:
            if self.state_dict['max_evals'] <= self.state_dict['evals']: raise CancelRunException

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
        indiv.constraints_sum = sum(indiv.constraints)
        indiv.is_feasible = self.optim.eval_feasibility(indiv)

    def on_individual_end(self, **kwargs:Any)->None:
        indiv = self.state_dict['last_indiv']
        if self.state_dict['best'] is None: self.state_dict['best'] = indiv
        else                              : self.state_dict['best'] = self.optim.get_best(indiv, self.state_dict['best'])
        self('individual_end')

    def on_gen_end(self, **kwargs:Any)->None:
        self('gen_end')
        self.state_dict['gen'] += 1

    def on_run_end(self, **kwargs:Any)->None:
        self('run_end')

    def on_cancel_evolve(self, **kwargs:Any)->None:
        self('cancel_evolve')

    def on_cancel_fitness(self, **kwargs:Any)->None:
        self('cancel_fitness')

    def on_cancel_each_constraint(self, **kwargs:Any)->None:
        self('cancel_each_constraint')

    def on_cancel_constraints(self, **kwargs:Any)->None:
        self('cancel_constraints')

    def on_cancel_gen(self, **kwargs:Any)->None:
        self('cancel_gen')

    def on_cancel_run(self, **kwargs:Any)->None:
        self('cancel_run')

class Recorder(Callback):
    _order = 99

    def __init__(self, optim):
        super().__init__(optim)
        self.bests = []

    def on_run_begin(self, pbar, metrics, **kwargs):
        self.pbar = pbar
        self.metrics = metrics
        self.pbar.names = metrics

    def on_gen_end(self, gen, update_each, show_graph, best, **kwargs):
        self.bests.append(best.fitness_value)
        if show_graph and (gen+1)%update_each==0: self.pbar.update_graph(self.get_plot_data())

    def on_run_end(self, show_graph, **kwargs):
        if show_graph: self.pbar.update_graph(self.get_plot_data())

    @property
    def best(self): return min(self.bests)

    @property
    def best_with_idx(self):
        idx = np.argmin(self.bests)
        return (idx, self.bests[idx])

    def get_plot_data(self):
        x = np.arange(len(self.bests))
        return [[x,self.bests]]

    def plot(self, ax=None, alpha=0.75, size=100, color='green', figsize=(8,5)):
        if ax is None: fig,ax = plt.subplots(1, 1, figsize=figsize)
        for g,n in zip(self.get_plot_data(),self.metrics): ax.plot(*g, alpha=alpha, label=n)
        idx,best = self.best_with_idx
        ax.scatter(idx, best, s=size, c=color)
        ax.set_title(f'Best fitness value: {best:.2f}\nGeneration: {idx}')
        ax.set_xlabel('Generations')
        ax.set_ylabel('Fitness value')

class CancelEvolveException(Exception): pass
class CancelFitnessException(Exception): pass
class CancelEachConstraintException(Exception): pass
class CancelConstraintsException(Exception): pass
class CancelGenException(Exception): pass
class CancelRunException(Exception): pass
