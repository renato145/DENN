from .imports import *
from .utils import *

__all__ = ['Callback', 'CallbackHandler', 'Recorder', 'CancelRunException', 'CancelGenException', 'CancelEvolException', 'CancelFitnessException']
class Callback():
    _order = 0

    def __init__(self, optim):
        self.optim = optim
        setattr(self.optim, self.cb_name, self)

    @property
    def cb_name(self): return camel2snake(self.__class__.__name__)

    def on_run_begin(self, **kwargs:Any)->None:
        pass
    def on_gen_begin(self, **kwargs:Any)->None:
        pass
    def on_evol_all_begin(self, **kwargs:Any)->None:
        pass
    def on_evol_one_begin(self, **kwargs:Any)->None:
        pass
    def on_fitness_all_begin(self, **kwargs:Any)->None:
        pass
    def on_fitness_one_begin(self, **kwargs:Any)->None:
        pass
    def on_fitness_one_end(self, **kwargs:Any)->None:
        pass
    def on_fitness_all_end(self, **kwargs:Any)->None:
        pass
    def on_evol_one_end(self, **kwargs:Any)->None:
        pass
    def on_evol_all_end(self, **kwargs:Any)->None:
        pass
    def on_gen_end(self, **kwargs:Any)->None:
        pass
    def on_run_end(self, **kwargs:Any)->None:
        pass
    def on_cancel_run(self, **kwargs:Any)->None:
        pass
    def on_cancel_evol(self, **kwargs:Any)->None:
        pass
    def on_cancel_fitness(self, **kwargs:Any)->None:
        pass


def _get_init_state(): return dict(gen=0, evals=0, bests=[])

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

    def on_run_begin(self, generations:int, pbar:PBar, metrics:Collection[str], max_evals:Optional[int])->None:
        gen_end = generations + self.state_dict['gen'] - 1
        self.state_dict.update(dict(run_gens=generations, gen_end=gen_end, pbar=pbar, metrics=metrics, max_evals=max_evals))
        self('run_begin')

    def on_gen_begin(self, **kwargs:Any)->None:
        self.state_dict['final_gen'] = self.state_dict['gen']==self.state_dict['gen_end']
        self('gen_begin')

    def on_evol_all_begin(self, **kwargs:Any)->None:
        self('evol_all_begin')

    def on_evol_one_begin(self, **kwargs:Any)->None:
        self('evol_one_begin')

    def on_fitness_all_begin(self, **kwargs:Any)->None:
        self('fitness_all_begin')

    def on_fitness_one_begin(self, indiv, **kwargs:Any)->None:
        self('fitness_one_begin')

    def on_fitness_one_end(self, indiv, **kwargs:Any)->None:
        indiv.gen = self.state_dict['gen']
        self.state_dict['evals'] += 1
        self('fitness_one_end')
        if self.state_dict['max_evals'] is not None:
            if self.state_dict['max_evals'] == self.state_dict['evals']: raise CancelRunException

    def on_fitness_all_end(self, best, **kwargs:Any)->None:
        self('fitness_all_end', best=best)

    def on_evol_one_end(self, **kwargs:Any)->None:
        self('evol_one_end')

    def on_evol_all_end(self, **kwargs:Any)->None:
        self('evol_all_end')

    def on_gen_end(self, update_each, show_graph, **kwargs:Any)->None:
        self('gen_end', update_each=update_each, show_graph=show_graph)
        self.state_dict['gen'] += 1

    def on_run_end(self, **kwargs:Any)->None:
        self('run_end')

    def on_cancel_run(self, **kwargs:Any)->None:
        self('cancel_run')

    def on_cancel_evol(self, **kwargs:Any)->None:
        self('cancel_evol')

    def on_cancel_fitness(self, **kwargs:Any)->None:
        self('cancel_fitness')

class Recorder(Callback):
    _order = -10

    def __init__(self, optim):
        super().__init__(optim)
        self.bests = []

    def on_run_begin(self, pbar, metrics, **kwargs):
        self.pbar = pbar
        self.metrics = metrics
        self.pbar.names = metrics

    def on_fitness_all_end(self, best, **kwargs):
        self.bests.append(best)

    def on_gen_end(self, gen, final_gen, update_each, show_graph, **kwargs):
        if show_graph and ((gen+1)%update_each==0 or final_gen):
            self.pbar.update_graph(self.get_plot_data())

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
        ax.legend(loc='upper right')
        ax.set_title(f'Best fitness value: {best:.2f}\nGeneration: {idx}')

class CancelRunException(Exception): pass
class CancelGenException(Exception): pass
class CancelEvolException(Exception): pass
class CancelFitnessException(Exception): pass

