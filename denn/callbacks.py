from .imports import *
from .utils import *

__all__ = ['Callback', 'CallbackHandler', 'Recorder']

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

    def on_run_begin(self, generations, **kwargs:Any)->None:
        self('run_begin')

    def on_gen_begin(self, **kwargs:Any)->None:
        self('gen_begin')

    def on_evol_all_begin(self, **kwargs:Any)->None:
        self('evol_all_begin')

    def on_evol_one_begin(self, **kwargs:Any)->None:
        self('evol_one_begin')

    def on_fitness_all_begin(self, **kwargs:Any)->None:
        self('fitness_all_begin')

    def on_fitness_one_begin(self, **kwargs:Any)->None:
        self('fitness_one_begin')

    def on_fitness_one_end(self, **kwargs:Any)->None:
        self('fitness_one_end')
        self.state_dict['evals'] += 1

    def on_fitness_all_end(self, **kwargs:Any)->None:
        self('fitness_all_end')

    def on_evol_one_end(self, **kwargs:Any)->None:
        self('evol_one_end')

    def on_evol_all_end(self, **kwargs:Any)->None:
        self('evol_all_end')

    def on_gen_end(self, **kwargs:Any)->None:
        self('gen_end')
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
    pass

class CancelRunException(Exception): pass
class CancelGenException(Exception): pass
class CancelEvolException(Exception): pass
class CancelFitnessException(Exception): pass

