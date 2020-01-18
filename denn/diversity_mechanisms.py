from .imports import *
from .optimization import *
from .utils import *
from .callbacks import *

__all__ = ['OnChangeRestartPopulation', 'RandomImmigrants', 'RandomImmigrantsOnChange', 'Hypermutation']

class OnChangeRestartPopulation(Callback):
    def on_detect_change_end_before_reval(self, change_detected:bool, **kwargs:Any)->None:
        if change_detected: self.optim.population.refresh()

class RandomImmigrants(Callback):
    def __init__(self, optim:'Optimization', replacement_rate:int=3):
        'http://www.gardeux-vincent.eu/These/Papiers/Bibli1/Grefenstette92.pdf'
        super().__init__(optim)
        self.replacement_rate = replacement_rate
        raise NotImplementedError

    def on_gen_end(self, **kwargs:Any)->dict:
        raise NotImplementedError

class RandomImmigrantsOnChange(Callback):
    def __init__(self, optim:'Optimization', replacement_rate:int=3):
        '''
        Modification on the original version to apply only when changes are detected.
        http://www.gardeux-vincent.eu/These/Papiers/Bibli1/Grefenstette92.pdf
        '''
        super().__init__(optim)
        self.replacement_rate = replacement_rate

    def on_detect_change_end(self, change_detected:bool, **kwargs:Any)->dict:
        idxs = []
        if change_detected:
            picked_idxs = np.random.choice(self.optim.population.n, self.replacement_rate, replace=False)
            for idx in picked_idxs: self.optim.population[idx].refresh()

        return {'detected_idxs':picked_idxs}

class Hypermutation(Callback):
    def __init__(self, value:float, anneal_func:Callable=SchedExp):
        '''From: "An investigation into the use of hypermutation as an adaptive operator in genetic
                  algorithms having continuous, time-dependent nonstationary environments"
        Increases `F` and `CR` when a change is detected.
        '''
        raise NotImplementedError

    def on_detect_change_end(self, change_detected:bool, **kwargs:Any)->None:
        pass
