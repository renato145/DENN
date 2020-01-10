from .imports import *
from .utils import *
from .callbacks import *

__all__ = ['OnChangeRestartPopulation', 'RandomImmigrants']

class OnChangeRestartPopulation(Callback):
    def on_detect_change_end_before_reval(self, change_detected:bool, **kwargs:Any)->None:
        if change_detected: self.optim.population.refresh()

class RandomImmigrants(Callback):
    'http://www.gardeux-vincent.eu/These/Papiers/Bibli1/Grefenstette92.pdf'
    def __init__(self, replacement_rate:int=3):
        self.replacement_rate = replacement_rate

    def on_detect_change_end(self, change_detected:bool, **kwargs:Any)->dict:
        idxs = []
        if change_detected:
            picked_idxs = np.random.choice(self.optim.population.n, self.replacement_rate, replace=False)
            for idx in picked_idxs: self.population[idx].refresh()

        return {'detected_idxs':picked_idxs}
