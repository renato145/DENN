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
    _order = 10 # Needs to run after `NNTrainer`

    def __init__(self, optim:'Optimization', replacement_rate:int=3):
        '''
        Modification on the original version to apply only when changes are detected.
        http://www.gardeux-vincent.eu/These/Papiers/Bibli1/Grefenstette92.pdf
        '''
        super().__init__(optim)
        self.replacement_rate = replacement_rate

    def on_detect_change_end(self, change_detected:bool, detected_idxs:Ints, **kwargs:Any)->dict:
        if change_detected:
            idxs = [o for o in range(self.optim.population.n) if o not in detected_idxs]
            picked_idxs = np.random.choice(idxs, self.replacement_rate, replace=False)
            for idx in picked_idxs: self.optim.population[idx].refresh()

        return {'detected_idxs':detected_idxs+picked_idxs.tolist()}

class Hypermutation(Callback):
    def __init__(self, optim:'Optimization', CR:float=0.7, beta_min:float=0.8, beta_max:float =1.0,
                 frequency_factor:float=6.0): #frequency_factor:float
        '''From: "An investigation into the use of hypermutation as an adaptive operator in genetic
                  algorithms having continuous, time-dependent nonstationary environments"
        Temporaly increases `F`, `beta_min` and `beta_max` when a change is detected.
        Params:
        - frequency_factor: multiply the optimization frequency to get the number of affected generations.
        '''
        super().__init__(optim)
        self.CR,self.beta_min,self.beta_max = CR,beta_min,beta_max
        self.generations = int(frequency_factor * optim.frequency)
        self.base_CR,self.base_beta_min,self.base_beta_max = optim.CR,optim.beta_min,optim.beta_max
        self.current_gen = 0

    def on_detect_change_end(self, change_detected:bool, **kwargs:Any)->None:
        if change_detected:
            self.optim.CR =  self.CR
            self.optim.beta_min = self.beta_min
            self.optim.beta_max = self.beta_max
            self.current_gen = 0

    def on_gen_end(self, **kwargs:Any)->None:
        # print(f'({self.current_gen}) CR={self.optim.CR} - beta_min={self.optim.beta_min} - beta_max={self.optim.beta_max}')
        if self.current_gen == self.generations:
            self.optim.CR = self.base_CR
            self.optim.beta_min = self.base_beta_min
            self.optim.beta_max = self.base_beta_max
            # import pdb; pdb.set_trace()

        self.current_gen += 1
