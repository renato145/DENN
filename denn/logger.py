import json
from .imports import *
from .utils import *
from .callbacks import *
from .optimization import *

__all__ = ['Logger']

class Logger(Callback):
    _order = 20
    def __init__(self, optim:Optimization, out_file:str='logger.json'):
        super().__init__(optim)
        self.path = optim.path / out_file

    def __repr__(self)->str: return f'{self.__class__.__name__}(path={str(self.path)!r})'
    def on_run_begin(self, **kwargs:Any)->None:
        population = self.optim.population
        self.data = {
            'limits': [population.lower_limit,population.upper_limit],
            'data': [],
        }

    def on_gen_begin(self, gen:int, time:int, best:Individual, **kwargs:Any)->None:
        data = []
        best_idx = best.idx
        i = 0
        for indiv in self.optim.population:
            if indiv.idx == best_idx: i+=1
            data.append({
                'idx': indiv.idx,
                'data': indiv.data.tolist(),
                'is_best': indiv.idx == best_idx,
                'fitness_value': float(indiv.fitness_value),
                'constraints_sum': float(indiv.constraints_sum),
                'is_feasible': bool(indiv.is_feasible),
                'time': time,
            })

        self.data['data'].append(data)

    def on_run_end(self, **kwargs:Any)->None: json.dump(self.data, self.path.open('w'))
