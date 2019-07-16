from .imports import *
from .callbacks import *
from .utils import *

__all__ = ['get_metrics', 'Metric', 'SpeedMetric', 'ModifiedOfflineError', 'ModifiedOfflineErrorConstraints']

def get_metrics(metrics:Optional[Collection['Metric']])->Collection[Optional['Metric']]:
    metrics = listify(metrics)
    metrics_classes = [e if not isinstance(e, partial) else e.func for e in metrics]
    if any(issubclass(metric, ThreadholdMetric) for metric in metrics_classes): metrics.append(ThreadholdStarter)
    return metrics

class Metric(Callback):
    _order = 90 # Needs to run before Recorder
    def __repr__(self)->str: return f'{self.__class__.__name__}: {self.metrics:.4f}'
    def optimal_fitness(self, time:int)->float: return self.optim.optimal_fitness_values[time]
    def get_worse(self)->'Individual':  return self.optim.population.get_worse()

class ThreadholdMetric(Metric): pass

class ThreadholdStarter(Callback):
    _order = 80 # Needs to run before Metrics

    def optimal_fitness(self, time:int)->float: return self.optim.optimal_fitness_values[time]
    def on_run_begin(self, **kwargs:Any)->dict: self.optim.state_dict.update({'threadhold_reached':False, 'threadhold_value':0.0})

    def on_gen_end(self, max_time_reached:bool, threadhold_reached:bool, time_evals:int, time:int, best:'Individual', **kwargs:Any)->Optional[dict]:
        if not threadhold_reached and best.is_feasible and not max_time_reached:
            this_best,optimal_best = best.fitness_value,self.optimal_fitness(time)
            threadhold_value = 1-min(this_best,1) if np.isclose(optimal_best, 0) else min((this_best/optimal_best)-1,0)
            return {'threadhold_value':threadhold_value}

    def on_time_change(self, **kwargs:Any)->dict:
        return {'threadhold_reached':False, 'threadhold_value':0.0}

class SpeedMetric(ThreadholdMetric):
    def __init__(self, optim:'Optimization', threadhold:float=0.1):
        assert optim.optimal_fitness_values is not None, 'No optimal fitness values were given.'
        super().__init__(optim)
        self.threadhold = threadhold
        self.speeds = -np.ones(self.optim.max_times)
        self.metrics = 0.0

    def on_gen_end(self, threadhold_reached:bool, threadhold_value:float, time:int, time_evals:int, max_time_reached:bool, **kwargs:Any)->Optional[dict]:
        if not threadhold_reached and not max_time_reached:
            if threadhold_value >= self.threadhold:
                self.speeds[time] = time_evals
                return {'threadhold_reached':True}

    def on_run_end(self, **kwargs:Any)->None:
        self.metrics = sum(self.speeds > -1) / len(self.speeds)

    def plot(self)->None:
        pass

class ModifiedOfflineError(Metric):
    def __init__(self, optim:'Optimization'):
        '''Get the absolute value of the deviation of the best solution found and optimal value of this time at each generation,
           and at the end of the run devides by the maximum number of generations.'''
        super().__init__(optim)
        self.metrics = 0

    def on_gen_end(self, best:'Individual', time:int, max_time_reached:bool, **kwargs:Any)->None:
        if not max_time_reached: self.metrics += np.abs(self.optimal_fitness(time) - best.fitness_value)

    def on_run_end(self, evals:int, **kwargs:Any)->None:
        self.metrics /= evals

class ModifiedOfflineErrorConstraints(ModifiedOfflineError):
    def __init__(self, optim:'Optimization'):
        '''Modification of `ModifiedOfflineError` for constrained problems: get the worst feasible solution in the population if the
         current best is not feasible.'''
        super().__init__(optim)

    def on_gen_end(self, best:'Individual', time:int, max_time_reached:bool, **kwargs:Any)->None:
        if not max_time_reached:
            if best.is_feasible: self.metrics += np.abs(self.optimal_fitness(time) - best.fitness_value)
            else               : self.metrics += np.abs(self.optimal_fitness(time) - self.get_worse().fitness_value)
