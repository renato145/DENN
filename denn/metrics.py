from .imports import *
from .callbacks import *
from .utils import *

__all__ = ['get_metrics', 'Metric', 'SpeedMetric']

def get_metrics(metrics:Optional[Collection['Metric']])->Collection[Optional['Metric']]:
    metrics = listify(metrics)
    metrics_classes = [e if isinstance(e, Metric) else e.func for e in metrics]
    if any(issubclass(metric, ThreadholdMetric) for metric in metrics_classes): metrics.append(ThreadholdStarter)
    return metrics

class Metric(Callback):
    _order = 90 # Needs to run before Recorder

class ThreadholdMetric(Metric): pass

class ThreadholdStarter(Callback):
    _order = 80 # Needs to run before Metrics

    def optimal_fitness(self, time:int)->float: return self.optim.optimal_fitness_values[time]
    def on_run_begin(self, **kwargs:Any)->dict: self.optim.state_dict.update({'threadhold_reached':False, 'threadhold_value':0.0})

    def on_gen_end(self, threadhold_reached:bool, time_evals:int, time:int, best:'Individual', **kwargs:Any)->Optional[dict]:
        if not threadhold_reached and best.is_feasible:
            this_best,optimal_best = best.fitness_value,self.optimal_fitness(time)
            threadhold_value = 1-min(this_best,1) if np.isclose(optimal_best, 0) else min((this_best/optimal_best)-1,0)
            return {'threadhold_value':threadhold_value}

    def on_time_change(self, max_time_reached:bool, **kwargs:Any)->dict: return {'threadhold_reached':max_time_reached, 'threadhold_value':0.0}

class SpeedMetric(ThreadholdMetric):
    def __init__(self, optim:'Optimization', threadhold:float=0.1):
        assert optim.optimal_fitness_values is not None, 'No optimal fitness values were given.'
        super().__init__(optim)
        self.threadhold = threadhold
        self.metrics = -np.ones(self.optim.max_times)
        self.success_rate = 0.0

    def on_gen_end(self, threadhold_reached:bool, threadhold_value:float, time:int, time_evals:int, **kwargs:Any)->Optional[dict]:
        if not threadhold_reached:
            if threadhold_value >= self.threadhold:
                self.metrics[time] = time_evals
                return {'threadhold_reached':True}

    def on_run_end(self, **kwargs:Any)->None: self.success_rate = sum(self.metrics > -1) / len(self.metrics)
