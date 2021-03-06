from .imports import *
from .callbacks import *
from .utils import *

__all__ = ['get_metrics', 'Metric', 'SpeedMetric', 'OfflineError', 'ModifiedOfflineError', 'AbsoluteRecoverRate',
           'NNTimer', 'NNErrors', 'FirstDistance']

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
            threadhold_value = this_best if np.isclose(optimal_best, 0) else abs(this_best - optimal_best)/abs(optimal_best)
            return {'threadhold_value':threadhold_value}

    def on_time_change(self, **kwargs:Any)->dict:
        return {'threadhold_reached':False, 'threadhold_value':0.0}

class SpeedMetric(ThreadholdMetric):
    def __init__(self, optim:'Optimization', threadhold:float=0.1):
        assert optim.optimal_fitness_values is not None, 'no optimal fitness values were given.'
        super().__init__(optim)
        self.threadhold = threadhold
        self.speeds = np.asarray([np.nan]*self.optim.max_times)
        self.metrics = 0.0

    def __repr__(self)->str: return f'{self.__class__.__name__}(success rate): {self.metrics:.4f}'

    def on_gen_end(self, threadhold_reached:bool, threadhold_value:float, time:int, time_evals:int, max_time_reached:bool, **kwargs:Any)->Optional[dict]:
        if not threadhold_reached and not max_time_reached:
            if threadhold_value <= self.threadhold:
                self.speeds[time] = time_evals
                return {'threadhold_reached':True}

    def on_run_end(self, **kwargs:Any)->None:
        self.metrics = 1 - (sum(np.isnan(self.speeds)) / len(self.speeds))

    def plot(self, ax:Optional[plt.Axes]=None, figsize:Tuple[int,int]=(8,5), title:str='Speed metric', **kwargs:Any)->plt.Axes:
        if ax is None: fig,ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_xlabel('times')
        ax.set_ylabel('speed metric')
        ax.plot(self.speeds, label='speed_metric', **kwargs)
        ax.set_title(title)
        ax.legend()
        return ax

class OfflineError(Metric):
    def __init__(self, optim:'Optimization'):
        '''Get the absolute value of the deviation of the best solution found and optimal value of this time at each generation,
           and at the end of the run devides by the maximum number of generations.'''
        super().__init__(optim)
        self.metrics = 0.0

    def on_gen_end(self, best:'Individual', gen:int, time:int, max_time_reached:bool, **kwargs:Any)->None:
        if time==10:self.gen_remain=gen
        if (time>10) and (not max_time_reached): self.metrics += np.abs(self.optimal_fitness(time) - best.fitness_value)
        

    def on_run_end(self, gen:int, **kwargs:Any)->None:
        self.metrics /= (gen-self.gen_remain)

class ModifiedOfflineError(OfflineError):
    def __init__(self, optim:'Optimization'):
        '''Modification of `ModifiedOfflineError` for constrained problems: get the worst feasible solution in the population if the
         current best is not feasible.'''
        super().__init__(optim)

    def on_gen_end(self, best:'Individual', gen:int, time:int, max_time_reached:bool, **kwargs:Any)->None:
        if time==10:self.gen_remain=gen
        if (time>10) and (not max_time_reached):
            if best.is_feasible: self.metrics += np.abs(self.optimal_fitness(time) - best.fitness_value)
            else               : self.metrics += np.abs(self.optimal_fitness(time) - self.get_worse().fitness_value)

    def on_run_end(self, gen:int, **kwargs:Any)->None:
        self.metrics /= (gen-self.gen_remain)

class AbsoluteRecoverRate(Metric):
    def __init__(self, optim:'optimization'):
        '''From: https://link.springer.com/content/pdf/10.1007%2F978-3-319-77538-8_56.pdf
        '''
        super().__init__(optim)
        assert optim.optimal_fitness_values is not None, 'no optimal fitness values were given.'
        self.time_values = []
        self.metrics = 0.0
        self.this_time_first_best = None
        self.acummulated_sum = 0.0
        self.n_time_gen = 0.0
        self.best_case = False

    def on_gen_end(self, best:'Individual', time:int, max_time_reached:bool, **kwargs:Any)->None:
        self.n_time_gen += 1
        if max_time_reached: return
        if self.this_time_first_best is None:
            if best.is_feasible and (best.time==time):
                self.this_time_first_best = best.fitness_value
                diff = abs(self.this_time_first_best - self.optimal_fitness(time))
                # TODO:
                # consider the case when n_time_gen is > 1 and diff is close to 0:
                # if diff < 1e-4:
                #     if self.n_time_gen == 1: self.best_case = True
                #     else : ???
                # For now we ignore the condition in the following line:
                if diff < 1e-4: self.best_case = True
        else:
            if best.is_feasible:
                self.acummulated_sum += abs(best.fitness_value - self.this_time_first_best)

    def on_time_change(self, time:int, **kwargs:Any)->None:
        if self.this_time_first_best is None: val = 0.0
        elif self.best_case: val = 1.0
        else: val = self.acummulated_sum / (self.n_time_gen * abs(self.optimal_fitness(time) - self.this_time_first_best))

        self.time_values.append(val)
        self.this_time_first_best = None
        self.acummulated_sum = 0.0
        self.n_time_gen = 0.0
        self.best_case = False

    def on_run_end(self, **kwargs:Any)->None:
        self.metrics = np.mean(self.time_values)

class NNTimer(Metric):
    def __init__(self, optim:'Optimization'):
        'Records the amount of time the optimization expends on the neural network.'
        super().__init__(optim)
        self.times = []
        self.total_time = 0.0
        self.metrics = 0.0

    def on_run_end(self, **kwargs:Any)->None:
        self.total_time = sum(self.times)
        elapsed = get_time() - self.optim.recorder.start_time
        self.metrics = self.total_time / elapsed

class NNErrors(Metric):
    def __init__(self, optim:'Optimization'):
        'Records the errors of predicted values by the neural network.'
        super().__init__(optim)
        self.metrics = []

class FirstDistance(Metric):
    def __init__(self, optim:'Optimization'):
        '''Records the euclidean distances between the best individual after the first generation
           vs the optimal solution, for each time.'''
        super().__init__(optim)
        self.metrics = []
        self.do_record_metric = False

    def on_detect_change_end(self, change_detected:bool, **kwargs:Any)->None:
        if change_detected: self.do_record_metric = True

    def on_gen_end(self, best:'Individual', time:int, max_time_reached:bool, **kwargs:Any)->None:
        if max_time_reached: return
        if self.do_record_metric:
            this_best = self.optim.optimal_positions[time]
            error = np.linalg.norm(best.data - this_best)
            self.metrics.append(error)
            self.do_record_metric = False
