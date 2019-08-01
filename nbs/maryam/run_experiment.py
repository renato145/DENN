import fire
from denn import *
from fastai.callbacks import * 
import torch
import torch.nn.functional as F
from torch import nn

Experiment = Enum('Experiment', 'exp1 exp2 exp3 exp4')
Method = Enum('Methods', 'noNNRestart noNNReval NNnorm NNdrop')

class DropoutModel(nn.Module):
    def __init__(self, d, w, nf):
        super().__init__()
        self.fc1 = nn.Linear(d,nf)
        self.fc2 = nn.Linear(nf*w,d)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        fts = torch.cat([self.fc1(x[:,i]) for i in range(x.size(1))], dim=1)
        return self.fc2(F.dropout(self.act(fts), p=0.5))

class SimpleModel(nn.Module):
    def __init__(self, d, w, nf):
        super().__init__()
        self.fc1 = nn.Linear(d,nf)
        self.fc2 = nn.Linear(nf*w,d)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        fts = torch.cat([self.fc1(x[:,i]) for i in range(x.size(1))], dim=1)
        return self.fc2(self.act(fts))

def get_functions(experiment:Experiment)->Collection[Callable]:
    if experiment in [Experiment.exp1, Experiment.exp2]:
        def fitness_func(indiv, b, t): return (indiv.data**2).sum()
        def constraint_func(indiv, b, t): return -b[t] + sum((1/np.sqrt(D))*indiv.data)
    elif experiment == Experiment.exp3:
        def fitness_func(indiv, b, t): return ((indiv.data + 0.1*t)**2).sum()
        def constraint_func(indiv, b, t): return 0
    elif experiment == Experiment.exp4:
        def fitness_func(indiv, b, t): return ((indiv.data-b[t]*np.sin(np.pi/2*t))**2).sum()
        def constraint_func(indiv, b, t): return 0

    return fitness_func,constraint_func

def main(experiment:str, method:str, replace_mech:Optional[str]=None, D:int=30, runs:int=30, frequency:int=1_000,
         max_times:int=100, nn_window:int=5, nn_nf:int=4, nn_pick:int=3):
    # Setting variables
    experiment_type = getattr(Experiment, experiment)
    method_type = getattr(Method, method)
    replace_type = getattr(ReplaceMechanism, replace_mech)
    path = Path(f'../../data/results/{experiment}')
    fitness_func,constraint_func = get_functions(experiment_type)
    is_nn = method_type in [Method.NNnorm, Method.NNdrop]
    experiment_name = f'{method}'
    total_generations = max_times * frequency + 1_000
    if is_nn: experiment_name += f'_{replace_mech}'

    # Read files
    ab = pd.read_csv(path/'dC_01.csv')['b'].values
    df = pd.read_csv(path/'best_known.csv')
    best_known_fitness = df['fitness'].values
    best_known_sumcv   = df['sum_constraints'].values

    # Initialize metrics
    results = {'mof':[], 'sr':[], 'nfe':[], 'fitness':[], 'sumcv':[], 'arr':[]}
    if is_nn: results['nn_time'] = []

    # Run
    for run in progress_bar(range(runs)):
        callbacks = []
        if is_nn:
            if method_type==Method.NNnorm:
                model = SimpleModel (d=D, w=nn_window, nf=nn_nf) 
                nn_trainer = partial(NNTrainerNoNoise, model=model, n=nn_pick, window=nn_window, replace_mechanism=replace_type)
            if method_type==Method.NNdrop:
                model = DropoutModel(d=D, w=nn_window, nf=nn_nf) 
                nn_trainer = partial(NNTrainer       , model=model, n=nn_pick, window=nn_window, replace_mechanism=replace_type)
            
            callbacks.append(nn_trainer)
        elif method_type==Method.noNNRestart:
            callbacks.append(OnChangeRestartPopulation)

        population = Population.new_random(dimension=D)
        speed_metric = partial(SpeedMetric, threadhold=0.2)
        print(ab)
        opt = Optimization(population, fitness_func, constraint_func, fitness_params=ab, constraint_params=[ab],
                           max_times=max_times, frequency=frequency, callbacks=callbacks,
                           metrics=[speed_metric, ModifiedOfflineError, OfflineError, AbsoluteRecoverRate],
                           optimal_fitness_values=best_known_fitness, optimal_sum_constraints=best_known_sumcv)
        opt.run(total_generations, silent=True)

        # Store results
        results['mof'].append(opt.modified_offline_error.metrics)
        results['arr'].append(opt.absolute_recover_rate.metrics)
        results['sr'].append(opt.speed_metric.metrics)
        results['nfe'].append(opt.speed_metric.speeds)
        results['fitness'].append(opt.recorder.best_times_fitness)
        results['sumcv'].append(opt.recorder.best_times_constraints)
        if is_nn: results['nn_time'].append(opt.nn_timer.metrics)

    # Get results and save
    pd.DataFrame({'mof':results['mof']}).to_csv(path/f'{experiment_name}_mof.csv', index=False)
    pd.DataFrame({'sr':results['sr']}).to_csv(path/f'{experiment_name}_sr.csv', index=False)
    pd.DataFrame(results['nfe']).to_csv(path/f'{experiment_name}_nfe.csv', index=False)
    pd.DataFrame(results['fitness']).to_csv(path/f'{experiment_name}_fitness.csv', index=False)
    pd.DataFrame(results['sumcv']).to_csv(path/f'{experiment_name}_sumcv.csv', index=False)
    pd.DataFrame(results['arr']).to_csv(path/f'{experiment_name}_arr.csv', index=False)
    if is_nn: pd.DataFrame(results['nn_time']).to_csv(path/f'{experiment_name}_nn_time.csv', index=False)

if __name__ == '__main__':
    fire.Fire(main)