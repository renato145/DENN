import fire
from denn import *
from fastai.callbacks import * 
import torch
import torch.nn.functional as F
from torch import nn

Experiment = Enum('Experiment', 'exp1 exp2 exp3 exp4')
Method = Enum('Method', 'noNN NNnorm NNdrop NNtime NNconv')
FuncName = Enum('FuncName', 'sphere rastrigin ackley rosenbrock')
DiversityMethod = Enum('DiversityMethod', 'RI Cw Cwc CwN CwcN HMuPure HMu Rst')

class DropoutModel(nn.Module):
    def __init__(self, d:int, w:int, nf:int, dropout:float=0.5):
        super().__init__()
        self.fc1 = nn.Linear(d,nf)
        self.fc2 = nn.Linear(nf*w,d)
        self.act = nn.ReLU(inplace=True)
        self.dropout = dropout
        
    def forward(self, x):
        fts = torch.cat([self.fc1(x[:,i]) for i in range(x.size(1))], dim=1)
        return self.fc2(F.dropout(self.act(fts), p=self.dropout))

class SimpleModel(nn.Module):
    def __init__(self, d, w, nf):
        super().__init__()
        self.fc1 = nn.Linear(d,nf)
        self.fc2 = nn.Linear(nf*w,d)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        fts = torch.cat([self.fc1(x[:,i]) for i in range(x.size(1))], dim=1)
        return self.fc2(self.act(fts))

class TimeModel(nn.Module):
    def __init__(self, d, w, nf):
        super().__init__()
        self.fc1 = nn.Linear(d,nf)
        self.fc2 = nn.Linear(nf*(w+1),d)
        self.act = nn.ReLU(inplace=True)
        self.emb = nn.Linear(1, nf)
        
    def forward(self, x, time):
        embs = self.emb(time.repeat(x.size(0)).float()[:,None])
        fts = torch.cat([self.fc1(x[:,i]) for i in range(x.size(1))]+[embs], dim=1)
        return self.fc2(self.act(fts))

class ConvModel(nn.Module):
    def __init__(self, d, w, nf=4, ks=3, n_conv=2):
        super().__init__()
        convs = []
        for i in range(n_conv):
            convs += [nn.Conv1d(d if i==0 else nf, nf if i==0 else nf*2, kernel_size=ks),
                      nn.ReLU(inplace=True)]
            if i > 0: nf *= 2

        self.convs = nn.Sequential(*convs)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(nf, d)
        
    def forward(self, x):
        out = self.convs(x.permute(0,2,1))
        out = self.pool(out).squeeze(2)
        return self.linear(out)

def get_functions(experiment:Experiment, D:int, func_name:FuncName)->Collection[Callable]:
    if func_name==FuncName.sphere:
        if experiment in [Experiment.exp1, Experiment.exp2]:
            def fitness_func(indiv, b, t): return (indiv.data**2).sum()
            def constraint_func(indiv, b, t): return -b[t] + sum((1/np.sqrt(D))*indiv.data)
        elif experiment == Experiment.exp3:
            def fitness_func(indiv, b, t): return ((indiv.data + 0.1*t)**2).sum()
            def constraint_func(indiv, b, t): return 0
        elif experiment == Experiment.exp4:
            def fitness_func(indiv, b, t): return ((indiv.data-b[t]*np.sin(np.pi/2*t))**2).sum()
            def constraint_func(indiv, b, t): return 0
    elif func_name==FuncName.rastrigin:
        if experiment in [Experiment.exp1, Experiment.exp2]:
            def fitness_func(indiv, b, t): return 10*D+((indiv.data**2)-10*np.cos(2*np.pi*indiv.data)).sum()
            def constraint_func(indiv, b, t): return -b[t] + sum((1/np.sqrt(D))*indiv.data)
        elif experiment == Experiment.exp3:
            def fitness_func(indiv, b, t): return 10*D+(((indiv.data+0.1*t)**2)-10*np.cos(2*np.pi*indiv.data)).sum()
            def constraint_func(indiv, b, t): return 0
        elif experiment == Experiment.exp4:
            def fitness_func(indiv, b, t): return 10*D+(((indiv.data-b[t]*np.sin(np.pi/2*t))**2)-10*np.cos(2*np.pi*indiv.data)).sum()
            def constraint_func(indiv, b, t): return 0                
    elif func_name==FuncName.rosenbrock:
        if experiment in [Experiment.exp1, Experiment.exp2]:
            def fitness_func(indiv, b, t): return ((100 * (indiv.data[1:] - indiv.data[:-1]**2)**2) + (1-indiv.data[:-1])**2).sum()
            def constraint_func(indiv, b, t): return -b[t] + sum((1/np.sqrt(D))*indiv.data)
        elif experiment == Experiment.exp3:
            def fitness_func(indiv, b, t): return ((100 * ((indiv.data[1:]+0.1*t) - (indiv.data[:-1]+0.1*t)**2)**2) + (1-indiv.data[:-1]-0.1*t)**2).sum()
            def constraint_func(indiv, b, t): return 0
        elif experiment == Experiment.exp4:
            def fitness_func(indiv, b, t): return (((100 * ((indiv.data[1:]-b[t]*np.sin(np.pi/2*t)) - (indiv.data[:-1]-b[t]*np.sin(np.pi/2*t))**2)**2) + (1-indiv.data[:-1]+b[t]*np.sin(np.pi/2*t))**2)).sum()
            def constraint_func(indiv, b, t): return 0  
    return fitness_func,constraint_func

def main(experiment:str, func_name:str, method:str, frequency:int=1, freq_save:int=1, diversity_method:Optional[str]=None,
scale_factor:str='Random', save:bool=True, pbar:bool=True, silent:bool=True,  cluster:bool=False, replace_mech:Optional[str]=None,
nn_window:int=5, nn_nf:int=4, nn_pick:int=3, nn_sample_size:int=1, nn_epochs:int=10, nn_train_window:Optional[int]=None, batch_size:int=4,
D:int=30, runs:int=5, max_times:int=100, dropout:float=0.5):
    # Setting variables
    experiment_type = getattr(Experiment, experiment)
    method_type = getattr(Method, method)
    is_nn = method_type in [Method.NNnorm, Method.NNdrop, Method.NNtime, Method.NNconv]
    func_type = getattr(FuncName, func_name)
    scale_factor = getattr(ScaleFactor, scale_factor) #ScaleFactor[scale_factor]
    total_generations = int(max_times * frequency * 1_000_000 + 1_000)

    # Setting path
    path = Path(f'../../data/results/{experiment}/{func_name}')
    if cluster: path = Path(f'DENN/data/cluster_results/{experiment}/{func_name}') # this is for the cluster

    # If its neural network, consider more parameters
    if is_nn:
        path = path / 'nn'
        name = f'freq{freq_save}nn_w{nn_window}nn_p{nn_pick}nn_s{nn_sample_size}nn_tw{nn_train_window}nn_bs{batch_size}nn_epoch{nn_epochs}' #nn_s{nn_sample_size}nn_tw{nn_train_window}
    else:
        path = path / 'nonn'
        name = f'freq{freq_save}'

    # Set diversity method
    if diversity_method is not None:
        name += f'div{diversity_method}'
        diversity_method = DiversityMethod[diversity_method] 
    else:
        name += 'divNo'

    # Set scale factor
    if scale_factor == ScaleFactor.Random:
        beta_min = 0.2
        beta_max = 0.8
        CR = 0.3
    elif scale_factor == ScaleFactor.Constant:
        beta_min = 0.2
        beta_max = 0.2
        CR = 0.9
    else:
        raise Exception(f'Invalid scale_factor: {scale_factor}.')

    # Create output folder
    out_path = path / name
    out_path.mkdir(parents=True, exist_ok=True)

    # Set fitness and constraint functions
    fitness_func,constraint_func = get_functions(experiment_type, D, func_type)

    # Set an experiment name to save metrics
    experiment_name = f'{method}'
    if is_nn:
        experiment_name += f'_{replace_mech}'
        replace_type = getattr(ReplaceMechanism, replace_mech)

    # Read files
    if cluster:
        tmp_path = Path(f'DENN/data/results/{experiment}/{func_name}')
        ab = pd.read_csv(tmp_path/'dC_01.csv')['b'].values
        df = pd.read_csv(tmp_path/'best_known.csv')
        best_known_fitness = df['fitness'].values
        best_known_sumcv   = df['sum_constraints'].values
        best_known_positions = np.load(f'DENN/nbs/maryam/{experiment}_{func_name}.npy')
    else:
        ab = pd.read_csv(path.parent/'dC_01.csv')['b'].values
        df = pd.read_csv(path.parent/'best_known.csv')
        best_known_fitness = df['fitness'].values
        best_known_sumcv   = df['sum_constraints'].values
        best_known_positions = np.load(f'{experiment}_{func_name}.npy')

    # Initialize metrics
    results = {'mof':[], 'sr':[], 'nfe':[], 'fitness':[], 'sumcv':[], 'arr':[], 'nn_errors':[], 'distances':[]}
    if is_nn: results['nn_time'] = []

    # Run
    it = progress_bar(range(runs)) if pbar else range(runs)
    for run in it:
        callbacks = []
        if is_nn:
            if method_type==Method.NNnorm:
                model = SimpleModel(d=D, w=nn_window, nf=nn_nf) 
                nn_trainer = partial(NNTrainer, model=model, n=nn_pick, sample_size=nn_sample_size, window=nn_window,
                                     train_window=nn_train_window, replace_mechanism=replace_type, bs=batch_size, epochs=nn_epochs)
            if method_type==Method.NNtime:
                model = TimeModel(d=D, w=nn_window, nf=nn_nf) 
                nn_trainer = partial(NNTrainerTime, model=model, n=nn_pick, sample_size=nn_sample_size, window=nn_window,
                                     train_window=nn_train_window, replace_mechanism=replace_type, bs=batch_size, epochs=nn_epochs)
            if method_type==Method.NNdrop:
                model = DropoutModel(d=D, w=nn_window, nf=nn_nf, dropout=dropout) 
                nn_trainer = partial(NNTrainerNoNoise, model=model, n=nn_pick, sample_size=nn_sample_size, window=nn_window,
                                     train_window=nn_train_window, replace_mechanism=replace_type, bs=batch_size, epochs=nn_epochs)
            if method_type==Method.NNconv:
                model = ConvModel(d=D, w=nn_window, nf=16, ks=3, n_conv=3) 
                nn_trainer = partial(NNTrainer, model=model, n=nn_pick, sample_size=nn_sample_size, window=nn_window,
                                     train_window=nn_train_window, replace_mechanism=replace_type, bs=batch_size, epochs=nn_epochs, cuda=True)#cuda was true
            
            callbacks.append(nn_trainer)

        # Setting diversity method
        if diversity_method is None:
            evolve_mechanism = EvolveMechanism.Normal
        elif diversity_method == DiversityMethod.RI:
            evolve_mechanism = EvolveMechanism.Normal
            if is_nn: callbacks.append(partial(RandomImmigrantsOnChange,replacement_rate=2))
            else    : callbacks.append(partial(RandomImmigrantsOnChange,replacement_rate=7))
        elif diversity_method == DiversityMethod.Cw:
            evolve_mechanism = EvolveMechanism.Crowding
        elif diversity_method == DiversityMethod.Cwc:
            evolve_mechanism = EvolveMechanism.CrowdingCosine
        elif diversity_method == DiversityMethod.CwN:
            evolve_mechanism = EvolveMechanism.CrowdingN
        elif diversity_method == DiversityMethod.CwcN:
            evolve_mechanism = EvolveMechanism.CrowdingCosineN
        elif diversity_method == DiversityMethod.HMuPure:
            evolve_mechanism = EvolveMechanism.Normal
            callbacks.append(Hypermutation)
        elif diversity_method == DiversityMethod.HMu:
            evolve_mechanism = EvolveMechanism.Normal
            callbacks.append(Hypermutation)
            if is_nn: callbacks.append(partial(RandomImmigrantsOnChange,replacement_rate=2))
            else    : callbacks.append(partial(RandomImmigrantsOnChange,replacement_rate=7))
        elif diversity_method == DiversityMethod.Rst:
            evolve_mechanism = EvolveMechanism.Normal
            if is_nn: callbacks.append(partial(RandomImmigrantsOnChange,replacement_rate=15))
            else    : callbacks.append(partial(RandomImmigrantsOnChange,replacement_rate=20))
            # callbacks.append(OnChangeRestartPopulation)
        else: raise Exception(f'Invalid diversity method: {diversity_method}.')

        #first pop created here and passed to optimization
        population = Population.new_random(dimension=D)
        speed_metric = partial(SpeedMetric, threadhold=0.1)
        opt = Optimization(population, fitness_func, constraint_func, fitness_params=ab, constraint_params=[ab],
                           max_times=max_times, frequency=frequency, callbacks=callbacks, beta_min=beta_min, beta_max=beta_max, CR=CR,
                           metrics=[speed_metric, ModifiedOfflineError, OfflineError, AbsoluteRecoverRate, FirstDistance],
                           optimal_fitness_values=best_known_fitness, optimal_sum_constraints=best_known_sumcv,
                           optimal_positions=best_known_positions,
                           evolve_mechanism=evolve_mechanism)
        opt.run(total_generations, show_graph=False, show_report=False, silent=silent)

        # Store results
        results['mof'].append(opt.modified_offline_error.metrics)
        results['arr'].append(opt.absolute_recover_rate.metrics)
        results['sr'].append(opt.speed_metric.metrics)
        # results['nfe'].append(opt.speed_metric.speeds)
        results['fitness'].append(opt.recorder.best_times_fitness)
        results['sumcv'].append(opt.recorder.best_times_constraints)
        results['distances'].append(opt.first_distance.metrics)
        if is_nn:
            results['nn_time'].append(opt.nn_timer.metrics)
            results['nn_errors'].append(opt.nn_errors.metrics)

        yield run

    # Get results and save
    if save:
        pd.DataFrame({'mof':results['mof']}).to_csv(out_path/f'{experiment_name}_mof.csv', index=False)
        pd.DataFrame({'sr':results['sr']}).to_csv(out_path/f'{experiment_name}_sr.csv', index=False)
        # pd.DataFrame(results['nfe']).to_csv(out_path/f'{experiment_name}_nfe.csv', index=False)
        pd.DataFrame(results['fitness']).to_csv(out_path/f'{experiment_name}_fitness.csv', index=False)
        pd.DataFrame(results['sumcv']).to_csv(out_path/f'{experiment_name}_sumcv.csv', index=False)
        pd.DataFrame(results['distances']).to_csv(out_path/f'{experiment_name}_dis.csv', index=False)
        pd.DataFrame(results['arr']).to_csv(out_path/f'{experiment_name}_arr.csv', index=False)
        if is_nn:
            pd.DataFrame(results['nn_time']).to_csv(out_path/f'{experiment_name}_nn_time.csv', index=False)
            pd.DataFrame(results['nn_errors']).to_csv(out_path/f'{experiment_name}_nn_errors.csv', index=False)

if __name__ == '__main__':
    fire.Fire(main)