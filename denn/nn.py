import torch
from torch import nn, functional as F, Tensor
from itertools import product
from .imports import *
from .callbacks import *
from .optimization import *
from .metrics import NNTimer, NNErrors

__all__ = ['ReplaceMechanism', 'NNTrainer', 'NNTrainerNoNoise', 'NNTrainerTime']

class ReplaceMechanism(IntEnum):
    '''Mechanism to replace individuals after a time change has been detected.
    - Random : Choose `n` random individuals to replace.
    - Closest: Choose the `n` closest individuals to replace.
    - Worst  : Choose the `n` worst individuals to replace.
    '''
    Random=1
    Closest=2
    Worst=3

class SamplingMethod(IntEnum):
    '''Method to sample individuals to get training examples:
    - All   : This will compute all possible combinatoins.
    - Limit : This will compute a fixed number of combinations.
    '''
    All=1
    Limit=2

class NNTrainer(Callback):
    def __init__(self, optim:'Optimization', model:nn.Module, replace_mechanism:ReplaceMechanism=ReplaceMechanism.Random, n:int=3, noise_range:float=0.5,
                 sample_size:int=1, window:int=5, min_batches:int=20, train_window:Optional[int]=None, bs:int=4, epochs:int=10,
                 loss_func:Callable=nn.MSELoss(), nn_optim:torch.optim.Optimizer=torch.optim.Adam,
                 sampling_method:SamplingMethod=SamplingMethod.Limit, sampling_limit:int=256, cuda:bool=False):#sampling_limit:int=128
        '''Uses neural network to initialize individuals after a change is detected.
        Params:
          - optim: Optimization class.
          - model: Neural network model.
          - replace_mechanism: ReplaceMechanism.
          - n: Number of individuals to predict and replace.
          - noise_range: Noise to add to the predictions.
          - sample_size: The number of individuals to take at each time change.
          - window: Number of past individuals to predict the next one.
          - min_batches: Minimun amount of batches to start training.
          - train_window: Number of past samples to train with (if not given, it will use all the traininig data).
          - bs: Batch size for the neural network.
          - epochs: Number of epochs to train the neural network at each time change.
          - loss_func: Loss function.
          - nn_optim : Optimizer for the neural network.
          - sampling_method: check `SamplingMethod`.
          - sampling_limit: used in case `sampling_method` is `SamplingMethod.Limit`.
          '''
        super().__init__(optim)
        self.model,self.replace_mechanism,self.n,self.noise_range,self.sample_size,self.window,self.min_batches,self.train_window,self.bs,self.epochs,self.loss_func =\
             model,     replace_mechanism,     n,     noise_range,     sample_size,     window,     min_batches,     train_window,     bs,     epochs,     loss_func
        self.sampling_method,self.sampling_limit = sampling_method,sampling_limit
        self.data,self.train_losses = [],[]
        self.nn_optim = nn_optim(model.parameters())
        self.d = optim.population.dimension
        self.n_individuals = optim.population.n
        self.device = torch.device('cuda') if cuda else torch.device('cpu')
        self.model.to(self.device)
        self.batch_per_time = self.sample_size**self.window
        self.data_x,self.data_y = [],[]
        self.sample_cache = None # this will store all the combinations for sampling
        self.lims = self.optim.population.lower_limit,self.optim.population.upper_limit
        optim.callbacks.append(NNTimer)
        optim.metrics.append(NNTimer)
        optim.callbacks.append(NNErrors)
        optim.metrics.append(NNErrors)

    def _replace_random(self, preds:np.ndarray)->Collection[int]:
        idxs = np.random.choice(self.n_individuals, size=self.n, replace=False)
        return idxs

    def _replace_closest(self, preds:np.ndarray)->Collection[int]:
        idxs = []
        for pred in preds:
            this_idxs,_ = self.optim.population.get_closest(pred)
            this_idx = [idx for idx in this_idxs if idx not in idxs][0]
            idxs.append(this_idx)

        return idxs

    def _replace_worst(self, preds:np.ndarray)->Collection[int]:
        idxs = [e.idx for e in self.optim.population.get_n_worse(self.n)]
        return idxs

    def modify_population(self, preds:np.ndarray)->Collection[int]:
        preds = preds.clip(*self.lims)
        if   self.replace_mechanism==ReplaceMechanism.Random : idxs = self._replace_random (preds)
        elif self.replace_mechanism==ReplaceMechanism.Closest: idxs = self._replace_closest(preds)
        elif self.replace_mechanism==ReplaceMechanism.Worst  : idxs = self._replace_worst  (preds)
        for i,idx in enumerate(idxs): self.optim.population[idx].data = preds[i]
        return idxs

    def on_detect_change_end(self, change_detected:bool, time:int, **kwargs:Any)->dict:
        start_time = get_time()
        idxs = []
        if change_detected:
            this_bests = [indiv.clone() for indiv in self.optim.population.get_n_best(self.sample_size)]
            self.data.append(this_bests)
            if (len(self.data)-self.window-1) > 0: self.update_data()
            if self.batch_per_time*(len(self.data)-self.window-1) >= self.min_batches:
                self.do_train()
                idxs = self.apply_predictions()
                this_time_predictions = [self.optim.population[i].data for i in idxs]
                if time<(self.optim.max_times-1):
                    next_best = self.optim.optimal_positions[time+1]
                    error = np.mean([np.linalg.norm(pred-next_best) for pred in this_time_predictions])
                    self.optim.nn_errors.metrics.append(error)

        self.optim.nn_timer.times.append(get_time() - start_time) # Check this
        return {'detected_idxs':idxs}

    def on_run_end(self, **kwargs:Any)->None:
        self.model.eval()

    def update_data(self)->None:
        w = self.window
        data_x = []
        this_x = self.data[-(w+1):-1]

        if self.sample_cache is None: self.sample_cache = list(product(range(self.sample_size), repeat=w))

        if self.sampling_method==SamplingMethod.All:
            samples = self.sample_cache
        elif self.sampling_method == SamplingMethod.Limit:
            samples = self.sample_cache
            if len(samples) > self.sampling_limit:
                picked_idxs = np.random.choice(len(samples), size=self.sampling_limit, replace=False)
                samples = [samples[i] for i in picked_idxs]

        for idxs in samples:
            tx = torch.from_numpy(np.vstack([x[idx].data for idx,x in zip(idxs,this_x)])).float()
            data_x.append(tx)

        data_x = torch.stack(data_x).float()
        data_y = torch.from_numpy(self.data[-1][0].data).float() # The best individual of the last time
        self.data_x.append(data_x)
        self.data_y.append(data_y.repeat(data_x.size(0),1))
        # Apply train_window
        if self.train_window is not None:
            self.data_x = self.data_x[-self.train_window:]
            self.data_y = self.data_y[-self.train_window:]

    def get_train_data(self)->Tuple[Tensor,Tensor]:
        return torch.cat(self.data_x).to(self.device),torch.cat(self.data_y).to(self.device)

    def do_train(self)->None:
        bs,epochs,model,loss_func,nn_optim = self.bs,self.epochs,self.model,self.loss_func,self.nn_optim
        model.train()
        data_x,data_y = self.get_train_data()
        n_batches = math.ceil(data_x.size(0)/bs)
        # Train loop
        losses = [] # TODO: make this a running average
        for epoch in range(epochs):
            for i in range(n_batches):
                xb,yb = data_x[i*bs:(i+1)*bs],data_y[i*bs:(i+1)*bs]
                yb_ = model(xb)
                loss = loss_func(yb,yb_)
                loss.backward()
                nn_optim.step()
                nn_optim.zero_grad()
                losses.append(loss.detach().cpu())

        self.train_losses.append(np.mean(losses))

    def get_next_best(self)->Tensor:
        with torch.no_grad():
            xb = torch.from_numpy(np.vstack([(e[0].data) for e in self.data[-self.window:]])).float()[None]
            pred = self.model.eval()(xb.to(self.device))[0]
            return pred.cpu()

    def apply_predictions(self)->Collection[int]:
        # Get predictions
        preds = self.get_next_best().view(1,-1).repeat(self.n,1)
        # Add noise
        noise = torch.FloatTensor(*preds.shape).uniform_(-self.noise_range,self.noise_range)
        preds.add_(noise)
        preds = preds.numpy()
        # Modify population
        return self.modify_population(preds)
    
class NNTrainerNoNoise(NNTrainer):
    'This method is used as the model for dropout'
    def apply_predictions(self)->Collection[int]:
        # Get predictions
        preds = [self.get_next_best() for _ in repeat(None, self.n)]
        preds = torch.stack(preds, dim=0).numpy()
        # Modify population
        return self.modify_population(preds)

class NNTrainerTime(NNTrainer):
    'This method gives the NN a time parameter `m(x,y)`.'
    def __init__(self, optim:'Optimization', model:nn.Module, replace_mechanism:ReplaceMechanism=ReplaceMechanism.Random, n:int=3, noise_range:float=0.5,
                 sample_size:int=1, window:int=5, min_batches:int=20, train_window:Optional[int]=None, bs:int=4, epochs:int=10,
                 loss_func:Callable=nn.MSELoss(), nn_optim:torch.optim.Optimizer=torch.optim.Adam,
                 sampling_method:SamplingMethod=SamplingMethod.Limit, sampling_limit:int=128, cuda:bool=False):
        super().__init__(optim, model=model, replace_mechanism=replace_mechanism, n=n, noise_range=noise_range,
                 sample_size=sample_size, window=window, min_batches=min_batches, train_window=train_window, bs=bs, epochs=epochs,
                 loss_func=loss_func, nn_optim=nn_optim, sampling_method=sampling_method, sampling_limit=sampling_limit, cuda=cuda)
        self.data_time = []

    def update_data(self)->None:
        time = self.optim.state_dict['time']
        w = self.window
        data_x = []
        this_x = self.data[-(w+1):-1]

        if self.sample_cache is None: self.sample_cache = list(product(range(self.sample_size), repeat=w))

        if self.sampling_method==SamplingMethod.All:
            samples = self.sample_cache
        elif self.sampling_method == SamplingMethod.Limit:
            samples = self.sample_cache
            if len(samples) > self.sampling_limit:
                picked_idxs = np.random.choice(len(samples), size=self.sampling_limit, replace=False)
                samples = [samples[i] for i in picked_idxs]

        for idxs in samples:
            tx = torch.from_numpy(np.vstack([x[idx].data for idx,x in zip(idxs,this_x)])).float()
            data_x.append(tx)

        data_x = torch.stack(data_x).float()
        data_y = torch.from_numpy(self.data[-1][0].data).float() # The best individual of the last time
        self.data_x.append(data_x)
        self.data_y.append(data_y.repeat(data_x.size(0),1))
        self.data_time.append(torch.LongTensor([time]))
        # Apply train_window
        if self.train_window is not None:
            self.data_x = self.data_x[-self.train_window:]
            self.data_y = self.data_y[-self.train_window:]
            self.data_time = self.data_time[-self.train_window:]

    def get_train_data(self)->Tuple[Tensor,Tensor,Tensor]:
        return torch.cat(self.data_x).to(self.device),torch.cat(self.data_time).to(self.device),torch.cat(self.data_y).to(self.device)

    def do_train(self)->None:
        bs,epochs,model,loss_func,nn_optim = self.bs,self.epochs,self.model,self.loss_func,self.nn_optim
        model.train()
        data_x,data_time,data_y = self.get_train_data()
        n_batches = math.ceil(data_x.size(0)/bs)
        # Train loop
        losses = [] # TODO: make this a running average
        for epoch in range(epochs):
            for i in range(n_batches):
                xb,timeb,yb = data_x[i*bs:(i+1)*bs],data_time[i*bs:(i+1)*bs],data_y[i*bs:(i+1)*bs]
                yb_ = model(xb,timeb)
                loss = loss_func(yb,yb_)
                loss.backward()
                nn_optim.step()
                nn_optim.zero_grad()
                losses.append(loss.detach().cpu())

        self.train_losses.append(np.mean(losses))

    def get_next_best(self)->Tensor:
        with torch.no_grad():
            xb = torch.from_numpy(np.vstack([(e[0].data) for e in self.data[-self.window:]])).float()[None]
            timeb = torch.LongTensor([self.optim.state_dict['time']+1])
            pred = self.model.eval()(xb.to(self.device), timeb.to(self.device))[0]
            return pred.cpu()
