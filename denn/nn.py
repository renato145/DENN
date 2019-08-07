import torch
from torch import nn, functional as F, Tensor
from .imports import *
from .callbacks import *
from .optimization import *
from .metrics import NNTimer

__all__ = ['ReplaceMechanism', 'NNTrainer', 'NNTrainerNoNoise']

class ReplaceMechanism(IntEnum):
    '''Mechanism to replace individuals after a time change has been detected.
    - Random : Choose `n` random individuals to replace.
    - Closest: Choose the `n` closest individuals to replace.
    - Worst  : Choose the `n` worst individuals to replace.
    '''
    Random=1
    Closest=2
    Worst=3

class NNTrainer(Callback):
    _order = 10 # Needs to run after restarting the population 

    def __init__(self, optim:'Optimization', model:nn.Module, replace_mechanism:ReplaceMechanism=ReplaceMechanism.Random, n:int=3, noise_range:float=0.5,
                 window:int=5, min_batches:int=20, bs:int=4, epochs:int=10, loss_func:Callable=nn.MSELoss(), nn_optim:torch.optim.Optimizer=torch.optim.Adam):
        'TODO: add documentation'
        super().__init__(optim)
        self.model,self.replace_mechanism,self.n,self.noise_range,self.window,self.min_batches,self.bs,self.epochs,self.loss_func =\
             model,     replace_mechanism,     n,     noise_range,     window,     min_batches,     bs,     epochs,     loss_func
        self.data,self.train_losses = [],[]
        self.nn_optim = nn_optim(model.parameters())
        self.d = optim.population.dimension
        self.n_individuals = optim.population.n
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        optim.callbacks.append(NNTimer)
        optim.metrics.append(NNTimer)

    def _replace_random(self, preds:np.ndarray=1)->None:
        idxs = np.random.choice(self.n_individuals, size=self.n, replace=False)
        for i,idx in enumerate(idxs): self.optim.population[idx].data = preds[i]

    def _replace_closest(self, preds:np.ndarray)->None:
        idxs = []
        for pred in preds:
            this_idxs,_ = self.optim.population.get_closest(pred)
            this_idx = [idx for idx in this_idxs if idx not in idxs][0]
            idxs.append(this_idx)
        
        for i,idx in enumerate(idxs): self.optim.population[idx].data = preds[i]

    def _replace_worst(self, preds:np.ndarray)->None:
        idxs = [e.idx for e in self.optim.population.get_n_worse(self.n)]
        for i,idx in enumerate(idxs): self.optim.population[idx].data = preds[i]

    def modify_population(self, preds:np.ndarray)->None:
        if   self.replace_mechanism==ReplaceMechanism.Random : self._replace_random (preds)
        elif self.replace_mechanism==ReplaceMechanism.Closest: self._replace_closest(preds)
        elif self.replace_mechanism==ReplaceMechanism.Worst  : self._replace_worst  (preds)

    def on_detect_change_end(self, change_detected:bool, best:Individual, **kwargs:Any)->None:
        start_time = get_time()
        if change_detected:
            self.data.append(best.clone())
            if len(self.data)-self.window >= self.min_batches:
                self.do_train()
                self.apply_predictions()

        self.optim.nn_timer.times.append(get_time() - start_time)

    def on_run_end(self, **kwargs:Any)->None:
        self.model.eval()

    def get_train_data(self)->Tuple[Tensor,Tensor]:
        w,d = self.window,self.d
        data = torch.from_numpy(np.vstack([e.data for e in self.data])).float()
        data_x,data_y = [],[]
        for i in range(data.size(0)-w):
            data_x.append(data[i:i+w])
            data_y.append(data[i+w])

        return torch.stack(data_x).to(self.device),torch.stack(data_y).to(self.device)

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
            xb = torch.from_numpy(np.vstack([(e.data) for e in self.data[-self.window:]])).float()[None]
            pred = self.model.eval()(xb.to(self.device))[0]
            return pred.cpu()

    def apply_predictions(self)->None:
        # Get predictions
        preds = self.get_next_best().view(1,-1).repeat(self.n,1)
        # Add noise
        noise = torch.FloatTensor(*preds.shape).uniform_(-self.noise_range,self.noise_range)
        preds.add_(noise)
        preds = preds.numpy()
        # Modify population
        self.modify_population(preds)

class NNTrainerNoNoise(NNTrainer):
    def apply_predictions(self)->None:
        # Get predictions
        preds = [self.get_next_best() for _ in repeat(None, self.n)]
        preds = torch.stack(preds, dim=0).numpy()
        # Modify population
        self.modify_population(preds)

