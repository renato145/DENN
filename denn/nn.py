import torch
from torch import nn, functional as F, Tensor
from .imports import *
from .callbacks import *
from .optimization import *

__all__ = ['NNTrainer']

class NNTrainer(Callback):
    def __init__(self, optim:'Optimization', model:nn.Module, window:int=5, min_batches:int=20, bs:int=4,
                 loss_func:Callable=nn.MSELoss(), nn_optim:torch.optim.Optimizer=torch.optim.Adam):
        'TODO: add documentation'
        super().__init__(optim)
        self.model,self.window,self.min_batches,self.bs,self.loss_func = model,window,min_batches,bs,loss_func
        self.data,self.train_losses = [],[]
        self.nn_optim = nn_optim(model.parameters())
        self.d = optim.population.dimension
        
    def on_detect_change_end(self, change_detected:bool, best:Individual, **kwargs:Any):
        if change_detected:
            self.data.append(best.clone())
            if len(self.data)-self.window >= self.min_batches:
                self.do_train()

    def on_run_end(self, **kwargs):
        self.model.eval()

    def get_train_data(self)->Tuple[Tensor,Tensor]:
        w,d = self.window,self.d
        data = torch.from_numpy(np.vstack([e.data for e in self.data])).float()
        data_x,data_y = [],[]
        for i in range(data.size(0)-w):
            data_x.append(data[i:i+w])
            data_y.append(data[i+w])
        
        return torch.stack(data_x),torch.stack(data_y)

    def do_train(self)->None:
        bs,model,loss_func,nn_optim = self.bs,self.model,self.loss_func,self.nn_optim
        model.train()
        data_x,data_y = self.get_train_data()
        n_batches = math.ceil(data_x.size(0)/bs)
        # Train loop
        for i in range(n_batches):
            xb,yb = data_x[i*bs:(i+1)*bs],data_y[i*bs:(i+1)*bs]
            yb_ = model(xb)
            loss = loss_func(yb,yb_)
            loss.backward()
            nn_optim.step()
            nn_optim.zero_grad()
            self.train_losses.append(loss.detach().cpu())

    def get_next_best(self)->Individual:
        # TODO
        return 0
