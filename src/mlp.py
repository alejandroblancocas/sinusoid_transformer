import torch
from torch import nn
import numpy as np
class BaseSinuRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first
        )

        self.fc = nn.Linear(hidden_size, 1)
    

    def forward(self,x,h0=None,):
        out, _ = self.rnn(x,h0)
        if self.batch_first:
            out = out[:,-1,:]
        else:
            out = out[-1,:,:]
        out = self.fc(out)
        return out
    
class TensorLoader: # class to get batches from tensors X, Y; much more efficient than DataLoader
    def __init__ (self, X, Y, bs=None):
        self.X = X
        self.Y = Y
        self.p0 = 0
        self.len = len(X)
        self.bs = bs if bs else self.len
        self.nbatch = self.len // self.bs
        if self.len % self.bs != 0: self.nbatch += 1 # para range de bucles for
        self.ite = 0

    def next (self):
        pf = self.p0 + self.bs
        if pf > self.len: pf = self.len
        resX = self.X[self.p0:pf, ...]
        resY = self.Y[self.p0:pf, ...]
        self.p0 += self.bs
        self.ite += 1
        if self.p0 >= self.len: 
            self.p0 = self.ite = 0
        return resX, resY
    
    @staticmethod   
    def train (X, Y, model, loss_fn, optimizer, epochs=1000, bs=None, device='cpu', trace=100): # train con batches usando TensorLoader
        dataloader = TensorLoader(X, Y, bs=bs) 
        model = model.to(device)
        model.train() # indica que estamos entrenando (para capas Dropout y BatchNorm), el opuesto es model.eval
        for e in range(epochs):
            optimizer.zero_grad() # reset gradients
            acum_loss = 0
            for _ in range(dataloader.nbatch): # para cada batch
                X, Y = dataloader.next()
                X, Y = X.to(device,dtype=torch.float32), Y.to(device,dtype=torch.float32)
                pred = model(X) # propagate
                loss = loss_fn(pred, Y) # prediction error
                acum_loss += loss.item()
                loss.backward() # back propagation
                optimizer.step() # update parameters

            if (e+1) % trace == 0: # traces
                print(f"loss: {acum_loss/dataloader.nbatch:>7f} [{(e+1):>5d} /{epochs:>5d}]")
    @staticmethod
    def test (X, Y, model, metric_list, bs=None, device='cpu'): # test con batches usando TensorLoader
        dataloader = TensorLoader(X, Y, bs=bs) 
        model = model.to(device)
        model.eval() # indica que estamos evaluando (para capas Dropout y BatchNorm)
        acum = np.zeros(len(metric_list))
        for _ in range(dataloader.nbatch): # para cada batch
            X, Y = dataloader.next()
            X, Y = X.to(device,dtype=torch.float32), Y.to(device,dtype=torch.float32)
            pred = model(X) # propagate
            for i,m in enumerate(metric_list):
                v = m(pred,Y).item()
                acum[i] += v

        acum /= dataloader.nbatch
        [print(f"{x:.6f}", end='; ') for x in acum]
        return acum
        
        
        
        
        
    
    