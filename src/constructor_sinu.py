import numpy as np
import torch

def sinu_constructed(x_min,x_max,numero_muestras,Amplitud,w,desfase=0):
    x = np.linspace(x_min,x_max,numero_muestras)
    y = Amplitud*np.sin(w*x + desfase)
    return x,y

def set_device ():
    if torch.backends.mps.is_available(): # para usar con mac silicon
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    torch.set_default_device(device)
    print(f"Using {torch.device(device)} device")
    return torch.device(device)

def make_windows_1step(y, L=10):
    # y: (T,)
    X, Y = [], []
    for i in range(len(y) - L):
        X.append(y[i:i+L])
        Y.append(y[i+L])
    X = torch.stack(X).unsqueeze(-1)  # (N, L, 1)
    Y = torch.stack(Y).unsqueeze(-1)  # (N, 1)
    return X, Y