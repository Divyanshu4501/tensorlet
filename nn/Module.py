from tensor import Tensor
import numpy as np
from backend import cp

class Module:
    def __init__(self):
        self.device = "cpu"

    def zero_grad(self):
        for p in self.parameters():
            if p.requires_grad:
                xp = cp if isinstance(p.data, cp.ndarray) else np
                p.grad = xp.zeros_like(p.data)

    def parameters(self):
        params = []
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor) and value.requires_grad:
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
        
        return params
    
    def to(self, device):
        self.device = device
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor):
                setattr(self, name, value.to(device))
            elif isinstance(value, Module):
                value.to(device)
        return self
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
