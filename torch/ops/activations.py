import numpy as np
from torch.backend import HAS_CUPY, cp
from torch.autograd.function import Function

def _xp(a):
    return cp if HAS_CUPY and isinstance(a, cp.ndarray) else np

class Relu(Function):
    def forward(self, a):
        mask = a > 0
        self.save_for_backward(mask)
        return _xp(a).maximum(a, 0)
    
    def backward(self, grad_output):
        (mask,) = self.saved_tensors
        return (grad_output*mask, )
    
class Sigmoid(Function):
    def forward(self, a):
        xp = _xp(a)
        e = xp.exp(-xp.abs(a))
        out = xp.where(a>=0, 1/(1+e), e/(1+e))
        self.save_for_backward(out)
        return out
        
    def backward(self, grad_output):
        (s,) = self.saved_tensors
        return (grad_output*s*(1-s),)
    
class Tanh(Function):
    def forward(self, a):
        out = _xp(a).tanh(a)
        self.save_for_backward(out)
        return out
    
    def backward(self, grad_output):
        t = self.saved_tensors
        return (grad_output*(1 - t*t),)