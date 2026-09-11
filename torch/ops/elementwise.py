import numpy as np
import cupy as cp
from torch.autograd.function import Function

def unbroadcast(grad, target_shape):
    ndims_added = grad.ndim - len(target_shape)
    xp = cp if isinstance(grad, cp.ndarray) else np
    if ndims_added > 0:
        grad = xp.sum(grad, axis = tuple(range(ndims_added)))
    
    for i, dim_size in enumerate(target_shape):
        if dim_size == 1 and grad.shape[i] > 1:
            grad = xp.sum(grad, axis = i, keepdims=True)
            
    return grad

class Add(Function):
    def forward(self, a, b):
        self.save_for_backward(a.shape, b.shape)
        return a + b
    def backward(self, grad_output):
        pass