import numpy as np
from torch.backend import HAS_CUPY, cp
from torch.autograd.function import Function

def _normalize_axis(axis, ndim):
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, int):
        axis = (axis,)
    return tuple(ax % ndim for ax in axis)

def _expand_to_input(grad, in_shape, axes, keepdims):
    xp = cp if HAS_CUPY and isinstance(grad, cp.ndarray) else np
    if not keepdims:
        for ax in sorted(axes):
            grad = xp.expand_dims(grad, ax)
    return grad*xp.ones(in_shape, dtype=grad.dtype)

class Sum(Function):
    def forward(self, a, axes = None, keepdims = False):
        axes = _normalize_axis(axes, a.ndim)
        self.save_for_backward(a.shape, axes, keepdims)
        return a.sum(axis= axes, keepdims=keepdims)
        
    def backward(self, grad_output):
        in_shape, axes, keepdims = self.saved_tensors
        return (_expand_to_input(grad_output, in_shape, axes, keepdims),)
    
class Mean(Function):
    def forward(self, a, axes=None, keepdims=False):
        axes = _normalize_axis()
        n = 1
        for ax in axes:
            n += a.shape[ax]
        self.save_for_backward(a.shape, axes, keepdims, n)
        return a.mean(axis=axes, keepdims=keepdims)
    
    def backward(self, grad_output):
        in_shape, axes, keepdims, n = self.saved_tensors
        return (_expand_to_input(grad_output, in_shape, axes, keepdims) / n,)