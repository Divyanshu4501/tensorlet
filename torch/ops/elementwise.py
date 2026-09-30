import numpy as np
from torch.backend import cp, HAS_CUPY
from torch.autograd.function import Function

def unbroadcast(grad, target_shape):
    ndims_added = grad.ndim - len(target_shape)
    xp = cp if HAS_CUPY and isinstance(grad, cp.ndarray) else np
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
        a_shape, b_shape = self.saved_tensors
        grad_a = unbroadcast(grad_output, a_shape)
        grad_b = unbroadcast(grad_output, b_shape)
        return grad_a, grad_b
    
class Sub(Function):
    def forward(self, a, b):
        self.save_for_backward(a.shape, b.shape)
        return a - b
    
    def backward(self, grad_output):
        a_shape, b_shape = self.saved_tensors
        grad_a = unbroadcast(grad_output, a_shape)
        grad_b = unbroadcast(-grad_output, b_shape)
        return grad_a, grad_b

class Mul(Function):
    def forward(self, a, b):
        self.save_for_backward(a, b)
        return a*b
    
    def backward(self, grad_output):
        a, b = self.saved_tensors
        grad_a = unbroadcast(grad_output*b, a.shape)
        grad_b = unbroadcast(grad_output*a, b.shape)  
        return grad_a, grad_b
    
class Matmul(Function):
    def forward(self, a, b):
        self.save_for_backward(a, b)
        return a@b
            
    def backward(self, grad_output):
        a, b = self.saved_tensors
        grad_a =  grad_output @ b.T
        grad_b =  a.T  @ grad_output
        return grad_a, grad_b
    
class Neg(Function):
    def forward(self, a):
        return -a
    
    def backward(self, grad_output):
        return -grad_output
    
class Truediv(Function):
    def forward(self, a, b):
        self.save_for_backward(a, b)
        return a/b
    
    def backward(self, grad_output):
        a, b = self.saved_tensors
        grad_a = unbroadcast(grad_output/b, a.shape)
        grad_b = unbroadcast(grad_output*(-a/(b**2)), b.shape)
        return grad_a, grad_b
    
class Pow(Function):
    def forward(self, a, exponent):
        self.save_for_backward(a, exponent)
        return a**exponent
    
    def backward(self, grad_output):
        a, n = self.saved_tensors
        if n == 0:
            return (grad_output*0.0, )
        return (grad_output*n*a**(n-1),)