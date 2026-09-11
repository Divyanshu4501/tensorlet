import numpy as np
from core.tensor_impl import TensorImpl
from torch.ops.elementwise import Add


class Tensor:
    def __init__(self, data, requires_grad=False, device='cpu'):
        self._impl = TensorImpl(data, device)
        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = None
    
    @property
    def data(self):
        return self._impl.data
    
    @property
    def device(self):
        return self._impl.device
    
    def __repr__(self):
        return f"Tensor: {self.data}, requires_grad: {self.requires_grad}"
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        op = Add(self, other)
        result_data = op.forward(self.data, other.data)
        requires_grad = self.requires_grad or other.requires_grad
        
        result = Tensor(result_data, requires_grad=requires_grad, device=self.device)
        result._ctx = op
        return result
        