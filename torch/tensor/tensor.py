import numpy as np
from torch.backend import cp, HAS_CUPY
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
    
    def backward(self, grad = None):
        if not self.requires_grad:
            raise RuntimeError("element 0 of tensors does not require grad and does not have a grad_fn")
        if grad is None:
            grad = np.ones_like(self.data) if self.device == 'cpu' else cp.ones_like(self.data)
        else:
            self.grad = np.array(grad, dtype=np.float64) if self.device == 'cpu' else cp.array(grad, dtype=cp.float64)
            
        self.grad = grad
        
        topo_order = []
        visited = set()
        
        def build_topo(tensor):
            if tensor not in visited:
                visited.add(tensor)
                if tensor._ctx is not None: 
                    for parent in tensor._ctx.parents:
                       build_topo(parent) 
                topo_order.append(tensor)
        
        build_topo(self)
        for t in reversed(topo_order):
            if t._ctx is not None:
                 grads = t._ctx.backward(t.grad)
                 for parent, grad_contribution in zip(t._ctx.parents, grads):
                    if parent.grad is None:
                         parent.grad = grad_contribution
                    else:
                        parent.grad = parent.grad + grad_contribution
    
    
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