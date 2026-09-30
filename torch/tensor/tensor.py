import numpy as np
from torch.backend import cp, HAS_CUPY
from core.tensor_impl import TensorImpl
from torch.ops.elementwise import Add, Sub, Mul, Matmul, Neg, Truediv

# neg, pow, truediv, relu, sigmoid, tanh, sum, mean, reshape, T


class Tensor:
    
    __array_ufunc__ = None
    def __init__(self, data, requires_grad=False, device='cpu'):
        self._impl = TensorImpl(data, device)
        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = None
        self._version = 0
    
    @property
    def data(self):
        return self._impl.data
    
    @data.setter
    def data(self, value):
        if isinstance(value, Tensor):
            value = value.data
        self._impl = TensorImpl(value, self.device)
        self._version += 1
    
    @property
    def device(self):
        return self._impl.device
    
    def backward(self, grad = None):
        if not self.requires_grad:
            raise RuntimeError("element 0 of tensors does not require grad and does not have a grad_fn")
        if grad is None:
            if grad.data.shape != 1:
                raise RuntimeError("grad can be implicitly created only for scalar outputs")
            xp = cp if self.device == 'cuda' else np
            grad = xp.ones_like(self.data)
        else:
            if isinstance(grad, Tensor):
                grad = grad.data
            grad = TensorImpl(grad, device=self.device).data
            if grad.shape != self.data.shape:
                raise RuntimeError(
                    f"Mismatch in shape: grad has shape {grad.shape}, "
                    f"but output has shape {self.data.shape}"
                )
            
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
            if t._ctx is None or not t.requires_grad:
                continue
            for parent, saved_v in zip(t._ctx.parents, t._ctx.saved_versions):
                if parent._version != saved_v:
                    raise RuntimeError(
                        "one of the variable needed for gradient computation"
                        "has been updated by an inplace operation"
                    )
            grads = t._ctx.backward(t.grad)
            for parent, grad_contribution in zip(t._ctx.parents, grads):
                if not parent.requires_grad:
                    continue
                if parent.grad is None:
                     parent.grad = grad_contribution
                else:
                    parent.grad = parent.grad + grad_contribution
    
    
    def __repr__(self):
        return f"Tensor: {self.data}, requires_grad: {self.requires_grad}"
    
    def _check_same_device(self, other):
                if isinstance(other, Tensor) and self.device != other.device:
                    raise ValueError(
                        f"Cannot combine tensors on different devices: "
                        f"'{self.device}' vs '{other.device}'. Move one of them with "
                        f".to('{self.device}') first."
                    )
    
    def _to_tensor(self, other):
        return other if isinstance(other, Tensor) else Tensor(other, device=self.device)
    
    def _apply(self, OpClass, *inputs):
        for t in inputs:
            self._check_same_device(t)
        op = OpClass(*inputs)
        result_data = op.forward(*(t.data for t in inputs))
        requires_grad = any(t.requires_grad for t in inputs)
        result = Tensor(result_data, requires_grad=requires_grad, device=self.device)
        if requires_grad:
            result._ctx = op
        return result
    
    def __add__(self, other):   return self._apply(Add, self, self._to_tensor(other))
    def __add__(self, other):      return self._apply(Add, self, self._to_tensor(other))
    def __radd__(self, other):     return self._apply(Add, self._to_tensor(other), self)
    def __sub__(self, other):      return self._apply(Sub, self, self._to_tensor(other))
    def __rsub__(self, other):     return self._apply(Sub, self._to_tensor(other), self)
    def __mul__(self, other):      return self._apply(Mul, self, self._to_tensor(other))
    def __rmul__(self, other):     return self._apply(Mul, self._to_tensor(other), self)
    def __truediv__(self, other):  return self._apply(Truediv, self, self._to_tensor(other))
    def __rtruediv__(self, other): return self._apply(Truediv, self._to_tensor(other), self)
    def __matmul__(self, other):   return self._apply(Matmul, self, self._to_tensor(other))
    def __rmatmul__(self, other):  return self._apply(Matmul, self._to_tensor(other), self)
    def __neg__(self):             return self._apply(Neg, self)