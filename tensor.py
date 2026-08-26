import numpy as np
import cupy as cp

from operations import Add, Mul, MatMul, Transpose, Neg, Sub, Pow, Div, ReLU, Sigmoid, Tanh, Sum, Mean, Reshape

class Tensor:
    def __init__(self, data, _ctx = None, requires_grad = False, device = 'cpu'):
        self._ctx = _ctx
        self.requires_grad = requires_grad
        self.device = device
        
        if isinstance(data, (np.darray, cp.darray)):
            self.data = data
        else:
            self.data = np.array(data, dtype= np.float64) if device == 'cpu' else cp.array(data, dtype= cp.float64)
            
        
        if self.requires_grad:
            self.grad = np.zeros_like(self.data) if device == 'cpu' else cp.zeros_like(self.data)
        else:
            self.grad = None
            
    def to(self, device):
        if self.device == device:
            return self
        
        if device == 'cuda':
            new_data = cp.asarray(self.data)
        else:
            new_data = cp.asnumpy(self.data)
            
        new_tensor = Tensor(new_data, _ctx=self._ctx, requires_grad=self.requires_grad, device=self.device)
        
        if self.grad is not None:
            new_tensor.grad = cp.asarray(self.grad) if device == 'cuda' else cp.asnumpy(self.grad)
            
        return new_tensor
        
    def __repr__(self):
        return f"Tensor: {self.data}, requires_grad = {self.requires_grad}"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        op = Add(self, other)
        result_data = op.forward(self.data, other.data)
        requires_grad = self.requires_grad or other.requires_grad
        
        return Tensor(result_data, _ctx = op, requires_grad=requires_grad)
    
    def sum(self):
        op = Sum(self)
        out_data = op.forward(self.data)
        return Tensor(out_data, _ctx = op ,requires_grad=self.requires_grad)
        

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        op = Mul(self, other)
        result_data = op.forward(self.data, other.data)
        requires_grad = self.requires_grad or other.requires_grad
        
        return Tensor(result_data, _ctx = op, requires_grad=requires_grad)
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        op = MatMul(self, other)
        result_data = op.forward(self.data, other.data)
        requires_grad = self.requires_grad or other.requires_grad
        
        return Tensor(result_data, _ctx = op, requires_grad=requires_grad)
    
    def __neg__(self):
        op = Neg(self)
        result_data = op.forward(self.data)
        return Tensor(result_data, _ctx=op, requires_grad=self.requires_grad)

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        op = Sub(self, other)
        result_data = op.forward(self.data, other.data)
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(result_data, _ctx=op, requires_grad=requires_grad)

    def __pow__(self, exponent):
        op = Pow(self)
        result_data = op.forward(self.data, exponent)
        return Tensor(result_data, _ctx=op, requires_grad=self.requires_grad)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        op = Div(self, other)
        result_data = op.forward(self.data, other.data)
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(result_data, _ctx=op, requires_grad=requires_grad)

    def relu(self):
        op = ReLU(self)
        result_data = op.forward(self.data)
        return Tensor(result_data, _ctx=op, requires_grad=self.requires_grad)

    def sigmoid(self):
        op = Sigmoid(self)
        result_data = op.forward(self.data)
        return Tensor(result_data, _ctx=op, requires_grad=self.requires_grad)

    def tanh(self):
        op = Tanh(self)
        result_data = op.forward(self.data)
        return Tensor(result_data, _ctx=op, requires_grad=self.requires_grad)

    def sum(self):
        op = Sum(self)
        result_data = op.forward(self.data)
        return Tensor(result_data, _ctx=op, requires_grad=self.requires_grad)

    def mean(self):
        op = Mean(self)
        result_data = op.forward(self.data)
        return Tensor(result_data, _ctx=op, requires_grad=self.requires_grad)

    def reshape(self, shape):
        op = Reshape(self)
        result_data = op.forward(self.data, shape)
        return Tensor(result_data, _ctx=op, requires_grad=self.requires_grad)
    
    @property
    def T(self):
        op = Transpose(self)
        result_data = op.forward(self.data)
        return Tensor(result_data, _ctx = op, requires_grad=self.requires_grad)
    
    def backward(self, grad = None):
        if not self.requires_grad:
            raise RuntimeError("element 0 of tensors does not require grad and does not have a grad_fn")
        if grad is None:
            self.grad = np.ones_like(self.data, dtype=np.float64)
        else:
            self.grad = np.array(grad, dtype=np.float64)
            
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
                t._ctx.backward(t.grad)