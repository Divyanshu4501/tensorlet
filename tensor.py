import numpy as np
from backend import cp, HAS_CUDA_BACKEND

from operations import Add, Mul, MatMul, Transpose, Neg, Sub, Pow, Div, ReLU, Sigmoid, Tanh, Sum, Mean, Reshape

class Tensor:
    def __init__(self, data, _ctx = None, requires_grad = False, device = 'cpu'):
        if device == 'cuda' and not HAS_CUDA_BACKEND:
            raise RuntimeError(
                "Requested device='cuda' but cupy/triton aren't installed in "
                "this environment. Install them (matching this machine's CUDA "
                "version) or use device='cpu'."
            )

        self._ctx = _ctx
        self.requires_grad = requires_grad
        self.device = device
        
        if device == 'cpu':
            self.data = cp.asnumpy(data).astype(np.float64) if isinstance(data, cp.ndarray) else np.array(data, dtype=np.float64)
        else:
            self.data = data.astype(cp.float64) if isinstance(data, cp.ndarray) else cp.array(data, dtype=cp.float64)
            
        
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
            
        new_tensor = Tensor(new_data, _ctx=self._ctx, requires_grad=self.requires_grad, device=device)
        
        if self.grad is not None:
            new_tensor.grad = cp.asarray(self.grad) if device == 'cuda' else cp.asnumpy(self.grad)
            
        return new_tensor
        
    def __repr__(self):
        return f"Tensor: {self.data}, requires_grad = {self.requires_grad}"

    def _check_same_device(self, other):
        if isinstance(other, Tensor) and other.device != self.device:
            raise ValueError(
                f"Cannot combine tensors on different devices: "
                f"'{self.device}' vs '{other.device}'. Move one of them with "
                f".to('{self.device}') first."
            )

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        self._check_same_device(other)
        op = Add(self, other)
        result_data = op.forward(self.data, other.data)
        requires_grad = self.requires_grad or other.requires_grad
        
        return Tensor(result_data, _ctx = op, requires_grad=requires_grad, device=self.device)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        self._check_same_device(other)
        op = Mul(self, other)
        result_data = op.forward(self.data, other.data)
        requires_grad = self.requires_grad or other.requires_grad
        
        return Tensor(result_data, _ctx = op, requires_grad=requires_grad, device=self.device)
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        self._check_same_device(other)
        op = MatMul(self, other)
        result_data = op.forward(self.data, other.data)
        requires_grad = self.requires_grad or other.requires_grad
        
        return Tensor(result_data, _ctx = op, requires_grad=requires_grad, device=self.device)
    
    def __neg__(self):
        op = Neg(self)
        result_data = op.forward(self.data)
        return Tensor(result_data, _ctx=op, requires_grad=self.requires_grad, device=self.device)

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        self._check_same_device(other)
        op = Sub(self, other)
        result_data = op.forward(self.data, other.data)
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(result_data, _ctx=op, requires_grad=requires_grad, device=self.device)

    def __pow__(self, exponent):
        op = Pow(self)
        result_data = op.forward(self.data, exponent)
        return Tensor(result_data, _ctx=op, requires_grad=self.requires_grad, device=self.device)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        self._check_same_device(other)
        op = Div(self, other)
        result_data = op.forward(self.data, other.data)
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(result_data, _ctx=op, requires_grad=requires_grad, device=self.device)

    def relu(self):
        op = ReLU(self)
        result_data = op.forward(self.data)
        return Tensor(result_data, _ctx=op, requires_grad=self.requires_grad, device=self.device)

    def sigmoid(self):
        op = Sigmoid(self)
        result_data = op.forward(self.data)
        return Tensor(result_data, _ctx=op, requires_grad=self.requires_grad, device=self.device)

    def tanh(self):
        op = Tanh(self)
        result_data = op.forward(self.data)
        return Tensor(result_data, _ctx=op, requires_grad=self.requires_grad, device=self.device)

    def sum(self):
        op = Sum(self)
        result_data = op.forward(self.data)
        return Tensor(result_data, _ctx=op, requires_grad=self.requires_grad, device=self.device)

    def mean(self):
        op = Mean(self)
        result_data = op.forward(self.data)
        return Tensor(result_data, _ctx=op, requires_grad=self.requires_grad, device=self.device)

    def reshape(self, shape):
        op = Reshape(self)
        result_data = op.forward(self.data, shape)
        return Tensor(result_data, _ctx=op, requires_grad=self.requires_grad, device=self.device)
    
    @property
    def T(self):
        op = Transpose(self)
        result_data = op.forward(self.data)
        return Tensor(result_data, _ctx = op, requires_grad=self.requires_grad, device=self.device)
    
    def backward(self, grad = None):
        if not self.requires_grad:
            raise RuntimeError("element 0 of tensors does not require grad and does not have a grad_fn")
        if grad is None:
            self.grad = np.ones_like(self.data) if self.device == 'cpu' else cp.ones_like(self.data)
        else:
            self.grad = np.array(grad, dtype=np.float64) if self.device == 'cpu' else cp.array(grad, dtype=cp.float64)
            
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
