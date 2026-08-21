import numpy as np
from operations import Add, Mul, MatMul, Transpose, Neg, Sub, Pow, Div, ReLU, Sigmoid, Tanh, Sum, Mean, Reshape

try:
    import tensorlet_cuda
    _CUDA_AVAILABLE = True
except ImportError:
    _CUDA_AVAILABLE = False
    
def is_available():
    return _CUDA_AVAILABLE

class Tensor:
    def __init__(self, data, _ctx=None, requires_grad=False, device=None):
        self._ctx = _ctx
        self.requires_grad = requires_grad
        
        if device is None:
            self.device = "cuda" if is_available() else "cpu"
        else:
            self.device = device
            
        if self.device == "cpu":
            self.data = np.array(data, dtype=np.float32)
        elif self.device == "cuda":
            self.data = tensorlet_cuda.to_device(np.array(data, dtype=np.float32))
        
        if self.requires_grad:
            if self.device == "cpu":
                self.grad = np.zeros_like(self.data, dtype=np.float32)
            elif self.device == "cuda":
                # Ensure you expose a zeros_like function in your C++ bindings
                self.grad = tensorlet_cuda.zeros_like(self.data)
        else:
            self.grad = None

    def cpu(self):
        """Moves the tensor to CPU"""
        if self.device == "cpu":
            return self
        cpu_data = tensorlet_cuda.to_cpu(self.data)
        return Tensor(data=cpu_data, _ctx=self._ctx, requires_grad=self.requires_grad, device="cpu")

    def cuda(self):
        """Moves the tensor to GPU"""
        if self.device == "cuda":
            return self
        return Tensor(data=self.data, _ctx=self._ctx, requires_grad=self.requires_grad, device="cuda")
            
    def to(self, device):
        if device == self.device:
            return self
        if device == "cpu":
            return self.cpu()
        elif device == "cuda":
            return self.cuda()
        else:
            raise ValueError(f"Unknown device: {device}")
        
    def __repr__(self):
        return f"Tensor: {self.data}, requires_grad={self.requires_grad}, device='{self.device}'"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        op = Add(self, other)
        result_data = op.forward(self.data, other.data)
        requires_grad = self.requires_grad or other.requires_grad
        # Propagate device to output
        return Tensor(result_data, _ctx=op, requires_grad=requires_grad, device=self.device)
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        op = Mul(self, other)
        result_data = op.forward(self.data, other.data)
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(result_data, _ctx=op, requires_grad=requires_grad, device=self.device)
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        op = MatMul(self, other)
        result_data = op.forward(self.data, other.data)
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(result_data, _ctx=op, requires_grad=requires_grad, device=self.device)
    
    def __neg__(self):
        op = Neg(self)
        result_data = op.forward(self.data)
        return Tensor(result_data, _ctx=op, requires_grad=self.requires_grad, device=self.device)

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
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
        return Tensor(result_data, _ctx=op, requires_grad=self.requires_grad, device=self.device)
    
    def backward(self, grad=None):
        if not self.requires_grad:
            raise RuntimeError("element 0 of tensors does not require grad and does not have a grad_fn")
        
        if grad is None:
            if self.device == "cpu":
                self.grad = np.ones_like(self.data, dtype=np.float32)
            elif self.device == "cuda":
                self.grad = tensorlet_cuda.ones_like(self.data)
        else:
            if self.device == "cpu":
                self.grad = np.array(grad, dtype=np.float32)
            elif self.device == "cuda":
                self.grad = tensorlet_cuda.to_device(np.array(grad, dtype=np.float32))
            
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