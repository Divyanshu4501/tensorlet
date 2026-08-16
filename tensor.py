import numpy as np
from operations import Add, Mul, MatMul, Transpose, Neg, Sub, Pow, Div, ReLU, Sigmoid, Tanh, Sum, Mean, Reshape

class Tensor:
    def __init__(self, data, _ctx = None, requires_grad = False):
        self.data = np.array(data, dtype=np.float64)
        self._ctx = _ctx
        self.requires_grad = requires_grad
        
        if self.requires_grad:
            self.grad = np.zeros_like(self.data, dtype=np.float64)
        else:
            self.grad = None
        
    def __repr__(self):
        return f"Tensor: {self.data}, requires_grad = {self.requires_grad}"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        op = Add(self, other)
        result_data = op.forward(self.data, other.data)
        requires_grad = self.requires_grad or other.requires_grad
        
        return Tensor(result_data, _ctx = op, requires_grad=requires_grad)

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
        print("\n========== COMPUTATION GRAPH ==========")
        for i, t in enumerate(topo_order):
            if t._ctx is None:
                # It's a leaf tensor (user-created input or weight)
                print(f"[{i}] LEAF TENSOR | Shape: {t.data.shape} | requires_grad: {t.requires_grad}")
            else:
                # It's a result of an operation
                op_name = t._ctx.__class__.__name__
                # Find the IDs of the parent tensors to show connections
                parent_indices = [topo_order.index(p) for p in t._ctx.parents]
                print(f"[{i}] OPERATION: {op_name} | Inputs: Node(s) {parent_indices} -> Output Shape: {t.data.shape}")
        print("=======================================\n")
        for t in reversed(topo_order):
            if t._ctx is not None:
                t._ctx.backward(t.grad)