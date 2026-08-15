import numpy as np
class Operations:
    def __init__(self, *tensors):
        self.parents = tensors
        self.saved_tensors = ()
        
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors
        
    def forward(self, *args):
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError
    
class Add(Operations):
    def forward(self, a, b):
        return a + b
    
    def backward(self, grad_output):
        if self.parents[0].requires_grad:
            self.parents[0].grad += grad_output * 1.0
            
        if self.parents[1].requires_grad:
            self.parents[1].grad += grad_output * 1.0
            
class Mul(Operations):
    def forward(self, a, b):
        self.save_for_backward(a,b)
        return a*b
    
    def backward(self, grad_output):
        a, b = self.saved_tensors
        if self.parents[0].requires_grad:
            self.parents[0].grad += grad_output * b
        if self.parents[1].requires_grad:
            self.parents[1].grad += a * grad_output

class MatMul(Operations):
    def forward(self, a, b):
        self.save_for_backward(a,b)
        return a @ b
    
    def backward(self, grad_output):
        a, b = self.saved_tensors
        if self.parents[0].requires_grad:
            self.parents[0].grad += grad_output @ b.T
        if self.parents[1].requires_grad:
            self.parents[1].grad += a.T @ grad_output
            
class Transpose(Operations):
    def forward(self, a):
        return a.T
    
    def backward(self, grad_output):
        if self.parents[0].requires_grad:
            self.parents[0].grad += grad_output.T
            
class Neg(Operations):
    def forward(self, a):
        return -a
    def backward(self, grad_output):
        if self.parents[0].requires_grad:
            self.parents[0].grad += -grad_output

class Sub(Operations):
    def forward(self, a, b):
        return a - b
    
    def backward(self, grad_output):
        if self.parents[0].requires_grad:
            self.parents[0].grad += grad_output
        if self.parents[1].requires_grad:
            self.parents[1].grad += -grad_output
            
class Pow(Operations):
    def forward(self, a, exponent):
        self.save_for_backward(a, exponent)
        return a ** exponent
    
    def backward(self, grad_output):
        a, exponent = self.saved_tensors
        if self.parents[0].requires_grad:
            self.parents[0].grad += grad_output * (exponent * a **(exponent - 1))
            
class Div(Operations):
    def forward(self, a, b):
        self.saved_for_backward(a, b)
        return a / b

    def backward(self, grad_output):
        a, b = self.saved_tensors
        if self.parents[0].requires_grad:
            self.parents[0].grad += grad_output / b
        if self.parents[1].requires_grad:
            # Quotient rule application
            self.parents[1].grad += grad_output * (-a / (b ** 2))


class ReLU(Operations):
    def forward(self, a):
        self.saved_for_backward(a)
        return np.maximum(0, a)

    def backward(self, grad_output):
        a, = self.saved_tensors
        if self.parents[0].requires_grad:
            # Gradient is 1 if a > 0, else 0
            self.parents[0].grad += grad_output * (a > 0)


class Sigmoid(Operations):
    def forward(self, a):
        out = 1.0 / (1.0 + np.exp(-a))
        # Saving the output directly makes the backward pass much faster
        self.saved_for_backward(out)
        return out

    def backward(self, grad_output):
        out, = self.saved_tensors
        if self.parents[0].requires_grad:
            self.parents[0].grad += grad_output * (out * (1.0 - out))


class Tanh(Operations):
    def forward(self, a):
        out = np.tanh(a)
        self.saved_for_backward(out)
        return out

    def backward(self, grad_output):
        out, = self.saved_tensors
        if self.parents[0].requires_grad:
            self.parents[0].grad += grad_output * (1.0 - out ** 2)


class Sum(Operations):
    def forward(self, a):
        self.saved_for_backward(a)
        return np.sum(a)

    def backward(self, grad_output):
        a, = self.saved_tensors
        if self.parents[0].requires_grad:
            self.parents[0].grad += grad_output * np.ones_like(a)


class Mean(Operations):
    def forward(self, a):
        self.saved_for_backward(a)
        return np.mean(a)

    def backward(self, grad_output):
        a, = self.saved_tensors
        if self.parents[0].requires_grad:
            self.parents[0].grad += grad_output * np.ones_like(a) / a.size


class Reshape(Operations):
    def forward(self, a, shape):
        self.saved_for_backward(a.shape)
        return np.reshape(a, shape)

    def backward(self, grad_output):
        original_shape, = self.saved_tensors
        if self.parents[0].requires_grad:
            self.parents[0].grad += np.reshape(grad_output, original_shape)