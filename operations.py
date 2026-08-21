import numpy as np

try:
    import tensorlet_cuda
except ImportError:
    tensorlet_cuda = None

def is_gpu_array(x):
    """Helper to determine if the underlying data is a GPU array."""
    return not isinstance(x, (np.ndarray, np.generic, int, float, list, tuple))

def unbroadcast(grad, target_shape):
    ndims_added = len(grad.shape) - len(target_shape)
    
    if is_gpu_array(grad):
        if ndims_added > 0:
            grad = tensorlet_cuda.sum(grad, axis=tuple(range(ndims_added)))
        for i, dim_size in enumerate(target_shape):
            if dim_size == 1 and grad.shape[i] > 1:
                grad = tensorlet_cuda.sum(grad, axis=i, keepdims=True)
        return grad
    else:
        if ndims_added > 0:
            grad = np.sum(grad, axis=tuple(range(ndims_added)))
        for i, dim_size in enumerate(target_shape):
            if dim_size == 1 and grad.shape[i] > 1:
                grad = np.sum(grad, axis=i, keepdims=True) 
        return grad

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
        self.save_for_backward(a.shape, b.shape)
        if is_gpu_array(a) or is_gpu_array(b):
            return tensorlet_cuda.add(a, b)
        return a + b
    
    def backward(self, grad_output):
        shape_a, shape_b = self.saved_tensors
        if self.parents[0].requires_grad:
            self.parents[0].grad += unbroadcast(grad_output, shape_a)
        if self.parents[1].requires_grad:
            self.parents[1].grad += unbroadcast(grad_output, shape_b)

class Mul(Operations):
    def forward(self, a, b):
        self.save_for_backward(a.shape, b.shape)
        if is_gpu_array(a) or is_gpu_array(b):
            return tensorlet_cuda.mul(a, b)
        return a * b

    def backward(self, grad_output):
        shape_a, shape_b = self.saved_tensors        
        a = self.parents[0].data
        b = self.parents[1].data
        
        if self.parents[0].requires_grad:
            grad_a = tensorlet_cuda.mul(grad_output, b) if is_gpu_array(grad_output) else grad_output * b
            self.parents[0].grad += unbroadcast(grad_a, shape_a)
            
        if self.parents[1].requires_grad:
            grad_b = tensorlet_cuda.mul(grad_output, a) if is_gpu_array(grad_output) else grad_output * a
            self.parents[1].grad += unbroadcast(grad_b, shape_b)

class MatMul(Operations):
    def forward(self, a, b):
        self.save_for_backward(a, b)
        if is_gpu_array(a) or is_gpu_array(b):
            return tensorlet_cuda.matmul(a, b)
        return a @ b
    
    def backward(self, grad_output):
        a, b = self.saved_tensors
        if self.parents[0].requires_grad:
            if is_gpu_array(grad_output):
                self.parents[0].grad += tensorlet_cuda.matmul(grad_output, tensorlet_cuda.transpose(b))
            else:
                self.parents[0].grad += grad_output @ b.T
        if self.parents[1].requires_grad:
            if is_gpu_array(grad_output):
                self.parents[1].grad += tensorlet_cuda.matmul(tensorlet_cuda.transpose(a), grad_output)
            else:
                self.parents[1].grad += a.T @ grad_output
            
class Transpose(Operations):
    def forward(self, a):
        if is_gpu_array(a):
            return tensorlet_cuda.transpose(a)
        return a.T
    
    def backward(self, grad_output):
        if self.parents[0].requires_grad:
            if is_gpu_array(grad_output):
                self.parents[0].grad += tensorlet_cuda.transpose(grad_output)
            else:
                self.parents[0].grad += grad_output.T
            
class Neg(Operations):
    def forward(self, a):
        if is_gpu_array(a):
            return tensorlet_cuda.neg(a)
        return -a
        
    def backward(self, grad_output):
        if self.parents[0].requires_grad:
            if is_gpu_array(grad_output):
                self.parents[0].grad += tensorlet_cuda.neg(grad_output)
            else:
                self.parents[0].grad += -grad_output

class Sub(Operations):
    def forward(self, a, b):
        self.save_for_backward(a.shape, b.shape)
        if is_gpu_array(a) or is_gpu_array(b):
            return tensorlet_cuda.sub(a, b)
        return a - b
    
    def backward(self, grad_output):
        shape_a, shape_b = self.saved_tensors
        if self.parents[0].requires_grad:
            self.parents[0].grad += unbroadcast(grad_output, shape_a)
        if self.parents[1].requires_grad:
            neg_grad = tensorlet_cuda.neg(grad_output) if is_gpu_array(grad_output) else -grad_output
            self.parents[1].grad += unbroadcast(neg_grad, shape_b)
            
class Pow(Operations):
    def forward(self, a, exponent):
        self.save_for_backward(a, exponent)
        if is_gpu_array(a):
            return tensorlet_cuda.pow(a, exponent)
        return a ** exponent
    
    def backward(self, grad_output):
        a, exponent = self.saved_tensors
        if self.parents[0].requires_grad:
            if is_gpu_array(grad_output):
                base_deriv = tensorlet_cuda.mul(exponent, tensorlet_cuda.pow(a, exponent - 1))
                self.parents[0].grad += tensorlet_cuda.mul(grad_output, base_deriv)
            else:
                self.parents[0].grad += grad_output * (exponent * a ** (exponent - 1))
            
class Div(Operations):
    def forward(self, a, b):
        self.save_for_backward(a, b)
        if is_gpu_array(a) or is_gpu_array(b):
            return tensorlet_cuda.div(a, b)
        return a / b

    def backward(self, grad_output):
        a, b = self.saved_tensors
        if self.parents[0].requires_grad:
            if is_gpu_array(grad_output):
                self.parents[0].grad += tensorlet_cuda.div(grad_output, b)
            else:
                self.parents[0].grad += grad_output / b
                
        if self.parents[1].requires_grad:
            if is_gpu_array(grad_output):
                b_sq = tensorlet_cuda.pow(b, 2)
                neg_a = tensorlet_cuda.neg(a)
                quotient = tensorlet_cuda.div(neg_a, b_sq)
                self.parents[1].grad += tensorlet_cuda.mul(grad_output, quotient)
            else:
                self.parents[1].grad += grad_output * (-a / (b ** 2))

class ReLU(Operations):
    def forward(self, a):
        self.save_for_backward(a)
        if is_gpu_array(a):
            return tensorlet_cuda.relu(a)
        return np.maximum(0, a)

    def backward(self, grad_output):
        a, = self.saved_tensors
        if self.parents[0].requires_grad:
            if is_gpu_array(grad_output):
                mask = tensorlet_cuda.greater_than_zero(a)
                self.parents[0].grad += tensorlet_cuda.mul(grad_output, mask)
            else:
                self.parents[0].grad += grad_output * (a > 0)

class Sigmoid(Operations):
    def forward(self, a):
        if is_gpu_array(a):
            out = tensorlet_cuda.sigmoid(a)
        else:
            out = 1.0 / (1.0 + np.exp(-a))
        self.save_for_backward(out)
        return out

    def backward(self, grad_output):
        out, = self.saved_tensors
        if self.parents[0].requires_grad:
            if is_gpu_array(grad_output):
                one_minus_out = tensorlet_cuda.sub(1.0, out)
                deriv = tensorlet_cuda.mul(out, one_minus_out)
                self.parents[0].grad += tensorlet_cuda.mul(grad_output, deriv)
            else:
                self.parents[0].grad += grad_output * (out * (1.0 - out))

class Tanh(Operations):
    def forward(self, a):
        if is_gpu_array(a):
            out = tensorlet_cuda.tanh(a)
        else:
            out = np.tanh(a)
        self.save_for_backward(out)
        return out

    def backward(self, grad_output):
        out, = self.saved_tensors
        if self.parents[0].requires_grad:
            if is_gpu_array(grad_output):
                out_sq = tensorlet_cuda.pow(out, 2)
                deriv = tensorlet_cuda.sub(1.0, out_sq)
                self.parents[0].grad += tensorlet_cuda.mul(grad_output, deriv)
            else:
                self.parents[0].grad += grad_output * (1.0 - out ** 2)

class Sum(Operations):
    def forward(self, a):
        self.save_for_backward(a)
        if is_gpu_array(a):
            return tensorlet_cuda.sum(a)
        return np.sum(a)

    def backward(self, grad_output):
        a, = self.saved_tensors
        if self.parents[0].requires_grad:
            if is_gpu_array(a):
                self.parents[0].grad += tensorlet_cuda.mul(grad_output, tensorlet_cuda.ones_like(a))
            else:
                self.parents[0].grad += grad_output * np.ones_like(a)

class Mean(Operations):
    def forward(self, a):
        self.save_for_backward(a)
        if is_gpu_array(a):
            return tensorlet_cuda.mean(a)
        return np.mean(a)

    def backward(self, grad_output):
        a, = self.saved_tensors
        if self.parents[0].requires_grad:
            if is_gpu_array(a):
                # Assuming your binding exposes a `.size` or you handle scaling in C++
                ones = tensorlet_cuda.ones_like(a)
                scaled_grad = tensorlet_cuda.div(grad_output, a.size)
                self.parents[0].grad += tensorlet_cuda.mul(scaled_grad, ones)
            else:
                self.parents[0].grad += grad_output * np.ones_like(a) / a.size

class Reshape(Operations):
    def forward(self, a, shape):
        self.save_for_backward(a.shape)
        if is_gpu_array(a):
            return tensorlet_cuda.reshape(a, shape)
        return np.reshape(a, shape)

    def backward(self, grad_output):
        original_shape, = self.saved_tensors
        if self.parents[0].requires_grad:
            if is_gpu_array(grad_output):
                self.parents[0].grad += tensorlet_cuda.reshape(grad_output, original_shape)
            else:
                self.parents[0].grad += np.reshape(grad_output, original_shape)