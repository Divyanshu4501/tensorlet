import numpy as np
import cupy as cp
import triton
from triton_kernels import (
    add_kernel, sub_kernel, mul_kernel, div_kernel, 
    neg_kernel, relu_kernel, sigmoid_kernel, tanh_kernel, 
    pow_kernel, matmul_kernel
)

def unbroadcast(grad, target_shape):
    xp = cp if isinstance(grad, cp.ndarray) else np
    ndims_added = grad.ndim - len(target_shape)
    if ndims_added > 0:
        grad = xp.sum(grad, axis=tuple(range(ndims_added)))
        
    for i, dim_size in enumerate(target_shape):
        if dim_size == 1 and grad.shape[i] > 1:
            grad = xp.sum(grad, axis=i, keepdims=True)
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
        if isinstance(a, cp.ndarray):
            out = cp.empty_like(a)
            n_elements = out.size
            
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            add_kernel[grid](a, b, out, n_elements, BLOCK_SIZE=1024)
            return out
        else:
            return a + b
    
    def backward(self, grad_output):
        shape_a, shape_b = self.saved_tensors
        if self.parents[0].requires_grad:
            self.parents[0].grad += unbroadcast(grad_output, shape_a)
        if self.parents[1].requires_grad:
            self.parents[1].grad += unbroadcast(grad_output, shape_b)

class Sub(Operations):
    def forward(self, a, b):
        self.save_for_backward(a.shape, b.shape)
        if isinstance(a, cp.ndarray):
            out = cp.empty_like(a)
            n_elements = out.size
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            sub_kernel[grid](a, b, out, n_elements, BLOCK_SIZE=1024)
            return out
        return a - b
    
    def backward(self, grad_output):
        shape_a, shape_b = self.saved_tensors
        if self.parents[0].requires_grad:
            self.parents[0].grad += unbroadcast(grad_output, shape_a)
        if self.parents[1].requires_grad:
            self.parents[1].grad += unbroadcast(-grad_output, shape_b)


class Mul(Operations):
    def forward(self, a, b):
        self.save_for_backward(a.shape, b.shape)
        if isinstance(a, cp.ndarray):
            out = cp.empty_like(a)
            n_elements = out.size
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            mul_kernel[grid](a, b, out, n_elements, BLOCK_SIZE=1024)
            return out
        return a * b

    def backward(self, grad_output):
        shape_a, shape_b = self.saved_tensors        
        a = self.parents[0].data
        b = self.parents[1].data
        if self.parents[0].requires_grad:
            self.parents[0].grad += unbroadcast(grad_output * b, shape_a)
        if self.parents[1].requires_grad:
            self.parents[1].grad += unbroadcast(grad_output * a, shape_b)


class Div(Operations):
    def forward(self, a, b):
        self.save_for_backward(a, b)
        if isinstance(a, cp.ndarray):
            out = cp.empty_like(a)
            n_elements = out.size
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            div_kernel[grid](a, b, out, n_elements, BLOCK_SIZE=1024)
            return out
        return a / b

    def backward(self, grad_output):
        a, b = self.saved_tensors
        if self.parents[0].requires_grad:
            self.parents[0].grad += unbroadcast(grad_output / b, a.shape)
        if self.parents[1].requires_grad:
            self.parents[1].grad += unbroadcast(grad_output * (-a / (b ** 2)), b.shape)


class Pow(Operations):
    def forward(self, a, exponent):
        self.save_for_backward(a, exponent)
        if isinstance(a, cp.ndarray):
            out = cp.empty_like(a)
            n_elements = out.size
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            pow_kernel[grid](a, exponent, out, n_elements, BLOCK_SIZE=1024)
            return out
        return a ** exponent
    
    def backward(self, grad_output):
        a, exponent = self.saved_tensors
        if self.parents[0].requires_grad:
            self.parents[0].grad += grad_output * (exponent * a ** (exponent - 1))


class MatMul(Operations):
    def forward(self, a, b):
        self.save_for_backward(a, b)
        if isinstance(a, cp.ndarray):
            # Triton MatMul is optimized for float32/fp16
            a = a.astype(cp.float32)
            b = b.astype(cp.float32)
            
            M, K = a.shape
            K_, N = b.shape
            assert K == K_, "Incompatible dimensions for MatMul"
            
            c = cp.empty((M, N), dtype=cp.float32)
            
            # Grid calculation for 2D tiling
            grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
            
            matmul_kernel[grid](
                a, b, c,
                M, N, K,
                a.strides[0] // a.itemsize, a.strides[1] // a.itemsize,
                b.strides[0] // b.itemsize, b.strides[1] // b.itemsize,
                c.strides[0] // c.itemsize, c.strides[1] // c.itemsize,
                BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32, GROUP_SIZE_M=8
            )
            return c
        return a @ b
    
    def backward(self, grad_output):
        a, b = self.saved_tensors
        if self.parents[0].requires_grad:
            self.parents[0].grad += grad_output @ b.T
        if self.parents[1].requires_grad:
            self.parents[1].grad += a.T @ grad_output


# ==========================================
# UNARY OPERATIONS
# ==========================================

class Neg(Operations):
    def forward(self, a):
        if isinstance(a, cp.ndarray):
            out = cp.empty_like(a)
            n_elements = out.size
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            neg_kernel[grid](a, out, n_elements, BLOCK_SIZE=1024)
            return out
        return -a

    def backward(self, grad_output):
        if self.parents[0].requires_grad:
            self.parents[0].grad += -grad_output


class ReLU(Operations):
    def forward(self, a):
        self.save_for_backward(a)
        if isinstance(a, cp.ndarray):
            out = cp.empty_like(a)
            n_elements = out.size
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            relu_kernel[grid](a, out, n_elements, BLOCK_SIZE=1024)
            return out
        return np.maximum(0, a)

    def backward(self, grad_output):
        a, = self.saved_tensors
        if self.parents[0].requires_grad:
            self.parents[0].grad += grad_output * (a > 0)


class Sigmoid(Operations):
    def forward(self, a):
        if isinstance(a, cp.ndarray):
            out = cp.empty_like(a)
            n_elements = out.size
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            sigmoid_kernel[grid](a, out, n_elements, BLOCK_SIZE=1024)
        else:
            out = 1.0 / (1.0 + np.exp(-a))
            
        self.save_for_backward(out)
        return out

    def backward(self, grad_output):
        out, = self.saved_tensors
        if self.parents[0].requires_grad:
            self.parents[0].grad += grad_output * (out * (1.0 - out))


class Tanh(Operations):
    def forward(self, a):
        if isinstance(a, cp.ndarray):
            out = cp.empty_like(a)
            n_elements = out.size
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            tanh_kernel[grid](a, out, n_elements, BLOCK_SIZE=1024)
        else:
            out = np.tanh(a)
            
        self.save_for_backward(out)
        return out

    def backward(self, grad_output):
        out, = self.saved_tensors
        if self.parents[0].requires_grad:
            self.parents[0].grad += grad_output * (1.0 - out ** 2)

class Transpose(Operations):
    def forward(self, a):
        return a.T
    
    def backward(self, grad_output):
        if self.parents[0].requires_grad:
            self.parents[0].grad += grad_output.T


class Sum(Operations):
    def forward(self, a):
        self.save_for_backward(a)
        xp = cp if isinstance(a, cp.ndarray) else np
        return xp.sum(a)

    def backward(self, grad_output):
        a, = self.saved_tensors
        xp = cp if isinstance(a, cp.ndarray) else np
        if self.parents[0].requires_grad:
            self.parents[0].grad += grad_output * xp.ones_like(a)


class Mean(Operations):
    def forward(self, a):
        self.save_for_backward(a)
        xp = cp if isinstance(a, cp.ndarray) else np
        return xp.mean(a)

    def backward(self, grad_output):
        a, = self.saved_tensors
        xp = cp if isinstance(a, cp.ndarray) else np
        if self.parents[0].requires_grad:
            self.parents[0].grad += grad_output * xp.ones_like(a) / a.size


class Reshape(Operations):
    def forward(self, a, shape):
        self.save_for_backward(a.shape)
        xp = cp if isinstance(a, cp.ndarray) else np
        return xp.reshape(a, shape)

    def backward(self, grad_output):
        original_shape, = self.saved_tensors
        xp = cp if isinstance(grad_output, cp.ndarray) else np
        if self.parents[0].requires_grad:
            self.parents[0].grad += xp.reshape(grad_output, original_shape)