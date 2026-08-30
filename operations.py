import numpy as np
from backend import cp, triton, HAS_CUDA_BACKEND
from triton_kernels import (
    add_kernel, sub_kernel, mul_kernel, div_kernel, 
    neg_kernel, relu_kernel, sigmoid_kernel, tanh_kernel, 
    pow_kernel, matmul_kernel
)

if HAS_CUDA_BACKEND and not hasattr(cp.ndarray, "data_ptr"):
    cp.ndarray.data_ptr = lambda self: self.data.ptr

def unbroadcast(grad, target_shape):
    xp = cp if isinstance(grad, cp.ndarray) else np
    ndims_added = grad.ndim - len(target_shape)
    if ndims_added > 0:
        grad = xp.sum(grad, axis=tuple(range(ndims_added)))
        
    for i, dim_size in enumerate(target_shape):
        if dim_size == 1 and grad.shape[i] > 1:
            grad = xp.sum(grad, axis=i, keepdims=True)
    return grad


def _broadcast_for_kernel(a, b):
    """The elementwise Triton kernels below (add/sub/mul/div) operate on
    flat, equal-sized buffers — they don't know how to broadcast shapes the
    way `a + b` does in NumPy. Without this, something as ordinary as
    `(X @ W) + bias` (batch, out) + (1, out) would read past the end of
    `bias`'s buffer on the GPU (wrong results, or an illegal memory access).
    So: broadcast both operands to their common shape *before* handing them
    to the kernel, exactly like NumPy would, and materialize them as
    contiguous buffers since the kernel indexes memory directly."""
    out_shape = np.broadcast_shapes(a.shape, b.shape)
    a_b = cp.ascontiguousarray(cp.broadcast_to(a, out_shape))
    b_b = cp.ascontiguousarray(cp.broadcast_to(b, out_shape))
    return a_b, b_b, out_shape


def _launch_elementwise(kernel, a, b):
    """Shared launch path for the binary elementwise kernels (add/sub/mul/div)."""
    a_b, b_b, out_shape = _broadcast_for_kernel(a, b)
    out = cp.empty(out_shape, dtype=a_b.dtype)
    n_elements = out.size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    kernel[grid](a_b, b_b, out, n_elements, BLOCK_SIZE=1024)
    return out


def _cuda_pow(a, exponent):
    """Elementwise a ** exponent on the GPU.

    The kernel used to call `tl.math.pow`, but that function simply doesn't
    exist on some Triton builds (confirmed on the cluster this ran on:
    `AttributeError: module 'triton.language.math' has no attribute 'pow'`).

    Even if it did exist, routing everything through a generic pow()
    wouldn't be correct here: the only place this project calls `** n` is
    `diff ** 2` in MSELoss, where `diff` is routinely negative. A generic
    GPU pow is usually implemented as `exp(exponent * log(x))`, and
    `log(negative)` is NaN — that would have silently poisoned every loss
    and gradient with NaNs on any batch where predictions were too high.

    So: for the common case (a non-negative integer exponent, which is all
    this project ever actually uses), compute it exactly via repeated
    elementwise multiplication — correct for negative bases, and doesn't
    depend on any pow-like primitive existing in this Triton version at
    all. Only fall back to the log/exp identity for a genuinely fractional
    or negative exponent, and only for non-negative bases (documented
    limitation, same as most GPU pow() implementations).
    """
    if float(exponent).is_integer() and exponent >= 0:
        n = int(exponent)
        if n == 0:
            return cp.ones_like(a)
        out = a
        for _ in range(n - 1):
            out = _launch_elementwise(mul_kernel, out, a)
        return cp.ascontiguousarray(out)

    out = cp.empty_like(a)
    n_elements = out.size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    pow_kernel[grid](a, float(exponent), out, n_elements, BLOCK_SIZE=1024)
    return out

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
            return _launch_elementwise(add_kernel, a, b)
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
            return _launch_elementwise(sub_kernel, a, b)
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
            return _launch_elementwise(mul_kernel, a, b)
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
            return _launch_elementwise(div_kernel, a, b)
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
            return _cuda_pow(a, exponent)
        return a ** exponent
    
    def backward(self, grad_output):
        a, exponent = self.saved_tensors
        if self.parents[0].requires_grad:
            self.parents[0].grad += grad_output * (exponent * a ** (exponent - 1))


class MatMul(Operations):
    def forward(self, a, b):
        self.save_for_backward(a, b)
        if isinstance(a, cp.ndarray):
            a = a.astype(cp.float32)
            b = b.astype(cp.float32)
            
            M, K = a.shape
            K_, N = b.shape
            assert K == K_, "Incompatible dimensions for MatMul"
            
            c = cp.empty((M, N), dtype=cp.float32)
            
            grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
            
            matmul_kernel[grid](
                a, b, c,
                M, N, K,
                a.strides[0] // a.itemsize, a.strides[1] // a.itemsize,
                b.strides[0] // b.itemsize, b.strides[1] // b.itemsize,
                c.strides[0] // c.itemsize, c.strides[1] // c.itemsize,
                BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32,
                GROUP_SIZE_M=8,
            )
            return c
        return a @ b
    
    def backward(self, grad_output):
        a, b = self.saved_tensors
        if self.parents[0].requires_grad:
            self.parents[0].grad += grad_output @ b.T
        if self.parents[1].requires_grad:
            self.parents[1].grad += a.T @ grad_output

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