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