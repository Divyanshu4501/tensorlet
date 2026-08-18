import numpy as np
import math
from tensor import Tensor
from nn.Module import Module

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        random_weights = np.random.randn(in_features, out_features) * 0.01
        self.weight = Tensor(random_weights, requires_grad=True)
        self.bias = Tensor(np.zeros((1, out_features)), requires_grad=True)
        
    def forward(self, X):
        return (X @ self.weight) + self.bias
    