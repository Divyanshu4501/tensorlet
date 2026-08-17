import numpy as np
import math
from tensor import Tensor
from nn.Module import Module

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        bound = 1/(math.sqrt(in_features))
        
        weight_data = np.random.uniform(-bound, bound, (in_features, out_features))
        self.weight = Tensor(weight_data, requires_grad=True)

        bias_data = np.random.uniform(-bound, bound, (1, out_features))
        self.bias = Tensor(bias_data, requires_grad=True)
        
    def forward(self, X):
        return (X @ self.weight) + self.bias
    