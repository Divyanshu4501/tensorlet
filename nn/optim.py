import numpy as np
from backend import cp


class SGD:
    def __init__(self, parameters, lr=0.1):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for p in self.parameters:
            if p.grad is not None:
                p.data -= self.lr * p.grad

    def zero_grad(self):
        # Was `p.grad = None`, which crashes the next loss.backward() call:
        # Operations.backward() does `parent.grad += grad_output`, and you
        # can't += onto None. Zero-fill the existing grad buffer instead,
        # matching each parameter's own device.
        for p in self.parameters:
            xp = cp if p.device == 'cuda' else np
            p.grad = xp.zeros_like(p.data)
