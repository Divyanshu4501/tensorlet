from torch.autograd.function import Function

class Add(Function):
    def forward(self, a, b):
        self.save_for_backward(a.shape, b.shape)
        return a + b
    def backward(self, grad_output):
        pass