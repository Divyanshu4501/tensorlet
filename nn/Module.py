from tensor import Tensor

class Module:
    def __init__(self):
        self.device = "cpu"

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None
            
    def parameters(self):
        params = []
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor) and value.requires_grad:
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
        
        return params
    
    def to(self, device):
        self.device = device
        for name, value in self.__dict__.items():
            if isInstance(value, Tensor) and value.requires_grad:
                setattr(self, name, value.to(device))
            elif isInstance(value, Module):
                value.to(device)
        return self
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)