class Function:
    def __init__(self, *tensors):
        self.parents = tensors
        self.saved_tensors = ()
        
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors
    
    def forward(self, *args):
        raise NotImplementedError
    def backward(self, grad_output):
        raise NotImplementedError