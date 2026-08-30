from tensor import Tensor

class MSELoss:
    def __call__(self, predictions, targets):
        
        if not isinstance(targets, Tensor):
            targets = Tensor(targets, device=predictions.device)
        elif targets.device != predictions.device:
            targets = targets.to(predictions.device)
        
        diff = predictions - targets
        squared_diff = diff ** 2
        
        return squared_diff.mean()