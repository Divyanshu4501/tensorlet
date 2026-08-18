from tensor import Tensor

class MSELoss:
    def __call__(self, predictions, targets):
        diff = predictions - targets
        squared_diff = diff ** 2
        total_loss = squared_diff.sum()
        N = predictions.data.size
        return total_loss * (1.0/N)