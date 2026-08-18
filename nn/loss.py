from tensor import Tensor

class MSELoss:
    def __call__(self, predictions, targets):
        diff = predictions - targets
        squared_diff = diff ** 2
        total_loss = squared_diff.sum()
        N = Tensor([predictions.data.size], requires_grad=False)
        return total_loss/N