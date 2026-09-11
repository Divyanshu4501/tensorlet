from torch import Tensor

a = Tensor([1,2,3,4], device='cpu', requires_grad=False)
b = Tensor([-1,-2,-3,-4], device='cpu', requires_grad=False)

print(a+b)

