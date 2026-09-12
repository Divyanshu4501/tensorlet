import numpy as np
from torch.backend import cp, HAS_CUPY
class TensorImpl:
    def __init__(self, data, device):
        self.device = device
        if device == 'cpu':
            self.data = cp.asnumpy(data).astype(np.float32) if HAS_CUPY and isinstance(data, cp.ndarray) else np.array(data, dtype=np.float32)
        elif device == 'cuda':
            self.data = data.astype(cp.float32) if HAS_CUPY and isinstance(data, cp.ndarray) else cp.array(data, dtype=cp.float32)
        else:
            raise ValueError(
                f"No device named {device}"
            )