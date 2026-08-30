"""
Central place that resolves the optional GPU backend (cupy + triton).

Why this exists
----------------
tensor.py and operations.py used to do a bare `import cupy as cp` at the top
of the file. That means the whole library — including the plain CPU/NumPy
path — crashed with an ImportError on any machine that doesn't have cupy
installed (e.g. a laptop with no NVIDIA GPU). That made it impossible to
even run `--device cpu` for a quick local sanity check before submitting a
job to the HPC cluster.

This module makes the GPU backend optional:
  - If cupy/triton are installed, they're used as-is.
  - If they aren't, `cp` becomes a stub whose only job is to fail loudly
    and clearly the moment someone actually tries to use `device='cuda'`,
    while leaving the CPU/NumPy path completely unaffected.
"""
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False

HAS_CUDA_BACKEND = HAS_CUPY and HAS_TRITON


class _MissingCudaArrayType:
    """Stand-in for cp.ndarray when cupy isn't installed, so that
    `isinstance(x, cp.ndarray)` style checks elsewhere in the codebase keep
    working (they simply always evaluate to False on this machine)."""
    pass


class _MissingCudaBackend:
    """Stand-in for the `cp` module when cupy/triton aren't installed.
    Any real use (other than the isinstance check via .ndarray) raises a
    clear, actionable error instead of a cryptic ImportError/AttributeError
    deep inside operations.py."""

    ndarray = _MissingCudaArrayType

    def __getattr__(self, name):
        raise RuntimeError(
            "tensorlet was asked to use device='cuda', but cupy and/or "
            "triton are not installed in this Python environment. Install "
            "them (matching the CUDA version on this machine, e.g. via "
            "`pip install cupy-cudaXXx triton`) or run with --device cpu."
        )


if not HAS_CUDA_BACKEND:
    cp = _MissingCudaBackend()
