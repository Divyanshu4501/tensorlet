try:
    import cupy as cp
    HAS_CUPY = True
except:
    cp = None
    HAS_CUPY = False