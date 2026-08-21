import numpy as np

try:
    import pyelsa as elsa
except ImportError:
    elsa = None


def shepp_logan(size):
    if elsa is None:
        raise ImportError("shepp_logan requires the optional pyelsa package")
    return np.rot90(elsa.phantoms.modifiedSheppLogan(size), -1)
