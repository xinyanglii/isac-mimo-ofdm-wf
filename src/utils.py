import numpy as np
import torch
import scipy.constants as spc


def freq2wavelen(freq: float | np.ndarray) -> float | np.ndarray:
    return spc.c / freq

def herm(x: np.ndarray| torch.Tensor) -> np.ndarray | torch.Tensor:
    return x.conj().T

def db2lin(x):
    return 10 ** (x / 10)

def lin2db(x):
    return 10 * np.log10(x)