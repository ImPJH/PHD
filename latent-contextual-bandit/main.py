from typing import Union
import numpy as np
from cfg import get_cfg
from models import *
import matplotlib.pyplot as plt

def logistic(x:np.ndarray, beta:np.ndarray) -> float:
    """
    x: k-dimensional input data
    beta: k-dimensional reward parameter
    """
    input_term = np.dot(x, beta)
    denominator = 1 + np.exp(-1 * input_term)
    return 1 / denominator

def vector_normalize(x:np.ndarray, bound:float) -> np.ndarray:
    return None

if __name__ == "__main__":
    theta = 
