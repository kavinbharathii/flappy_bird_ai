
from math import e
import numpy as np

def sigmoid(x):
    return 1 / (1 + e**-x)

def tanh(x):
    return np.tanh(x)

