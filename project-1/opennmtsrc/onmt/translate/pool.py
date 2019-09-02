import numpy as np

def maxpool(a):
    a_minus = ((a < 0) * a).min(axis=0)
    a_plus = ((a > 0) * a).max(axis=0)
    return np.where(np.abs(a_minus) > np.abs(a_plus), a_minus, a_plus)

def meanpool(a):
    return a.sum(axis=0)/a.shape[0]

def lastpool(a):
    return a[-1,:]
