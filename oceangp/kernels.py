"""Covariance (Kernel) functions for Gaussian Processes."""

import numpy as np

def square_exponential(x1, x2, length_scale=1.0, variance=1.0):
    """Squared Exponential (RBF )Kernel between two 1-D input arrays.
    
    Args:
        x1 : array_like
            First set of inputs, shape (n,).
        x2 : array_like
            Second set of inputs, shape (m,).
        length_scale : float, optional
            Length-scale (ell). Larger values give smoother functions by keeping
            distant inputs correlated. Default 1.0.
        variance : float, optional
            Prior variance of the function at any point. Default 1.0.
    
    Returns:
        K : ndarray
            Covariance matrix of shape (n, m) where K[i, j] = k(x1[i], x2[j]).
    """

    x1 = np.asarray(x1, dtype=np.float64).reshape(-1,1)
    x2 = np.asarray(x2, dtype=np.float64).reshape(1,-1)
    sqdist = (x1 - x2)**2
    K = variance*np.exp(-sqdist/(2*length_scale**2))

    return K

