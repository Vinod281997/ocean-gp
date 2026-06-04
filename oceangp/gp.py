""" Gaussian Process Regression with Squared Exponential Kernel """

import numpy as np
from .kernels import square_exponential
from scipy.linalg import cho_factor, cho_solve

class GaussianProcess:
    """ Zero mean Gaussian Process Regression with squared exponential kernels.
    
    Attributes:
        length_scale : float, optional
            Kernel length-scale. Default 1.0
        variance : float, optional
            Kernel variance. Default 1.0
        variance_noise : float, optional
            Observation noise variance; also stablizes the Cholesky. Default 1e-4    
    
    """

    def __init__(self, length_scale=1.0, variance=1.0, variance_noise=1e-4):
        self.length_scale = length_scale
        self.variance = variance
        self.variance_noise = variance_noise
        self._fitted = False

    def fit(self, X_train, y_train):
        """Condition the GP on training data.
        
        Args:
            X_train : array_like
                time points of training data, shape (n,)
            y_train : array_like
                observed flow values, shape (n,)
                
        Returns:
            self: GaussianProcess
        """

        self.X_train = np.asarray(X_train, dtype=np.float64)
        self.y_train = np.asarray(y_train, dtype=np.float64)
        K = square_exponential(self.X_train, self.X_train, self.length_scale, self.variance)

