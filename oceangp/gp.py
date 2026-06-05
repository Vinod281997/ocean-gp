""" Gaussian Process Regression with Squared Exponential Kernel """

import numpy as np
from .kernels import square_exponential
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

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
        K = square_exponential(self.X_train, self.X_train, 
                               self.length_scale, self.variance)
        K = K + self.variance_noise*np.eye(len(self.X_train))
        self._chol = cho_factor(K)
        self._alpha = cho_solve(self._chol, self.y_train)
        self._fitted = True
        return self
    
    def predict(self, X_test):
        """Posterior prediction at test points.
        
        Args:
            X_test : array_like
                time points of test data, shape (m,)

        Returns:
            mean : ndarray
                Posterior mean at test points, shape (m,)
            cov : ndarray
                Posterior covariance at test points, shape (m,)        
        """

        if not self._fiited:
            raise ValueError("Call fit() before predict().")
        
        X_test = np.asarray(X_test, dtype=np.float64)
        K_s = square_exponential(self.X_train, X_test, 
                                 self.length_scale, self.variance)
        mean = K_s.T @ self._alpha
        K_ss = square_exponential(X_test, X_test, self.length_scale, self.variance)
        v = cho_solve(self._chol, K_s)
        cov = K_ss - K_s.T @ v
        return mean, cov
    
    def log_marginal_likelihood(self):
        """Log marginal likelihood of the training data under the fitted GP.
        
        Returns:
            float
                The log marginal likelihood log p(y_train | X_train, theta) 
                where theta are the hyperparameters.
                
        """

        if not self._fitted:
            raise RuntimeError("Call fit() before marginal_likelihood().")
        
        n = len(self.y_train)
        data_fit = -0.5 * (self.y_train @ self._alpha)
        log_det = 2.0 * np.sum(np.diag(self._chol[0]))
        return data_fit - 0.5 * log_det - 0.5 * n * np.log(2*np.pi)
    
    def fit_hyperparameters(self, X_train, y_train, n_runs=8, rng=None):
        """Find hyperparameters that maximize the log marginal likelihood.
        
        Args:
            X_train : array_like
                time points of training data, shape (n,)
            y_train : array_like
                observed flow values, shape (n,)
            n_runs : int, optional
                Number of random initializations for hyperparameter 
                optimization. Multiple runs can help avoid local 
                optima. Default 8.
            rng : int or arra_like[int], optional
                Random seed. default None.
                 
        Returns:
            self: GaussianProcess
        """

        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64)
        n = len(X)
        rng = np.random.default_rng(rng)

        def neg_lml(log_theta):
            ell, variance, variance_noise = np.exp(log_theta)
            Ky = square_exponential(X, X, ell, variance) + variance_noise * np.eye(n) 
            try:
                chol = cho_factor(Ky)
            except:
                return 1e25

            alpha = cho_solve(chol, y)
            log_det = 2.0 * np.sum(np.diag(chol[0]))
            return 0.5 * (y @ alpha) + 0.5 * log_det + 0.5 * n * np.log(2*np.pi)
        
        best = None
        for _ in range(n_runs):
            x0 = rng.uniform(-3, 3, size=3)
            res = minimize(neg_lml, x0, method="L-BFGS_B")
            if best is None or res.fun < best.fun:
                best = res
        
        self.length_scale, self.variance, self.variance_noise = np.exp(best.x)
        self.fit(X_train, y_train)
        return self



        

        


