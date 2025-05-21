import numpy as np
from abc import ABC, abstractmethod

class Loss(ABC):
    def __init__(self):
        return
    @abstractmethod
    def evaluate(self, error:np.ndarray, inv_cov: np.ndarray):
        pass
       
class GaussianLoss(Loss):
    def __init__(self):
        super().__init__()
    
    def evaluate(self, error:np.ndarray, inv_cov: np.ndarray):
        return 0.5 * error.T @ inv_cov @ error
    
class StudentTLoss(Loss):
    def __init__(self, dof: float):
        super().__init__()
        self._dof = dof
    
    def evaluate(self, error, inv_cov):
        D = error.shape[0]
        return 0.5 * (self._dof + D) * np.log(1.0 + (error.T @ inv_cov @ error / self._dof))
    
class CauchyLoss(StudentTLoss):
    def __init__(self, dof:float = 1.0 ):
        super().__init__(dof)
    
    def evaluate(self, error, inv_cov):
        return super().evaluate(error, inv_cov)

class SkewLaplaceLoss(Loss):
    def __init__(self, lamb:float):
        super().__init__()
        self._lambda = lamb
    
    def evaluate(self, error, inv_cov):
        if error.shape[0] > 1:
            raise NotImplementedError("Haven't implemented multivariate skew laplace loss yet.")
        alpha = np.sqrt(1 + self._lambda**2 * inv_cov)
        return (-1 * self._lambda * error @ inv_cov ) + (alpha * np.sqrt(inv_cov) @ np.abs(error))

