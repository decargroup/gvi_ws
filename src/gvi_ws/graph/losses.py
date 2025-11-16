import numpy as np
from navlie.batch.losses import LossFunction
from abc import ABC, abstractmethod


class Loss(ABC):
    def __init__(self):
        return

    @abstractmethod
    def evaluate(self, error: np.ndarray, inv_cov: np.ndarray):
        pass


class GaussianLoss(Loss):
    def __init__(self):
        super().__init__()

    def evaluate(self, error: np.ndarray, inv_cov: np.ndarray):
        return 0.5 * error.T @ inv_cov @ error


class StudentTLoss(Loss):
    def __init__(self, dof: float):
        super().__init__()
        self._dof = dof

    def evaluate(self, error, inv_cov):
        D = error.shape[0]
        return (
            0.5
            * (self._dof + D)
            * np.log(1.0 + (error.T @ inv_cov @ error / self._dof))
        )


class CauchyLoss(StudentTLoss):
    def __init__(self, dof: float = 1.0):
        super().__init__(dof)

    def evaluate(self, error, inv_cov):
        return super().evaluate(error, inv_cov)


class SkewLaplaceLoss(Loss):
    def __init__(self, lamb: float):
        super().__init__()
        self._lambda = lamb

    def evaluate(self, error, inv_cov):
        if error.shape[0] > 1:
            raise NotImplementedError(
                "Haven't implemented multivariate skew laplace loss yet."
            )
        alpha = np.sqrt(1 + self._lambda**2 * inv_cov)
        return (-1 * self._lambda * error @ inv_cov) + (
            alpha * np.sqrt(inv_cov) @ np.abs(error)
        )


# Custom Loss for navlie MAP method
class AsymmetricCauchyLoss(LossFunction):
    """
    Asymmetric (two-piece) Cauchy robust loss with normalization.
    The PDF integrates to 1:
        f(e) = [2 / (π (c_pos + c_neg))] * 1 / [1 + (e / c)^2]
    where c = c_pos if e >= 0 else c_neg.
    """

    def __init__(self, c_pos: float = 1.0, c_neg: float = 1.0):
        self.c_pos = float(c_pos)
        self.c_neg = float(c_neg)

    def loss(self, e: float) -> float:
        """
        Negative log-likelihood style robust loss:
            ρ(e) = -log( f(e) )
        """
        c = self.c_pos if e >= 0 else self.c_neg
        A = 2.0 / (np.pi * (self.c_pos + self.c_neg))
        return -np.log(A) + np.log1p((e / c) ** 2)

    def weight(self, e: float) -> float:
        """
        Robust M-estimation weight:
            w(e) = ψ(e) / e = (1 / e) * ∂ρ/∂e
        For normalized asymmetric Cauchy:
            ψ(e) = (2e / c^2) / (1 + (e/c)^2)
            w(e) = 2 / (c^2 + e^2)
        But we drop constants to preserve relative weighting.
        """
        c = self.c_pos if e >= 0 else self.c_neg
        return 1.0 / (1.0 + (e / c) ** 2)
