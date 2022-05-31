from abc import abstractmethod

import numpy as np


class AnnealingStrategy:

    @abstractmethod
    def update(self, T, k):
        pass


class LinearStrategy(AnnealingStrategy):

    def __init__(self, beta):
        self.beta = beta

    def update(self, T, k):
        return T - self.beta


class GeomStrategy(AnnealingStrategy):

    def __init__(self, alpha):
        self.alpha = alpha

    def update(self, T, k):
        return T * self.alpha


class CauchyStrategy(AnnealingStrategy):

    def update(self, T, k):
        return k / (k - 1) * T


class ExpStrategy(AnnealingStrategy):

    def __init__(self, c):
        self.c = c

    def update(self, T, k):
        return T * np.exp(-self.c)
