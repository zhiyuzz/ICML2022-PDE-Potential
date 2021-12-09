import numpy as np
from scipy import special
from scipy.optimize import fsolve


# One dimensional parameter-free OLO based on \bar V_{1/2} (the novel potential)
# C: scaling constant
# Z: sufficient statistic; sum of coins before (excluding) time t, divided by sqrt(2t)
class OneDimPositive:
    def __init__(self, C):
        self.C = C
        self.Z = 0
        self.t = 1

    def __potential(self, x):
        return self.C * (np.sqrt(np.pi) * x * special.erfi(x) - np.exp(x ** 2))

    def get_prediction(self):
        potential_diff = self.__potential(self.Z + 1 / np.sqrt(2 * self. t)) - self.__potential(self.Z - 1 / np.sqrt(2 * self. t))
        return potential_diff * np.sqrt(self.t) / 2

    def update(self, gt):
        self.Z = (np.sqrt(2 * self.t) * self.Z - gt) / np.sqrt(2 * (self.t + 1))
        self.t += 1


# One dimensional parameter-free OLO based on \bar V_{-1/2} (the existing potential)
# C: scaling constant
# Z: sufficient statistic; sum of coins before (excluding) time t, divided by sqrt(2t)
class OneDimNegative:
    def __init__(self, C):
        self.C = C
        self.Z = 0
        self.t = 1

    def __potential(self, x):
        return self.C * np.exp(x ** 2)

    def get_prediction(self):
        potential_diff = self.__potential(self.Z + 1 / np.sqrt(2 * self. t)) - self.__potential(self.Z - 1 / np.sqrt(2 * self. t))
        return potential_diff / np.sqrt(self.t) / 2

    def update(self, gt):
        self.Z = (np.sqrt(2 * self.t) * self.Z - gt) / np.sqrt(2 * (self.t + 1))
        self.t += 1


# One-dimensional KT algorithm
# eps: initial wealth; Wealth: current wealth; beta: betting fraction; prediction: bet
class OneDimKT:
    def __init__(self, eps):
        self.Wealth = eps
        self.beta = 0
        self.prediction = 0
        self.t = 1

    def get_prediction(self):
        self.prediction = self.beta * self.Wealth
        return self.prediction

    def update(self, gt):
        self.Wealth = self.Wealth - gt * self.prediction
        self.beta = (self.beta * self.t - gt) / (self.t + 1)
        self.t += 1


# Compute the instrumental variable in the Fenchel conjugate of \bar V_{1/2}
# (the "z" in the discussion of Corollary 12)
# Input x represents the |u| in Corollary 12
def potential_conjugate(x, C):
    guess = np.sqrt(np.log(1 + x / np.sqrt(2) / C))
    return fsolve(lambda z: np.sqrt(np.pi / 2) * C * special.erfi(z) - x, np.array([guess]))


# Additional baseline: the wealth version of V_{-1/2}
class OneDimNegWealth:
    def __init__(self, eps):
        self.Wealth = eps
        self.beta = 0
        self.Z = 0
        self.prediction = 0
        self.t = 1

    def get_prediction(self):
        temp1 = np.exp((self.Z + 1 / np.sqrt(2 * self.t)) ** 2)
        temp2 = np.exp((self.Z - 1 / np.sqrt(2 * self.t)) ** 2)
        self.beta = (temp1 - temp2) / (temp1 + temp2)
        self.prediction = self.beta * self.Wealth
        return self.prediction

    def update(self, gt):
        self.Z = (np.sqrt(2 * self.t) * self.Z - gt) / np.sqrt(2 * (self.t + 1))
        self.Wealth = self.Wealth - gt * self.prediction
        self.t += 1
