import numpy as np
from scipy import special


class OneDimPositive:
    # One dimensional parameter-free OLO based on \bar V_{1/2} (the novel potential)
    # C: scaling constant
    # Z: sufficient statistic; sum of coins before (excluding) time t, divided by sqrt(2t)
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


class OneDimNegative:
    # One dimensional parameter-free OLO based on \bar V_{-1/2} (the existing potential)
    # C: scaling constant
    # Z: sufficient statistic; sum of coins before (excluding) time t, divided by sqrt(2t)
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


class OneDimKT:
    # One-dimensional KT algorithm
    # C: scaling constant
    # Z: sufficient statistic; sum of coins before (excluding) time t, divided by sqrt(2t)
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
