from Algorithms.Algorithm_1d import *
from Algorithms.Algorithm_OGD import OGDBall


class HigherDimPositive:
    # d: dimension of the domain
    def __init__(self, C, d):
        self.A_r = OneDimPositive(C)
        self.A_B = OGDBall(d)
        self.zt = np.empty(d)

    def get_prediction(self):
        self.zt = self.A_B.get_prediction()
        return self.A_r.get_prediction() * self.zt

    def update(self, gt):
        self.A_r.update(gt @ self.zt)
        self.A_B.update(gt)


class HigherDimNegative:
    # d: dimension of the domain
    def __init__(self, C, d):
        self.A_r = OneDimNegative(C)
        self.A_B = OGDBall(d)
        self.zt = np.empty(d)

    def get_prediction(self):
        self.zt = self.A_B.get_prediction()
        return self.A_r.get_prediction() * self.zt

    def update(self, gt):
        self.A_r.update(gt @ self.zt)
        self.A_B.update(gt)


class HigherDimKT:
    # d: dimension of the domain
    def __init__(self, C, d):
        self.A_r = OneDimKT(C)
        self.A_B = OGDBall(d)
        self.zt = np.empty(d)

    def get_prediction(self):
        self.zt = self.A_B.get_prediction()
        return self.A_r.get_prediction() * self.zt

    def update(self, gt):
        self.A_r.update(gt @ self.zt)
        self.A_B.update(gt)


# Additional baseline: the wealth version of V_{-1/2}
# class HigherDimNegWealth:
#     # d: dimension of the domain
#     def __init__(self, C, d):
#         self.A_r = OneDimNegWealth(C)
#         self.A_B = OGDBall(d)
#         self.zt = np.empty(d)
#
#     def get_prediction(self):
#         self.zt = self.A_B.get_prediction()
#         return self.A_r.get_prediction() * self.zt
#
#     def update(self, gt):
#         self.A_r.update(gt @ self.zt)
#         self.A_B.update(gt)
