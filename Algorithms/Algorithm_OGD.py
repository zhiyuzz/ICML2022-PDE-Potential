import numpy as np


# Online gradient descent on the unit L2 norm ball, with learning rate 1/sqrt(t)
# d: dimension of the ball
class OGDBall:
    def __init__(self, d):
        self.prediction = np.zeros(d)
        self.t = 1

    def get_prediction(self):
        return self.prediction

    # Projected gradient step
    def update(self, gt):
        temp = self.prediction - gt / np.sqrt(self.t)
        if np.linalg.norm(temp) > 1:
            self.prediction = temp / np.linalg.norm(temp)
        else:
            self.prediction = temp
        self.t += 1
