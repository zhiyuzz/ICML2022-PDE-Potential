from matplotlib import pyplot as plt
from Algorithms.Algorithm_higherD import *

plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 3

T = 50
data = np.loadtxt("Data/Processed.txt", delimiter=',', max_rows=T)
_, col = data.shape

alg = HigherDimPositive(1, col - 1)
sum_losses = np.empty(T)

for t in range(T):

    # Get prediction from the OLO algorithm
    prediction = alg.get_prediction()

    # Get the features and target
    target = data[t, 0]
    feature = data[t, 1:]

    # Compute the output of the model
    output = prediction @ feature

    # Compute cumulative losses
    if t == 0:
        sum_losses[t] = np.abs(output - target)
    else:
        sum_losses[t] = np.abs(output - target) + sum_losses[t - 1]

    # Update
    if output >= target:
        gt = feature
    else:
        gt = -feature
    alg.update(gt)
