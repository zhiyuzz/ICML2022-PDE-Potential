from matplotlib import pyplot as plt
from Algorithms.Algorithm_higherD import *

plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 3

T = 50
data = np.loadtxt("Data/Processed.txt", delimiter=',', max_rows=T)
_, col = data.shape

alg = HigherDimPositive(1, col - 1)
predictions = np.empty(T)
sum_losses = np.empty(T)

for t in range(T):

    # Get predictions
    predictions[t] = alg.get_prediction()

    # Compute cumulative losses
    if t == 0:
        sum_losses[t] = np.abs(predictions[t] - u_star)
    else:
        sum_losses[key][t] = np.abs(predictions[key][t] - u_star) + sum_losses[key][t - 1]

    # Update
    if predictions[key][t] >= u_star:
        gt = 1
    else:
        gt = -1
    algorithms[key].update(gt)
