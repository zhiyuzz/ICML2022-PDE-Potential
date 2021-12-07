from matplotlib import pyplot as plt
from Algorithms.Algorithm_higherD import *

plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 3

# Time horizon
T = 50000
data = np.loadtxt("Data/Processed_music.txt", max_rows=T)
_, col = data.shape

# data = np.loadtxt("Data/Processed_wine.txt")
# T, col = data.shape

# T = 20000
# data = np.loadtxt("Data/Processed_superconductor.txt", max_rows=T)
# _, col = data.shape

# Random index set
rand_index = np.random.permutation(T)

# Hyperparameter
C = 1

algorithms = {
    "pos": HigherDimPositive(C, col - 1),
    "neg": HigherDimNegative(C, col - 1),
    "KT": HigherDimNegWealth(np.sqrt(np.exp(1)) * C, col - 1)
}

sum_losses = {
    "pos": np.empty(T),
    "neg": np.empty(T),
    "KT": np.empty(T)
}

for t in range(T):
    ind = rand_index[t]

    # Get the features and target (0.001 and 0.01 are good for our algorithm)
    target = data[ind, 0] * 10
    feature = data[ind, 1:] / np.linalg.norm(data[ind, 1:])

    for key in algorithms:
        # Get prediction from the OLO algorithm
        prediction = algorithms[key].get_prediction()

        # Compute the output of the model
        output = prediction @ feature

        # Compute cumulative losses
        if t == 0:
            sum_losses[key][t] = np.abs(output - target)
        else:
            sum_losses[key][t] = np.abs(output - target) + sum_losses[key][t - 1]

        # Update
        if output >= target:
            gt = feature
        else:
            gt = -feature
        algorithms[key].update(gt)

plt.figure()
plt.plot(np.arange(1, T + 1), sum_losses["pos"], '-', label=r"$\bar V_{1/2}$ (ours)")
plt.plot(np.arange(1, T + 1), sum_losses["neg"], '-', label=r"$\bar V_{-1/2}$")
plt.plot(np.arange(1, T + 1), sum_losses["KT"], '-', label="KT")
plt.legend()
