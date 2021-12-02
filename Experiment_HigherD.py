from matplotlib import pyplot as plt
from Algorithms.Algorithm_higherD import *

plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 3

# Time horizon
T = 1000
# data = np.loadtxt("Data/Processed.txt", max_rows=T)
# _, col = data.shape

data = np.empty([T, 91])
for row in range(T):
    data[row, 0] = 2000
    data[row, 1:] =

# Hyperparameter
C = 1

# Scaling factor of the loss
gamma = 1

# Shifting parameter representing the initial guess
initial = np.zeros(col - 1)

algorithms = {
    "pos": HigherDimPositive(C, col - 1),
    "neg": HigherDimNegative(C, col - 1),
    "KT": HigherDimKT(np.sqrt(np.exp(1)) * C, col - 1)
}

sum_losses = {
    "pos": np.empty(T),
    "neg": np.empty(T),
    "KT": np.empty(T)
}

for t in range(T):
    # Get the features and target
    target = data[t, 0]
    feature = data[t, 1:]

    for key in algorithms:
        # Get prediction from the OLO algorithm
        prediction = algorithms[key].get_prediction() + initial

        # Compute the output of the model
        output = prediction @ feature / gamma

        # Compute cumulative losses
        if t == 0:
            sum_losses[key][t] = gamma * np.abs(output - target)
        else:
            sum_losses[key][t] = gamma * np.abs(output - target) + sum_losses[key][t - 1]

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
