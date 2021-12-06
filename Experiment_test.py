from matplotlib import pyplot as plt
from Algorithms.Algorithm_higherD import *

plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 3

# Time horizon
T = 1000
data = np.loadtxt("Data/Processed.txt", max_rows=T)
data = data[:, 0:2]
_, col = data.shape

# Random index set
rand_index = np.random.permutation(T)

# col = 91
# data = np.empty([T, col])
# for row in range(T):
#     data[row, 0] = 1
#     data[row, 1:] = np.ones(col - 1) / np.sqrt(col - 1)

# Hyperparameter
C = 1

# algorithms = {
#     "pos": HigherDimPositive(C, col - 1),
#     "neg": HigherDimNegative(C, col - 1),
#     "KT": HigherDimKT(np.sqrt(np.exp(1)) * C, col - 1)
# }

algorithms = {
    "pos": OneDimPositive(C),
    "neg": OneDimNegative(C),
    "KT": OneDimKT(np.sqrt(np.exp(1)) * C)
}

predictions = {
    "pos": np.empty(T),
    "neg": np.empty(T),
    "KT": np.empty(T)
}

sum_losses = {
    "pos": np.empty(T),
    "neg": np.empty(T),
    "KT": np.empty(T)
}

gradients = {
    "pos": np.empty(T),
    "neg": np.empty(T),
    "KT": np.empty(T)
}

for t in range(T):
    ind = rand_index[t]

    # Get the features and target
    # target = data[ind, 0]
    target = np.array([2000]) + 1000 * np.random.uniform()
    # feature = data[ind, 1:] * 6
    feature = np.array([1])

    for key in algorithms:
        # Get prediction from the OLO algorithm

        # prediction = algorithms[key].get_prediction()
        prediction = np.array([algorithms[key].get_prediction()])

        # Compute the output of the model
        output = prediction @ feature

        # test
        predictions[key][t] = prediction

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

        gradients[key][t] = gt

        algorithms[key].update(gt)

plt.figure()
plt.plot(np.arange(1, T + 1), sum_losses["pos"], '-', label=r"$\bar V_{1/2}$ (ours)")
plt.plot(np.arange(1, T + 1), sum_losses["neg"], '-', label=r"$\bar V_{-1/2}$")
plt.plot(np.arange(1, T + 1), sum_losses["KT"], '-', label="KT")
plt.legend()

# plt.figure()
# plt.plot(np.arange(1, T + 1), predictions["pos"], '-', label=r"$\bar V_{1/2}$ (ours)")
# plt.plot(np.arange(1, T + 1), predictions["neg"], '-', label=r"$\bar V_{-1/2}$")
# plt.plot(np.arange(1, T + 1), predictions["KT"], '-', label="KT")
# plt.legend()
