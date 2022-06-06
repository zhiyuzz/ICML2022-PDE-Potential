from matplotlib import pyplot as plt
from Algorithms.Algorithm_1d import *

plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['lines.markersize'] = 6

C = 1   # Hyperparameter
T = 500    # Time horizon

# Setting for repeated experiment
p = 0.2    # Mean of the loss gradients; loss gradients are iid RV on the support {-1, 1}
N = 50  # Number of repeated trials
rng = np.random.default_rng(2022)   # Initialize the random number generator

# Create arrays to store cumulative losses of each algorithm, in each repeated experiment
sum_losses = {
    "pos": np.empty([N, T]),
    "neg": np.empty([N, T]),
    "KT": np.empty([N, T])
}

for n in range(N):
    # Create instances of the considered algorithms
    algorithms = {
        "pos": OneDimPositive(C),
        "neg": OneDimNegative(C),
        "KT": OneDimKT(np.sqrt(np.exp(1)) * C)
    }

    # The random loss gradients
    gradients = 2 * rng.binomial(1, (1+p)/2, T) - 1

    # Compute the predictions of the three algorithms
    for t in range(T):
        for key in algorithms:
            # Get predictions
            prediction = algorithms[key].get_prediction()

            # Define the loss
            gt = gradients[t]

            # Compute cumulative losses
            if t == 0:
                sum_losses[key][n, t] = prediction * gt
            else:
                sum_losses[key][n, t] = sum_losses[key][n, t-1] + prediction * gt

            algorithms[key].update(gt)

mean_pos = np.mean(-sum_losses["pos"], axis=0)
mean_neg = np.mean(-sum_losses["neg"], axis=0)
mean_KT = np.mean(-sum_losses["KT"], axis=0)

# std_pos = np.std(-sum_losses["pos"], axis=0)
# std_neg = np.std(-sum_losses["neg"], axis=0)
# std_KT = np.std(-sum_losses["KT"], axis=0)

plt.figure()
plt.plot(np.arange(1, T + 1), mean_pos, '-', label=r"$\bar V_{1/2}$ (ours)")
plt.plot(np.arange(1, T + 1), mean_neg, '-', label=r"$\bar V_{-1/2}$")
plt.plot(np.arange(1, T + 1), mean_KT, '-', label="KT")

# plt.fill_between(np.arange(1, T + 1), mean_pos - std_pos, mean_pos + std_pos, color='C0', alpha=0.2)
# plt.fill_between(np.arange(1, T + 1), mean_neg - std_neg, mean_neg + std_neg, color='C1', alpha=0.2)
# plt.fill_between(np.arange(1, T + 1), mean_KT - std_KT, mean_KT + std_KT, color='C2', alpha=0.2)

plt.xlabel('t')
plt.ylabel('Negative cumulative loss')
plt.legend(loc='upper left')

plt.savefig("Figures/OneD_Stochastic.pdf", bbox_inches='tight')
