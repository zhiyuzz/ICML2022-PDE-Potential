from matplotlib import pyplot as plt
from Algorithms.Algorithm_1d import *

plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['lines.markersize'] = 6

C = 1   # Hyperparameter

# Different settings of u_star
settings = [0.01, 0.01 * np.sqrt(10), 0.1, 0.1 * np.sqrt(10), 1, np.sqrt(10), 10]

final_sum = {
    "pos": np.empty(len(settings)),
    "neg": np.empty(len(settings)),
    "KT": np.empty(len(settings))
}

# Loss function l_t = |x_t-u_Star|
for ind in range(len(settings)):
    u_star = settings[ind]

    # Time horizon
    T = 500

    # Create instances of different algorithms
    algorithms = {
        "pos": OneDimPositive(C),
        "neg": OneDimNegative(C),
        "KT": OneDimKT(np.sqrt(np.exp(1)) * C)
    }

    # Create arrays to store cumulative losses
    sum_losses = {
        "pos": 0,
        "neg": 0,
        "KT": 0
    }

    # Compute the predictions of the three algorithms
    for t in range(T):
        for key in algorithms:

            # Get predictions
            prediction = algorithms[key].get_prediction()

            # Compute cumulative losses
            if t == 0:
                sum_losses[key] = np.abs(prediction - u_star)
            else:
                sum_losses[key] += np.abs(prediction - u_star)

            # Update
            if prediction >= u_star:
                gt = 1
            else:
                gt = -1
            algorithms[key].update(gt)

    for key in algorithms:
        final_sum[key][ind] = sum_losses[key]

plt.figure()
plt.plot(settings, final_sum["KT"] - final_sum["pos"], linestyle='--', marker='o', label=r"$\bar V_{1/2}$ (ours)")
# plt.plot(settings, final_sum["KT"] - final_sum["neg"], linestyle='--', marker='o', label=r"$\bar V_{-1/2}$")
# plt.legend(loc="upper left")
plt.xscale("log")

plt.xlabel(r"$u^*$ (log scale)")
plt.ylabel("Saved regret compared to KT")
plt.title(r"$T=500$")

plt.savefig("Figures/OneD_Gap.pdf", bbox_inches='tight')
