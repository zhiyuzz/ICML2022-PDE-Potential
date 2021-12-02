from matplotlib import pyplot as plt
from Algorithms.Algorithm_1d import *

plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 3

C = 1   # Hyperparameter

# Different settings of u_star
settings = [0.1, 1, 10, 100, 1000, 10000]

# Loss function l_t = |x_t-u_Star|
for ind in range(len(settings)):
    u_star = settings[ind]

    # Time horizon
    if u_star >= 100:
        T = 500
    else:
        T = 200

    # Create instances of different algorithms
    algorithms = {
        "pos": OneDimPositive(C),
        "neg": OneDimNegative(C),
        "KT": OneDimKT(np.sqrt(np.exp(1)) * C)
    }

    # Create arrays to store the predictions and cumulative losses
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

    # Compute the predictions of the three algorithms
    for t in range(T):
        for key in algorithms:

            # Get predictions
            predictions[key][t] = algorithms[key].get_prediction()

            # Compute cumulative losses
            if t == 0:
                sum_losses[key][t] = np.abs(predictions[key][t] - u_star)
            else:
                sum_losses[key][t] = np.abs(predictions[key][t] - u_star) + sum_losses[key][t - 1]

            # Update
            if predictions[key][t] >= u_star:
                gt = 1
            else:
                gt = -1
            algorithms[key].update(gt)

    # Compute the regret upper bound from Corollary 12 (associated with \bar V_{1/2})
    instrumental = potential_conjugate(u_star, C)
    bound = np.empty(T)
    for t in range(T):
        bound[t] = C * np.sqrt(t + 1) * np.exp(instrumental ** 2)

    # Plotting the results; for clarity, scale the vertical axis
    if u_star >= 100:
        bound = bound / 10000
        for key in algorithms:
            sum_losses[key] = sum_losses[key] / 10000

    plt.figure()
    plt.plot(np.arange(1, T + 1), sum_losses["pos"], '-', label=r"$\bar V_{1/2}$ (ours)")
    plt.plot(np.arange(1, T + 1), sum_losses["neg"], '-', label=r"$\bar V_{-1/2}$")
    plt.plot(np.arange(1, T + 1), sum_losses["KT"], '-', label="KT")
    plt.plot(np.arange(1, T + 1), bound, '--', label="Upper bound")

    plt.title(r"$u^*=$" + str(u_star))
    plt.xlabel('Time')
    if u_star >= 100:
        plt.ylabel("Regret (x10000)")
    else:
        plt.ylabel("Regret")
    plt.legend()

    plt.savefig("Figures/OneD_" + str(ind + 1) + ".pdf", bbox_inches='tight')
