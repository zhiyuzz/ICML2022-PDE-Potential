from matplotlib import pyplot as plt
from Algorithms.Algorithm_higherD import *

plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 3

# Dataset 1
T = 50000
data = np.loadtxt("Data/Processed_music_scaled.txt", max_rows=T)
_, col = data.shape

# Dataset 2
# data = np.loadtxt("Data/Processed_wine.txt")
# T, col = data.shape

# Dataset 3
# T = 20000
# data = np.loadtxt("Data/Processed_superconductor.txt", max_rows=T)
# _, col = data.shape

# Hyperparameter
C = 1

# Different settings of scaling factor (for the target)
settings = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

# Repetition for each setting
reps = 5

# Random seed
rng = np.random.default_rng(1)

# Main loop
for setting in range(len(settings)):
    scaling = settings[setting]

    algorithms = {
        "pos": HigherDimPositive(C, col - 1),
        "neg": HigherDimNegative(C, col - 1),
        "KT": HigherDimKT(np.sqrt(np.exp(1)) * C, col - 1)
    }

    sum_losses = {
        "pos": np.empty([reps, T]),
        "neg": np.empty([reps, T]),
        "KT": np.empty([reps, T])
    }

    sum_losses_mean = {
        "pos": np.empty(T),
        "neg": np.empty(T),
        "KT": np.empty(T)
    }

    for rep in range(reps):
        # Random index set
        rand_index = rng.permutation(T)

        for t in range(T):
            ind = rand_index[t]

            # Get the features and target
            target = data[ind, 0] * scaling
            feature = data[ind, 1:]

            for key in algorithms:
                # Get prediction from the OLO algorithm
                prediction = algorithms[key].get_prediction()

                # Compute the output of the model
                output = prediction @ feature

                # Compute cumulative losses
                if t == 0:
                    sum_losses[key][rep, t] = np.abs(output - target)
                else:
                    sum_losses[key][rep, t] = np.abs(output - target) + sum_losses[key][rep, t - 1]

                # Update
                if output >= target:
                    gt = feature
                else:
                    gt = -feature
                algorithms[key].update(gt)

    # Compute the mean of the cumulative losses
    for key in algorithms:
        sum_losses_mean[key] = np.mean(sum_losses[key], axis=0)

    plt.figure()
    plt.plot(np.arange(1, T + 1), sum_losses_mean["pos"], '-', label=r"$\bar V_{1/2}$ (ours)")
    plt.plot(np.arange(1, T + 1), sum_losses_mean["neg"], '-', label=r"$\bar V_{-1/2}$")
    plt.plot(np.arange(1, T + 1), sum_losses_mean["KT"], '-', label="KT")
    plt.legend()

    plt.xlabel('Time')
    plt.ylabel("Cumulative loss")
    plt.title(r"$\gamma=$" + str(settings[setting]))
    plt.savefig("Figures/HigherD_" + str(setting + 1) + ".pdf", bbox_inches='tight')

