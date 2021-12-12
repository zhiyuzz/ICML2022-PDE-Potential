from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from Algorithms.Algorithm_higherD import *

plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 3

scale_format = ScalarFormatter(useMathText=True)
scale_format.set_powerlimits((0, 0))

T = 50000
data = np.loadtxt("Data/Processed_music_scaled.txt", max_rows=T)
_, col = data.shape

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

    sum_losses = {
        "pos": np.empty([reps, T]),
        "neg": np.empty([reps, T]),
        "KT": np.empty([reps, T])
    }

    for rep in range(reps):
        # Reinitialize algorithms
        algorithms = {
            "pos": HigherDimPositive(C, col - 1),
            "neg": HigherDimNegative(C, col - 1),
            "KT": HigherDimKT(np.sqrt(np.exp(1)) * C, col - 1)
        }

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

    plt.figure()
    plt.plot(np.arange(1, T + 1), np.mean(sum_losses["pos"], axis=0), '-', label=r"$\bar V_{1/2}$ (ours)")
    plt.plot(np.arange(1, T + 1), np.mean(sum_losses["neg"], axis=0), '-', label=r"$\bar V_{-1/2}$")
    plt.plot(np.arange(1, T + 1), np.mean(sum_losses["KT"], axis=0), '-', label="KT")

    plt.gca().yaxis.set_major_formatter(scale_format)

    plt.xlabel('Time')
    plt.ylabel("Cumulative loss")
    plt.title(r"$\gamma=$" + str(settings[setting]))
    plt.legend(loc="upper left")

    plt.savefig("Figures/HigherD_" + str(setting + 1) + ".pdf", bbox_inches='tight')
