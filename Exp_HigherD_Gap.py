from matplotlib import pyplot as plt
from Algorithms.Algorithm_higherD import *

plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 3

T = 50000
data = np.loadtxt("Data/Processed_music_scaled.txt", max_rows=T)
_, col = data.shape

# Hyperparameter
C = 1

# Different settings of scaling factor (for the target)
settings = [0.00001, 0.00001 * np.sqrt(10), 0.0001, 0.0001 * np.sqrt(10), 0.001, 0.001 * np.sqrt(10), 0.01]

final_sum = {
    "pos": np.empty(len(settings)),
    "neg": np.empty(len(settings)),
    "KT": np.empty(len(settings))
}

# Repetition for each setting
reps = 5

# Random seed
rng = np.random.default_rng(1)

# Main loop
for setting in range(len(settings)):
    scaling = settings[setting]

    sum_losses = {
        "pos": np.empty(reps),
        "neg": np.empty(reps),
        "KT": np.empty(reps)
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
                    sum_losses[key][rep] = np.abs(output - target)
                else:
                    sum_losses[key][rep] += np.abs(output - target)

                # Update
                if output >= target:
                    gt = feature
                else:
                    gt = -feature
                algorithms[key].update(gt)

    for key in final_sum:
        final_sum[key][setting] = np.mean(sum_losses[key])

plt.figure()
plt.plot(settings, (final_sum["KT"] - final_sum["pos"]) / 1000, linestyle='--', marker='o', label=r"$\bar V_{1/2}$ (ours)")
plt.xscale("log")

plt.xlabel(r"$u^*$ (log scale)")
plt.ylabel("Saved regret compared to KT (x1000)")
plt.title(r"$T=50000$")
plt.savefig("Figures/HigherD_Gap.pdf", bbox_inches='tight')
