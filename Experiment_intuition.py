from matplotlib import pyplot as plt
from Algorithms.Algorithm_1d import *

plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 3

T = 20     # Time horizon
C = 1      # Constant C
u_star = 100  # Comparator

algorithms = {
    "pos": OneDimPositive(C),
    "neg": OneDimNegative(C)
}

predictions = {
    "pos": np.empty(T),
    "neg": np.empty(T)
}

for t in range(T):
    for key in algorithms:
        # Get predictions
        predictions[key][t] = algorithms[key].get_prediction()

        # Update
        if predictions[key][t] >= u_star:
            gt = 1
        else:
            gt = -1
        algorithms[key].update(gt)

plt.figure()
plt.axhline(y=u_star, color='y', linestyle='--', label=r"$u^*$")
plt.plot(np.arange(1, T + 1), predictions["pos"], '-', label=r"$\bar V_{1/2}$ (ours)")
plt.plot(np.arange(1, T + 1), predictions["neg"], '-', label=r"$\bar V_{-1/2}$")

plt.title(r"$u^*=$" + str(u_star) + r", $C$=1")
plt.xlabel('Time')
plt.ylabel(r"Prediction $x_t$")
plt.legend()
plt.savefig("Figures/OneD_intuition.pdf", bbox_inches='tight')
