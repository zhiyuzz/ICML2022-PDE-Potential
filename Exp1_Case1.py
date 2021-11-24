from matplotlib import pyplot as plt
from Algorithms.Algorithms_1d import *

# Time horizon
T = 200

# Loss function l_t = |x_t-u_Star|
u_star = 1

# Create copies of different algorithms
A1 = OneDimPositive(1)
A2 = OneDimNegative(1)
A3 = OneDimKT(np.exp(1))

predictions1 = np.empty(T)
predictions2 = np.empty(T)
predictions3 = np.empty(T)

sumLoss1 = np.empty(T)
sumLoss2 = np.empty(T)
sumLoss3 = np.empty(T)

for t in range(T):
    # Get predictions
    predictions1[t] = A1.get_prediction()
    predictions2[t] = A2.get_prediction()
    predictions3[t] = A3.get_prediction()

    # Compute cumulative losses
    if t == 0:
        sumLoss1[t] = np.abs(predictions1[t] - u_star)
        sumLoss2[t] = np.abs(predictions2[t] - u_star)
        sumLoss3[t] = np.abs(predictions3[t] - u_star)
    else:
        sumLoss1[t] = np.abs(predictions1[t] - u_star) + sumLoss1[t - 1]
        sumLoss2[t] = np.abs(predictions2[t] - u_star) + sumLoss2[t - 1]
        sumLoss3[t] = np.abs(predictions3[t] - u_star) + sumLoss3[t - 1]

    # Update
    if predictions1[t] >= u_star:
        gt = 1
    else:
        gt = -1
    A1.update(gt)

    if predictions2[t] >= u_star:
        gt = 1
    else:
        gt = -1
    A2.update(gt)

    if predictions3[t] >= u_star:
        gt = 1
    else:
        gt = -1
    A3.update(gt)

plt.rcParams.update({'font.size': 14})
plt.plot(np.arange(1, T + 1), sumLoss1 / 1000, '-', label='Positive')
plt.plot(np.arange(1, T + 1), sumLoss2 / 1000, '-', label='Negative')
plt.plot(np.arange(1, T + 1), sumLoss3 / 1000, '-', label='KT')

plt.xlabel('Time')
plt.ylabel('Cumulative loss (x1000)')
plt.legend()
