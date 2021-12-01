import numpy as np

data = np.loadtxt("Data/YearPredictionMSD.txt", delimiter=',')
for row in range(data.shape[0]):
    norm = np.linalg.norm(data[row, 1:])
    data[row, 1:] /= norm

np.savetxt("Data/Processed.txt", data[:, :])
