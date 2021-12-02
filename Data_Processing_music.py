from sklearn import preprocessing
import numpy as np

# Specify the amount of samples and import the raw data
T = 50000
data = np.loadtxt("Data/YearPredictionMSD.txt", delimiter=',', max_rows=T)

# Scale all the features to [0,1], and append the intercept term
min_max_scaler = preprocessing.MinMaxScaler()
data[:, 1:] = min_max_scaler.fit_transform(data[:, 1:])
data = np.append(data, np.ones([T, 1]), 1)

# Scale such that the maximum L2 norm of feature vectors is 1
scaler = np.max(np.linalg.norm(data[:, 1:], axis=1))
data[:, 1:] /= scaler

np.savetxt("Data/Processed_music.txt", data[:, :])
