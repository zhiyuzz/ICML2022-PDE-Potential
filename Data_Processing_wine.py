from sklearn import preprocessing
import numpy as np

# Specify the amount of samples and import the raw data
data_raw = np.loadtxt("Data/winequality-white.csv", delimiter=';', skiprows=1)
row, col = data_raw.shape

data = np.empty(data_raw.shape)
data[:, 0] = data_raw[:, -1]
data[:, 1:] = data_raw[:, 0:-1]

# Scale all the features to [0,1], and append the intercept term
min_max_scaler = preprocessing.MinMaxScaler()
data[:, 1:] = min_max_scaler.fit_transform(data[:, 1:])
data = np.append(data, np.ones([row, 1]), 1)

# Scale such that the maximum L2 norm of feature vectors is 1
scaler = np.max(np.linalg.norm(data[:, 1:], axis=1))
data[:, 1:] /= scaler

np.savetxt("Data/Processed_wine.txt", data[:, :])
