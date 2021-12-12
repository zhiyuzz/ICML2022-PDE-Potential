from sklearn import preprocessing
import numpy as np

# Specify the amount of samples and import the raw data
data = np.loadtxt("Data/YearPredictionMSD.txt", delimiter=',')
rows, cols = data.shape

# Scale all the features to [0,1]
min_max_scaler = preprocessing.MinMaxScaler()
data[:, 1:] = min_max_scaler.fit_transform(data[:, 1:])

# Scale such that the norm of feature vectors is 1
scaler = np.linalg.norm(data[:, 1:], axis=1)
for row in range(rows):
    data[row, 1:] /= scaler[row]

np.savetxt("Data/Processed_music_scaled.txt", data[:, :])
