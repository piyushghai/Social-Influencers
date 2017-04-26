import numpy as np


def preprocessData(filename):
    trainfile = open(filename)
    for line in trainfile:
        head = line.rstrip().split(',')
        break

    X_train_A = []

    X_train_B = []
    y_train = []

    for line in trainfile:
        splitted = line.rstrip().split(',')
        label = int(splitted[0])
        A_features = [float(item) for item in splitted[1:12]]
        B_features = [float(item) for item in splitted[12:]]
        y_train.append(label)
        X_train_A.append(A_features)
        X_train_B.append(B_features)
    # print A_features
    trainfile.close()

    y_train = np.array(y_train)
    X_train_A = np.array(X_train_A)
    X_train_B = np.array(X_train_B)

    for i in range(len(y_train)):
        if y_train[i] == 0:
            y_train[i] = -1

    X_train = transform_features(X_train_A) - transform_features(X_train_B)
    np.random.seed(1)  # Seed the random number generator to preserve the dev/test split
    permutation = np.random.permutation(X_train.shape[0])
    X_train = X_train[permutation,]
    y_train = y_train[permutation,]
    X_train = normalize(X_train)

    indices = np.random.permutation(X_train.shape[0])
    training_idx, test_idx = indices[:5000], indices[5000:]
    X_t = X_train[training_idx, :]
    X_dev = X_train[test_idx, :]
    y_t = y_train[training_idx,]
    y_dev = y_train[test_idx,]

    X_train = X_t
    Y_train = y_t

    X_train = X_train[:, [2, 5, 8, 9]]
    X_dev = X_dev[:, [2, 5, 8, 9]]

    return X_train, Y_train, X_dev, y_dev


def transform_features(x):
    return np.log(1 + x)


def normalize(X):
    X_norm = X
    cols = X.shape[1]
    for i in range(cols):
        m = np.mean(X[:, i])
        std = np.std(X[:, i])
        X_norm[:, i] = (X[:, i] - m) / std
    return X_norm
    # return X
