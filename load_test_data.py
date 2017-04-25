import numpy as np


def loadTestData(filename):
    testfile = open(filename)
    # ignore the test header
    testfile.next()

    X_test_A = []
    X_test_B = []
    Y_test = []
    for line in testfile:
        splitted = line.rstrip().split(',')
        label = int(splitted[0])
        A_features = [float(item) for item in splitted[0:11]]
        B_features = [float(item) for item in splitted[11:]]
        Y_test.append(label)
        X_test_A.append(A_features)
        X_test_B.append(B_features)
    testfile.close()

    X_test_A = np.array(X_test_A)
    X_test_B = np.array(X_test_B)

    # transform features in the same way as for training to ensure consistency
    X_test = transform_features(X_test_A) - transform_features(X_test_B)
    X_test = normalize(X_test)

    for i in range(len(Y_test)):
        if Y_test[i] == 0:
            Y_test[i] = -1

    return X_test, Y_test


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
