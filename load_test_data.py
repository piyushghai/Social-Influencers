import numpy as np


def loadTestData(filename):
    testfile = open(filename)
    # ignore the test header
    for line in testfile:
        head = line.rstrip().split(',')
        break

    X_test_A = []
    X_test_B = []
    Y_test = []
    for line in testfile:
        splitted = line.rstrip().split(',')
        A_features = [float(item) for item in splitted[0:11]]
        B_features = [float(item) for item in splitted[11:]]
        X_test_A.append(A_features)
        X_test_B.append(B_features)
    testfile.close()

    X_test_A = np.array(X_test_A)
    X_test_B = np.array(X_test_B)

    # transform features in the same way as for training to ensure consistency
    X_test = transform_features(X_test_A) - transform_features(X_test_B)
    X_test = normalize(X_test)

    Y_test = getYLabels()

    X_test = X_test[:, [2, 5, 8, 9]]

    return X_test, Y_test


def getYLabels():
    testRec = open('sample_predictions.csv')
    for line in testRec:
        head = line.rstrip().split(',')
        break
    yPred = []
    for line in testRec:
        splitted = line.rstrip().split(',')
        val = float(splitted[1])
        if val >= 0.5:
            yPred.append(1)
        else:
            yPred.append(-1)
    return yPred


def transform_features(x):
    return np.log(1 + x)


def normalize(X):
    # X_norm = X
    # cols = X.shape[1]
    # for i in range(cols):
    #     m = np.mean(X[:, i])
    #     std = np.std(X[:, i])
    #     X_norm[:, i] = (X[:, i] - m) / std
    # return X_norm
    return X
