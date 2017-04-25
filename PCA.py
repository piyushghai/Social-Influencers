import load_test_data
import pre_process
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
import numpy as np

def transform_features(x):
    return np.log(1 + x)

def PCA(x,x_test):
    X_std = StandardScaler().fit_transform(x)
    sklearn_pca = sklearnPCA(n_components='mle', svd_solver='full')
    x = sklearn_pca.fit_transform(X_std)
    x_test = sklearn_pca.transform(x_test)

    return x,x_test

def preprocessData(filename,testFileName):
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

    testfile = open(testFileName)
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
    Y_test = getYLabels()

    X_train, X_test = PCA(X_train,X_test)

    indices = np.random.permutation(X_train.shape[0])
    training_idx, test_idx = indices[:5000], indices[5000:]
    X_t = X_train[training_idx, :]
    X_dev = X_train[test_idx, :]
    y_t = y_train[training_idx,]
    y_dev = y_train[test_idx,]

    X_train = X_t
    Y_train = y_t

    return X_train, Y_train, X_dev, y_dev, X_test, Y_test


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

