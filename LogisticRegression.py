import numpy as np
from scipy.special import expit

import load_test_data
import pre_process

X_train, Y_train, X_dev, Y_dev = pre_process.preprocessData('train.csv')
X_test, Y_test = load_test_data.loadTestData('test.csv')


def prediction(x, W):
    return expit(x.dot(W))


def updateWeights(gradient, learningRate, W):
    W = W - (learningRate * gradient)
    return W


def logisticRegression(X, Y, maxIter, learningRate, lmda):
    W = np.zeros(X.shape[1])
    newLoss = ExpLoss(X, Y, W, lmda)
    prevLoss = 0.0
    count = 0

    while (True):
        count += 1
        for i in range(len(Y)):
            # Not converged... so continue
            d = X[i,]
            gradient = ExpLossGradient(d, Y[i], W, lmda)
            W = updateWeights(gradient, learningRate, W)
        prevLoss = newLoss
        #         print W
        newLoss = ExpLoss(X, Y, W, lmda)
        print "Iteration # : ", count, " Loss Value : ", newLoss

        if abs(newLoss - prevLoss) < 0.001:
            #             print "Difference in old and new loss less than ", 0.0001
            #             print "Total Iterations till now : ", count
            #             print "prevLoss", prevLoss
            #             print "newLoss", newLoss
            break

        if count == maxIter:
            #             print "MaxIterations reached!"
            break

    return W


def ExpLoss(X, Y, W, lmda):
    loss = lmda * (W.dot(W))
    yHat = X.dot(W)
    activation = -Y * yHat
    activationExp = np.exp(activation)
    loss += np.sum(activationExp)
    return loss


def ExpLossGradient(x, y, W, lmda):
    grad = (x.dot(W))
    grad = -y * grad
    grad = np.exp(grad)
    grad = -y * x * grad
    Wgrad = 2 * lmda * W
    Wgrad = Wgrad + grad
    return Wgrad


def runExperiments(X, Y, X_dev, Y_dev, X_test, Y_test, lmda, learningRate, maxIter=10):
    W = logisticRegression(X, Y, maxIter, learningRate, lmda)
    nCorrect = 0
    nIncorrect = 0
    for i in range(len(Y_test)):
        y_hat = np.sign(X_test[i,].dot(W))
        if y_hat == Y_test[i]:
            nCorrect += 1
        else:
            nIncorrect += 1
    accuracy_t = (nCorrect * 1.0 / (nIncorrect + nCorrect))

    nCorrect = 0
    nIncorrect = 0

    for i in range(len(Y_dev)):
        y_hat = np.sign(X_dev[i,].dot(W))
        if y_hat == Y_dev[i]:
            nCorrect += 1
        else:
            nIncorrect += 1

    accuracy_d = (nCorrect * 1.0 / (nIncorrect + nCorrect))

    nCorrect = 0
    nIncorrect = 0

    for i in range(len(Y)):
        y_hat = np.sign(X[i,].dot(W))
        if y_hat == Y[i]:
            nCorrect += 1
        else:
            nIncorrect += 1

    accuracy_tr = (nCorrect * 1.0 / (nIncorrect + nCorrect))

    return accuracy_d, accuracy_t, accuracy_tr


if __name__ == "__main__":
    accuracy_d, accuracy_t, accuracy_tr = runExperiments(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, lmda=.0001,
                                                         learningRate=.001,
                                                         maxIter=100)

    print 'Accuracy for Logistic Regresstion on Dev set : ', accuracy_d

    print 'Accuracy for Logistic Regresstion on Train set : ', accuracy_tr

    print 'Accuracy for Logistic Regresstion on Test set : ', accuracy_t
