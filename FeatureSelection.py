import numpy as np

import load_test_data
import pre_process

X_train, Y_train, X_dev, Y_dev = pre_process.preprocessData('train.csv')
X_test, Y_test = load_test_data.loadTestData('test.csv')


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


def runExperiments(X, Y, X_dev, Y_dev, lmda, learningRate, maxIter=10):
    W = logisticRegression(X, Y, maxIter, learningRate, lmda)
    nCorrect = 0
    nIncorrect = 0
    for i in range(len(Y_dev)):
        y_hat = np.sign(X_dev[i,].dot(W))
        if y_hat == Y_dev[i]:
            nCorrect += 1
        else:
            nIncorrect += 1

    accuracy_d = (nCorrect * 1.0 / (nIncorrect + nCorrect))

    return accuracy_d


def getAccuracy(X_train, y_train, featureSet, X_test, Y_test, lmda, learningRate, maxIter):
    featIdx = list(featureSet)
    X_t = X_train[:, featIdx]
    X_test = X_test[:, featIdx]
    return runExperiments(X_t, y_train, X_test, Y_test, lmda, learningRate, maxIter)


def FeatureSelection():
    ss = set()
    fs = [x for x in range(11)]
    fs = set(fs)
    bestAccuracy = 0.0
    while (True):
        bestFeature = None
        for feature in fs:
            if feature not in ss:
                ssPrime = ss.copy()
                ssPrime.add(feature)
                print ssPrime
                accu = getAccuracy(X_train, Y_train, ssPrime, X_train, Y_train, .001, 0.01, 10)
                print bestAccuracy, accu
                if accu > bestAccuracy:
                    bestFeature = feature
                    bestAccuracy = accu
        if bestFeature is not None:
            bestSet = set()
            bestSet.add(bestFeature)
            ss |= bestSet
        if bestFeature == None or len(ss) == 11:
            break
    return ss


if __name__ == "__main__":
    bestFeat = FeatureSelection()
    print "Best Features ", bestFeat
