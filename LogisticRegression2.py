from sklearn import linear_model

import load_test_data
import pre_process
import roc_curves

X_train, Y_train, X_dev, Y_dev = pre_process.preprocessData('train.csv')
X_test, Y_test = load_test_data.loadTestData('test.csv')

if __name__ == "__main__":
    model = linear_model.LogisticRegression(fit_intercept=False, max_iter=100, penalty='l2', verbose=1)
    model.fit(X_train, Y_train)
    # p_train = model.predict_proba(X_train)[:, 1]
    # roc_curves.plotROCCuves(Y_train, p_train, 'roc_lr_train.png', 'Logistic Regression')

    p_dev = model.predict_proba(X_dev)[:, 1]
    roc_curves.plotROCCuves(Y_dev, p_dev, 'roc_lr_dev.png', 'Logistic Regression')