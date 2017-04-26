from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import GaussianNB

import load_test_data
import pre_process

X_train, Y_train, X_dev, Y_dev = pre_process.preprocessData('train.csv')
X_test, Y_test = load_test_data.loadTestData('test.csv')

if __name__ == "__main__":
    model = GaussianNB()
    model.fit(X_train, Y_train)

    p_train = model.predict_proba(X_train)[:, 1]
    p_dev = model.predict_proba(X_dev)[:, 1]

    false_positive_rate, true_positive_rate, _ = roc_curve(Y_train, p_train)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print "ROC _ Train  -- ", roc_auc

    false_positive_rate, true_positive_rate, _ = roc_curve(Y_dev, p_dev)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print "ROC _ Dev  -- ", roc_auc
