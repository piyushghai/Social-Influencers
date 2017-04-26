import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import roc_curve, auc

import load_test_data
import pre_process

X_train, Y_train, X_dev, Y_dev = pre_process.preprocessData('train.csv')
X_test, Y_test = load_test_data.loadTestData('test.csv')

if __name__ == "__main__":
    model = linear_model.LogisticRegression(fit_intercept=False, max_iter=10, penalty='l2', verbose=1, C=100)
    model.fit(X_train, Y_train)

    p_train = model.predict_proba(X_train)[:, 1]
    # # roc_curves.plotROCCuves(Y_train, p_train, 'roc_lr_train.png', 'Logistic Regression')
    #
    p_dev = model.predict_proba(X_dev)[:, 1]
    # roc_curves.plotROCCuves(Y_dev, p_dev, 'roc_lr_dev.png', 'Logistic Regression')

    # probs = model.predict_proba(X_test)[:, 1]
    # write_to_csv.writeToCSV('preds_lr.csv', probs)

    # Analyze the results
    false_positive_rate, true_positive_rate, _ = roc_curve(Y_train, p_train)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print "ROC _ Train  -- ", roc_auc
    roc_label = '{0} {1:0.5f}'.format("Logistic Regression - Train", roc_auc)

    # plt.plot(false_positive_rate, true_positive_rate, 'b--', label=roc_label, linestyle='dashed', linewidth=0.5)

    false_positive_rate, true_positive_rate, _ = roc_curve(Y_dev, p_dev)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print "ROC _ Dev  -- ", roc_auc
    # roc_label = '{0} {1:0.5f}'.format("Logistic Regression - Dev", roc_auc)
    #
    # plt.plot(false_positive_rate, true_positive_rate, 'r--', label=roc_label, linestyle='dotted', linewidth=0.5)
    #
    # # Graph Labels
    # plt.title('Receiver Operating Characteristic (ROC)')
    # plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'k--')  # plot the diagonal
    # plt.xlim([-0.1, 1.2])
    # plt.ylim([-0.1, 1.2])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    #
    # plt.savefig('lr_.png')
