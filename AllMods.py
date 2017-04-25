import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import roc_curve, auc

import load_test_data
import pre_process

X_train, Y_train, X_dev, Y_dev = pre_process.preprocessData('train.csv')
X_test, Y_test = load_test_data.loadTestData('test.csv')

if __name__ == "__main__":
    lr1 = linear_model.LogisticRegression(fit_intercept=False, max_iter=100, penalty='l2', verbose=1, C=1.0)
    lr1.fit(X_train, Y_train)

    lr5 = linear_model.LogisticRegression(fit_intercept=False, max_iter=100, penalty='l2', verbose=1, C=5.0)
    lr5.fit(X_train, Y_train)

    lr10 = linear_model.LogisticRegression(fit_intercept=False, max_iter=100, penalty='l2', verbose=1, C=10.0)
    lr10.fit(X_train, Y_train)

    lr100 = linear_model.LogisticRegression(fit_intercept=False, max_iter=100, penalty='l2', verbose=1, C=100.0)
    lr100.fit(X_train, Y_train)

    plot_label = ['Logistic Regression : 1',
                  'Logistic Regression : 0.01']
    plot_color = ['b--', 'r--']

    for i, clf in enumerate((lr1, lr10)):
        ydev = clf.predict_proba(X_dev)[:, 1]

        # Analyze the results
        false_positive_rate, true_positive_rate, _ = roc_curve(Y_dev, ydev)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        roc_label = '{0} {1:0.5f}'.format(plot_label[i], roc_auc)

        # Graph results
        if i == 0:
            plt.plot(false_positive_rate, true_positive_rate, plot_color[i], label=roc_label, linestyle='dashed',
                     linewidth=0.5)
        else:
            plt.plot(false_positive_rate, true_positive_rate, plot_color[i], label=roc_label, linestyle='dotted',
                     linewidth=1)

    # Graph Labels
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'k--')  # plot the diagonal
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    # plt.show()
    plt.savefig('lr_comp.png')
