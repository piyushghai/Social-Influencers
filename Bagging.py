from sklearn import ensemble
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score

import load_test_data
import pre_process
import write_to_csv

X_train, Y_train, X_dev, Y_dev = pre_process.preprocessData('train.csv')
X_test, Y_test = load_test_data.loadTestData('test.csv')

if __name__ == "__main__":
    model = ensemble.GradientBoostingClassifier(learning_rate=0.01, max_depth=20)
    model.fit(X_train, Y_train)

    probs_tr = model.predict_proba(X_train)
    precision, recall, thresholds = precision_recall_curve(Y_train, probs_tr[:, 1])
    print('AuC score on training data:', roc_auc_score(Y_train, probs_tr[:, 1]))

    p_test = model.predict_proba(X_test)

    write_to_csv.writeToCSV('preds_bagg.csv', p_test[:, 1])
