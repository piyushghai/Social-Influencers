import numpy as np
from sklearn import cross_validation
from sklearn import ensemble
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score

import load_test_data
import pre_process
import write_to_csv

X_train, Y_train, X_dev, Y_dev = pre_process.preprocessData('train.csv')
X_test, Y_test = load_test_data.loadTestData('test.csv')

if __name__ == "__main__":
    model = ensemble.GradientBoostingClassifier()

    print "40 Fold CV Score: ",
    np.mean(cross_validation.cross_val_score(model, X_train, Y_train, cv=40, scoring='roc_auc'))
    model.fit(X_train, Y_train)
    probs = model.predict_proba(X_train)

    precision, recall, thresholds = precision_recall_curve(Y_train, probs[:, 1])
    print 'AuC score on training data:', roc_auc_score(Y_train, probs[:, 1])

    probs_test = model.predict_proba(X_test)

    write_to_csv.writeToCSV('preds_gb_40.csv', probs_test[:, 1])
