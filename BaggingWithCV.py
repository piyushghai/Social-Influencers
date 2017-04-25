# Bagged Decision Trees for Classification
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

import load_test_data
import pre_process
import write_to_csv

X_train, Y_train, X_dev, Y_dev = pre_process.preprocessData('train.csv')
X_test, Y_test = load_test_data.loadTestData('test.csv')

if __name__ == "__main__":
    seed = 1729
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cart = DecisionTreeClassifier()
    num_trees = 200
    model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
    model.fit(X_train, Y_train)
    probs_tr = model.predict_proba(X_train)
    #
    precision, recall, thresholds = precision_recall_curve(Y_train, probs_tr[:, 1])
    print('AuC score on training data:', roc_auc_score(Y_train, probs_tr[:, 1]))

    probs_test = model.predict_proba(X_test)
    # probs_test = model_selection.cross_val_predict(model, X_test, cv=kfold, method='predict_proba')
    #
    write_to_csv.writeToCSV('preds_bagg_cv.csv', probs_test[:, 1])
