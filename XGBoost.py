from xgboost import XGBClassifier
from sklearn import ensemble
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score

import load_test_data
import pre_process
import write_to_csv

X_train, Y_train, X_dev, Y_dev = pre_process.preprocessData('train.csv')
X_test, Y_test = load_test_data.loadTestData('test.csv')

if __name__ == "__main__":
    model = XGBClassifier()
    model.fit(X_train, Y_train)
