from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
import load_test_data
import pre_process
import write_to_csv

X_train, Y_train, X_dev, Y_dev = pre_process.preprocessData('train.csv')
X_test, Y_test = load_test_data.loadTestData('test.csv')

seed=7
num_trees = 100
max_features = 'auto'
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)


model.fit(X_train, Y_train)
prob = model.predict_proba(X_test)
y_pred=[]
for x in prob:
    y_pred.append(x[0])

write_to_csv.writeToCSV('predRF.csv', y_pred)
