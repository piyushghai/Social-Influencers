import os

# mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'
# os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb
import load_test_data
import pre_process
import write_to_csv
import roc_curves
from sklearn.model_selection  import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

def parameterTuning(X,Y):
    cv_params = {'learning_rate': [0.001,0.002,0.003,0.004, 0.005, 0.006,0.007], 'subsample': [0.5,0.7, 0.8, 0.9], 'max_depth': [1,3, 5, 7, 8, 9],
                 'min_child_weight': [1, 3, 5,7]}
    ind_params = {'n_estimators': 1000, 'seed': 0, 'colsample_bytree': 0.8,
                  'objective': 'binary:logistic'}

    optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), cv_params, scoring = 'accuracy', cv = 5, n_jobs = -1)
    optimized_GBM.fit(X, Y)
    print(optimized_GBM.grid_scores_)
    print(optimized_GBM.best_params_)

def model(X,Y,X_test,X_dev):
    xgdmat = xgb.DMatrix(X, Y)

    our_params = {'eta': 0.001, 'seed':40, 'subsample': 0.3, 'colsample_bytree': 0.5,'gamma':0,'nthread':4,
 'scale_pos_weight':1,'reg_alpha':0.002,'objective': 'binary:logistic', 'max_depth':3, 'min_child_weight':9,'cv':20}

    final_gb = xgb.train(our_params, xgdmat, num_boost_round=6000)
    y_pred = final_gb.predict(xgdmat)

    print('AuC score on training data:', roc_auc_score(Y_train, y_pred))
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0

    print(accuracy_score(y_pred, Y_train))

    testdmat = xgb.DMatrix(X_test)
    y_pred = final_gb.predict(testdmat)

    devdmat = xgb.DMatrix(X_dev)
    y_preddev = final_gb.predict(devdmat)

    write_to_csv.writeToCSV('predBoost.csv', y_pred)
    return y_preddev

if __name__ == "__main__":
    X_train, Y_train, X_dev, Y_dev = pre_process.preprocessData('train.csv')
    X_test, Y_test = load_test_data.loadTestData('test.csv')

    for i in range(len(Y_train)):
        if Y_train[i] == -1:
            Y_train[i] = 0

    #parameterTuning(X_train,Y_train)
    devPred = model(X_train,Y_train,X_test,X_dev)
    roc_curves.plotROCCuves(Y_dev, devPred, 'boost', 'XGBoost (Dev) - ')


