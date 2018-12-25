import pandas as pd
import numpy as np
import json
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import model_selection, metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection  import GridSearchCV
import time

#load data
config = json.load(open('settings.json'))
train = pd.read_csv(config['train_modified'])
test = pd.read_csv(config['test_modified'])
num_classes = 8

from sklearn.model_selection import train_test_split
train_part, test_part = train_test_split(train, test_size=0.2)

target='Response'
IDcol = 'Id'

predictors = [x for x in train_part.columns if x not in [target, IDcol]]

X=train_part[predictors]
y=train_part[target]

#https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
def modelfit0(alg,dtrain,dtest,predictors,useTrainCV=True,cv_folds=5,early_stopping_rounds=100):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgbtrain = xgb.DMatrix(dtrain[predictors],label=dtrain[target])
        cvresult = xgb.cv( params=xgb_param,dtrain=xgbtrain,num_boost_round = alg.get_params()['n_estimators'],
                          nfold = cv_folds,metrics='mlogloss',early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    alg.fit(dtrain[predictors],dtrain['Response'],eval_metric='mlogloss')#'auc'
    print(time.strftime("%H:%M:%S") + 'finish fit train.')

    dtrain_predictions = alg.predict(dtrain[predictors])
    print(time.strftime("%H:%M:%S") + 'finish predict train.')
    #dtrain_predictprob = alg.predict_proba(dtrain[predictors])[:,1]
    print ('\nModel Report: n_estimators--' + str(alg.n_estimators))
    print ('Accuracy:%.4g'%metrics.accuracy_score(dtrain['Response'],dtrain_predictions))

    dtest_predictions = alg.predict(dtest[predictors])
    dtest_predictions['Response']+=1 #restore to risks 1-8
    print(time.strftime("%H:%M:%S") + 'finish predict test.')

    result = pd.DataFrame({"Id": dtest['Id'].values, "Response": dtest_predictions})
    result.to_csv(config['submission'], index=False)

train_part, test_part = train_test_split(train, test_size=0.2)
X1=train_part
X1['Response'] -=1 #reduce to fit [0,classes)

print(time.strftime("%H:%M:%S") + '> fit model and predict...')
xgbc = xgb.XGBClassifier(objective='multi:softprob', num_class=num_classes, 
                         learning_rate=0.01, 
                         n_estimators = 400,
                         max_depth=4,min_child_weight=2,gamma=0,reg_alpha=1,
                         subsample=0.9, colsample_bytree=0.5, 
                         early_stopping_rounds=100, seed=0)
#modelfit0(xgbc,train_part,test,predictors)

print(time.strftime("%H:%M:%S") + '> fit model and predict.')
