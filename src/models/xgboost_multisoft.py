import pandas as pd 
import xgboost as xgb
import json
from prepareData import *
import time

config = json.load(open('settings.json'))

train_ohd,test_ohd,ind_list,ld,features = process_data(model_prefix='pr', feature_count=8)

param = {'max_depth' : 4, 
         'eta' : 0.01, 
         'silent' : 1, 
         'min_child_weight' : 10, 
         'subsample' : 0.5,
         'early_stopping_rounds' : 100,
         'objective'   : 'multi:softprob',
         'num_class' : 8,
         'colsample_bytree' : 0.3,
         'seed' : 0}

num_round=7000

print(time.strftime("%H:%M:%S") + '> start training xgboost multisift-8...')
   
for j in range(10):
    print(time.strftime("%H:%M:%S") + ' train data ->' + str(j))
    X_1, X_2 = ind_list[j][1], ind_list[j][0]
    y_1, y_2 = train_ohd.iloc[X_1]['Response'] - 1, train_ohd.iloc[X_2]['Response'] - 1
    
    dtrain=xgb.DMatrix(train_ohd.iloc[X_1][features],y_1,missing=float('nan'))
    dval=xgb.DMatrix(train_ohd.iloc[X_2][features],y_2,missing=float('nan'))
    
#    watchlist  = [(dtrain,'train'), (dval,'valid')]
    
    bst = xgb.train(param, dtrain, num_round)
    for k in range(1,9):
        train_ohd['pr%s' % (k)].iloc[X_2] = bst.predict(dval).T[k-1]

train_ohd.sort_index(axis=1, inplace=True)
train_ohd.to_csv(config['train_p1'],index=False)

###
y = train_ohd['Response'] - 1
dtrain=xgb.DMatrix(train_ohd[features],y,missing=float('nan'))
dtest=xgb.DMatrix(test_ohd[features],missing=float('nan'))

#watchlist  = [(dtrain,'train')]

bst = xgb.train(param, dtrain, num_round)
for k in range(1,9):
	print(time.strftime("%H:%M:%S") + ' test data ->' + str(i))
    test_ohd['pr%s' % (k)] = bst.predict(dtest).T[k-1]

test_ohd.sort_index(axis=1, inplace=True)
test_ohd.to_csv(config['test_p1'],index=False)
print(time.strftime("%H:%M:%S") + '> finish training xgboost multisift-8.')
