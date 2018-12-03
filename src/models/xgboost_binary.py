import pandas as pd 
import xgboost as xgb
import json
from label_decoders import *
from prepareData import *

config = json.load(open('settings.json'))

train_ohd,test_ohd,ind_list,ld,features = process_data(model_prefix='xgb', feature_count=13)


param = {'max_depth' : 4, 
         'eta' : 0.01, 
         'silent' : 1, 
         'min_child_weight' : 1, 
         'subsample' : 0.5,
         'early_stopping_rounds' : 100,
         'objective'   : 'binary:logistic',
         'eval_metric': 'auc',
         'colsample_bytree':0.3,
         'seed' : 0}

num_round=7000

print('start training xgboost...')

i = 0
for l in ld:
    i = i + 1    
    
    for j in range(10):
        
        X_1, X_2 = ind_list[j][1], ind_list[j][0]
        y_1, y_2 = train_ohd.iloc[X_1]['Response'], train_ohd.iloc[X_2]['Response']
        
        dtrain=xgb.DMatrix(train_ohd.iloc[X_1][features],label=l(y_1),missing=float('nan'))
        dval=xgb.DMatrix(train_ohd.iloc[X_2][features],label=l(y_2),missing=float('nan'))
        
#        watchlist  = [(dtrain,'train'), (dval,'valid')]
        
        bst = xgb.train(param, dtrain, num_round)
        train_ohd['xgb%i' % (i)].iloc[X_2] = bst.predict(dval)


train_ohd.to_csv(config['train_xgb'],index=0)

y = train_ohd['Response']

i = 0
for l in ld:
    i = i + 1    

###1
    dtrain=xgb.DMatrix(train_ohd[features],label=l(y),missing=float('nan'))
    dtest=xgb.DMatrix(test_ohd[features],missing=float('nan'))
    
#    watchlist  = [(dtrain,'train')]
    
    bst = xgb.train(param, dtrain, num_round)
    test_ohd['xgb%s' % (i)] = bst.predict(dtest)

test_ohd.to_csv(config['test_xgb'],index=0)
print('finish training xgboost...')
