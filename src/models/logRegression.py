import pandas as pd 
from sklearn.linear_model import LogisticRegression
import json
import csv
from sklearn import metrics
from label_decoders import *
from prepareData import *
import time

config = json.load(open('settings.json'))

train_ohd,test_ohd,ind_list,ld,features = process_data(model_prefix='lr', feature_count=13)

print(time.strftime("%H:%M:%S") + '> start training LogisticRegression...')

i = 0
for l in ld:
    i = i + 1    
    
    for j in range(10):
        
        X_1, X_2 = ind_list[j][1], ind_list[j][0]
        y_1, y_2 = train_ohd.iloc[X_1]['Response'], train_ohd.iloc[X_2]['Response']
        
        lr = LogisticRegression(random_state=1)
        lr.fit(train_ohd[features].iloc[X_1],l(y_1))
        train_ohd.iloc[X_2]['lr%s' % (i)] = lr.predict_proba(train_ohd[features].iloc[X_2]).T[1]


train_ohd.to_csv(config['train_lr'],index=0)

y = train_ohd['Response']

i = 0
for l in ld:
    i = i + 1    

###1
    lr = LogisticRegression(random_state=1)
    lr.fit(train_ohd[features],l(y)), i
    test_ohd['lr%s' % (i)] = lr.predict_proba(test_ohd[features]).T[1]

test_ohd.to_csv(config['test_lr'],index=0)
print(time.strftime("%H:%M:%S") + '> finish training LogisticRegression...')