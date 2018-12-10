import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import json
from label_decoders import *
from prepareData import *
import time

config = json.load(open('settings.json'))

train_ohd,test_ohd,ind_list,ld,features = process_data(model_prefix='rf', feature_count=13)

print(time.strftime("%H:%M:%S") + '> start training RandomForestClassifier...')
i = 0
for l in ld:
    i = i + 1    
    print(time.strftime("%H:%M:%S") + ' train data ->' + str(i))
    for j in range(10):
        
        X_1, X_2 = ind_list[j][1], ind_list[j][0]
        y_1, y_2 = train_ohd.iloc[X_1]['Response'], train_ohd.iloc[X_2]['Response']
        
        rf = RandomForestClassifier(n_estimators=500, random_state=1)
        rf.fit(train_ohd[features].iloc[X_1],l(y_1))
        train_ohd['rf%s' % (i)].iloc[X_2] = rf.predict_proba(train_ohd[features].iloc[X_2]).T[1]


train_ohd.to_csv(config['train_rf'],index=0)

y = train_ohd['Response']

i = 0
for l in ld:
    i = i + 1    
    print(time.strftime("%H:%M:%S") + ' test data ->' + str(i))
###1
    rf = RandomForestClassifier(n_estimators=500, random_state=1)
    rf.fit(train_ohd[features],l(y))
    test_ohd['rf%s' % (i)] = rf.predict_proba(test_ohd[features]).T[1]

test_ohd.to_csv(config['test_rf'],index=0)
print(time.strftime("%H:%M:%S") + '> finish training RandomForestClassifier.')
