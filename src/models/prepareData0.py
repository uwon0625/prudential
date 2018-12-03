import os
import csv
from dotenv import find_dotenv, load_dotenv
import requests
import logging
import pandas as pd 
import json

def init_env():
    # find .env automatically by walking up directories until it's found
    dotenv_path = find_dotenv()
    # load up the entries as environment variables
    load_dotenv(dotenv_path)
    # payload for login to kaggle
    payload = {
        '__RequestVerificationToken': '',
        'action': 'login',
        'username': os.environ.get("KAGGLE_USERNAME"),
        'password': os.environ.get("KAGGLE_PASSWORD"),
        'rememberme': 'false'
    }
    return payload

def extract_data(url, file_path, payload):
    with requests.Session() as c:
        c.post('https://www.kaggle.com/account/login', data=payload)
        with open(file_path, 'w') as handle:
            response = c.get(url, stream=True)
            handle.write(response.text)

def read_data():
    # set the path of the raw data
    raw_data_path = os.path.join(os.path.pardir,'data','raw')
    train_file_path = os.path.join(raw_data_path, 'train.csv')
    test_file_path = os.path.join(raw_data_path, 'test.csv')
    # read the data with all default parameters
    train_df = pd.read_csv(train_file_path, index_col='Id')
    test_df = pd.read_csv(test_file_path, index_col='Id')
    test_df['Response'] = 0
    df = pd.concat((train_df, test_df), axis=0)
    return df

def process_data(df):
    # catogorical features    
    df['Product_Info_2_char'] = df.Product_Info_2.str[0]
    df['Product_Info_2_num'] = df.Product_Info_2.str[1]
    df['Product_Info_2'] = pd.factorize(df['Product_Info_2'])[0]
    df['Product_Info_2_char'] = pd.factorize(df['Product_Info_2_char'])[0]
    df['Product_Info_2_num'] = pd.factorize(df['Product_Info_2_num'])[0]

    #from important features
    df['BMI_Age'] = df['BMI'] * df['Ins_Age']

    #combine medical keywords
    med_keyword_columns = df.columns[df.columns.str.startswith('Medical_Keyword_')]
    df['Med_Keywords_Count'] = df[med_keyword_columns].sum(axis=1)

    df.apply(lambda x: sum(x.isnull()),1)
    df['Response'] = df['Response'].astype(int)    
   
    #null values
    df['countna'] = df.apply(lambda x: sum(x.isnull()),1)
    df.fillna(-1, inplace=True)
    return df


# get logger
# logger = logging.getLogger(__name__)
# logger.info('getting raw data')

#need to use Kaggle API to download datasets
# urls
# train_url = 'https://www.kaggle.com/c/prudential-life-insurance-assessment/download/train.csv'
# test_url = 'https://www.kaggle.com/c/prudential-life-insurance-assessment/download/test.csv'
# file paths
# raw_data_path = os.path.join(os.path.pardir,'data','raw')
# train_data_path = os.path.join(raw_data_path, 'train.csv')
# test_data_path = os.path.join(raw_data_path, 'test.csv')

# payload = init_env()
# # extract data
# extract_data(train_url,train_data_path,payload)
# extract_data(test_url,test_data_path,payload)
# logger.info('downloaded raw training and test data')


# get data: train and test
all_data = read_data()
all_data = process_data(all_data)

# split train and test
train_ohd = all_data[all_data['Response']>0].copy()
test_ohd = all_data[all_data['Response']<1].copy()

features=train_ohd.columns.tolist()
#features = [x.replace('=','_') for x in features]
features = [x.replace('_','i') for x in features]
train_ohd.columns = features
features_t=test_ohd.columns.tolist()
#features_t = [x.replace('=','i') for x in features_t]
features_t = [x.replace('_','i') for x in features_t]
test_ohd.columns = features_t

#features.remove("Id")
features.remove("Response")

train_ohd['lr1'] = [0]*train_ohd.shape[0]
train_ohd['lr2'] = [0]*train_ohd.shape[0]
train_ohd['lr3'] = [0]*train_ohd.shape[0]
train_ohd['lr4'] = [0]*train_ohd.shape[0]
train_ohd['lr5'] = [0]*train_ohd.shape[0]
train_ohd['lr6'] = [0]*train_ohd.shape[0]
train_ohd['lr7'] = [0]*train_ohd.shape[0]
train_ohd['lr8'] = [0]*train_ohd.shape[0]
train_ohd['lr9'] = [0]*train_ohd.shape[0]
train_ohd['lr10'] = [0]*train_ohd.shape[0]
train_ohd['lr11'] = [0]*train_ohd.shape[0]
train_ohd['lr12'] = [0]*train_ohd.shape[0]
train_ohd['lr13'] = [0]*train_ohd.shape[0]


l = train_ohd.shape[0]
ind_list = [(list(range(0,l//10)), [x for x in range(0,l) if x not in list(range(0,l//10))]), 
            (list(range(l//10,l//10*2)), [x for x in range(0,l) if x not in list(range(l//10,l//10*2))]),
            (list(range(l//10*2,l//10*3)), [x for x in range(0,l) if x not in list(range(l//10*2,l//10*3))]),
            (list(range(l//10*3,l//10*4)), [x for x in range(0,l) if x not in list(range(l//10*3,l//10*4))]),
            (list(range(l//10*4,l//10*5)), [x for x in range(0,l) if x not in list(range(l//10*4,l//10*5))]),
            (list(range(l//10*5,l//10*6)), [x for x in range(0,l) if x not in list(range(l//10*5,l//10*6))]),
            (list(range(l//10*6,l//10*7)), [x for x in range(0,l) if x not in list(range(l//10*6,l//10*7))]),
            (list(range(l//10*7,l//10*8)), [x for x in range(0,l) if x not in list(range(l//10*7,l//10*8))]),
            (list(range(l//10*8,l//10*9)), [x for x in range(0,l) if x not in list(range(l//10*8,l//10*9))]),
            (list(range(l//10*9,l)), [x for x in range(0,l) if x not in list(range(l//10*9,l))])]


config = json.load(open('settings.json'))
all_data.to_csv(config['staged_full'],index=0)

with open(config['staged_index'],'w') as f:
    writer = csv.writer(f,delimiter = ",")
    writer.write(ind_list)    