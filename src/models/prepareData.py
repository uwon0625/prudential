import os
import csv
import requests
import logging
import pandas as pd
import json
import time
from label_decoders import *
from dotenv import find_dotenv,load_dotenv

config = json.load(open('settings.json'))

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

#process changed: need to use Kaggle API to download datasets
def extract_data(url, file_path, payload):
    with requests.Session() as c:
        c.post('https://www.kaggle.com/account/login', data=payload)
        with open(file_path, 'w') as handle:
            response = c.get(url, stream=True)
            handle.write(response.text)

def read_data():
	print(time.strftime("%H:%M:%S") + '> load data ...')
	config = json.load(open('settings.json'))
	# set the path of the raw data
	train_file_path = config['train']
	test_file_path = config['test']
	# read the data with all default parameters
	train_df = pd.read_csv(train_file_path)
	test_df = pd.read_csv(test_file_path)
	test_df['Response'] = 0
	df = train_df.append(test_df)
	return df

def process_feature(df):
	print(time.strftime("%H:%M:%S") + '> process feature ...')
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

	df.apply(lambda x: sum(x.isnull()), 1)
	df['Response'] = df['Response'].astype(int)

    #null values
	df['countna'] = df.apply(lambda x: sum(x.isnull()), 1)
	df.fillna(-1, inplace=True)
	return df

def save_staged_data():
	df = read_data()
	df = process_feature(df)

	# split train and test
	train_ohd = df[df['Response'] > 0].copy()
	test_ohd = df[df['Response'] < 1].copy()

	features = train_ohd.columns.tolist()
	features = [x.replace('=','_') for x in features]
	features = [x.replace('_', 'i') for x in features]
	train_ohd.columns = features
	features_t = test_ohd.columns.tolist()
	features_t = [x.replace('=','i') for x in features_t]
	features_t = [x.replace('_', 'i') for x in features_t]
	test_ohd.columns = features_t

	train_ohd.sort_index(axis=1, inplace=True)
	train_ohd.to_csv(config['train_modified'],index=False)
	test_ohd.sort_index(axis=1, inplace=True)
	test_ohd.to_csv(config['test_modified'],index=False)

def process_data(model_prefix, feature_count=13, load_staged_data = True):
	print(time.strftime("%H:%M:%S") + '> process data ...')

	if (load_staged_data == False):
		save_staged_data()

	train_ohd = pd.read_csv(config['train_modified'])
	features = train_ohd.columns.tolist()
	test_ohd = pd.read_csv(config['test_modified'])
	features_t = test_ohd.columns.tolist()

	features.remove("Id")
	features.remove("Response")

	#https://datascience.stackexchange.com/questions/9255/creating-new-columns-by-iterating-over-rows-in-pandas-dataframe
	for i in range(1, feature_count+1):
		col_name = model_prefix + str(i)
		train_ohd[col_name] = [0]*train_ohd.shape[0]

	ld = [labels_decoder1, labels_decoder2,labels_decoder3,labels_decoder4,labels_decoder5,labels_decoder6,labels_decoder7, labels_decoder8]
	if (feature_count ==13):
		ld.extend([labels_decoder9, labels_decoder10,labels_decoder11,labels_decoder12,labels_decoder13])

	l = train_ohd.shape[0]

	##https://stackoverflow.com/questions/12319025/filters-in-python3
	ind_list = [(range(0,l//10), list(filter(lambda x: x not in range(0,l//10), range(0,l)))), 
            (range(l//10,l//10*2), list(filter(lambda x: x not in range(l//10,l//10*2), range(0,l)))),
            (range(l//10*2,l//10*3), list(filter(lambda x: x not in range(l//10*2,l//10*3), range(0,l)))),
            (range(l//10*3,l//10*4), list(filter(lambda x: x not in range(l//10*3,l//10*4), range(0,l)))),
            (range(l//10*4,l//10*5), list(filter(lambda x: x not in range(l//10*4,l//10*5), range(0,l)))),
            (range(l//10*5,l//10*6), list(filter(lambda x: x not in range(l//10*5,l//10*6), range(0,l)))),
            (range(l//10*6,l//10*7), list(filter(lambda x: x not in range(l//10*6,l//10*7), range(0,l)))),
            (range(l//10*7,l//10*8), list(filter(lambda x: x not in range(l//10*7,l//10*8), range(0,l)))),
            (range(l//10*8,l//10*9), list(filter(lambda x: x not in range(l//10*8,l//10*9), range(0,l)))),
            (range(l//10*9,l), list(filter(lambda x: x not in range(l//10*9,l), range(0,l))))]

	return train_ohd, test_ohd,ind_list,ld,features
