import os
import csv
import requests
import logging
import pandas as pd
import json
import time
from label_decoders import *

def read_data():
	print(time.strftime("%H:%M:%S") + '> load data ...')
	config = json.load(open('settings.json'))
	# set the path of the raw data
	train_file_path = config['train']
	test_file_path = config['test']
	# read the data with all default parameters
	train_df = pd.read_csv(train_file_path, index_col='Id')
	test_df = pd.read_csv(test_file_path, index_col='Id')
	test_df['Response'] = 0
	df = pd.concat((train_df, test_df), axis=0)
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



def process_data(model_prefix, feature_count=13):
	print(time.strftime("%H:%M:%S") + '> process data ...')
	df = read_data()
	df = process_feature(df)

	# split train and test
	train_ohd = df[df['Response'] > 0].copy()
	test_ohd = df[df['Response'] < 1].copy()

	features = train_ohd.columns.tolist()
	#features = [x.replace('=','_') for x in features]
	features = [x.replace('_', 'i') for x in features]
	train_ohd.columns = features
	features_t = test_ohd.columns.tolist()
	#features_t = [x.replace('=','i') for x in features_t]
	features_t = [x.replace('_', 'i') for x in features_t]
	test_ohd.columns = features_t

	#features.remove("Id")
	features.remove("Response")

	#https://datascience.stackexchange.com/questions/9255/creating-new-columns-by-iterating-over-rows-in-pandas-dataframe
	for i in range(1, feature_count+1):
		col_name = model_prefix + str(i)
		train_ohd[col_name] = [0]*train_ohd.shape[0]

	ld = [labels_decoder1, labels_decoder2,labels_decoder3,labels_decoder4,labels_decoder5,labels_decoder6,labels_decoder7, labels_decoder8]
	if (feature_count ==13):
		ld.append([labels_decoder9, labels_decoder10,labels_decoder11,labels_decoder12,labels_decoder13])

	l = train_ohd.shape[0]
	ind_list = [(list(range(0, l//10)), [x for x in range(0,l) if x not in list(range(0,l//10))]), 
				(list(range(l//10, l//10*2)), [x for x in range(0,l) if x not in list(range(l//10,l//10*2))]),
				(list(range(l//10*2, l//10*3)), [x for x in range(0,l) if x not in list(range(l//10*2,l//10*3))]),
				(list(range(l//10*3, l//10*4)), [x for x in range(0,l) if x not in list(range(l//10*3,l//10*4))]),
				(list(range(l//10*4, l//10*5)), [x for x in range(0,l) if x not in list(range(l//10*4,l//10*5))]),
				(list(range(l//10*5, l//10*6)), [x for x in range(0,l) if x not in list(range(l//10*5,l//10*6))]),
				(list(range(l//10*6, l//10*7)), [x for x in range(0,l) if x not in list(range(l//10*6,l//10*7))]),
				(list(range(l//10*7, l//10*8)), [x for x in range(0,l) if x not in list(range(l//10*7,l//10*8))]),
				(list(range(l//10*8, l//10*9)), [x for x in range(0,l) if x not in list(range(l//10*8,l//10*9))]),
				(list(range(l//10*9, l)), [x for x in range(0,l) if x not in list(range(l//10*9,l))])]
	return train_ohd, test_ohd,ind_list,ld,features
