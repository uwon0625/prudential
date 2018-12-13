import pandas as pd 

train_file_path = '../features/train_lr2.csv'
df = pd.read_csv(train_file_path)
df.sort_index(axis=1, inplace=True)
df.to_csv(train_file_path.replace('.csv','_32.csv'))
