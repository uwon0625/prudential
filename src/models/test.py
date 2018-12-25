import pandas as pd 

file_path = '../submissions/submission.csv'
df = pd.read_csv(file_path)
df['Response']+=1
df.to_csv(file_path.replace('.csv','_adj.csv'), index=False)
