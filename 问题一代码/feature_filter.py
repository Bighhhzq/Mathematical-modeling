import pandas as pd
from 问题一代码.remove_vars import clean_zi_var1
from 问题一代码.dcor import dcor

clean_set0 = pd.read_csv('./clean451.csv', index_col=0, encoding='gb18030')
print(clean_set0)
clean_var_set = clean_set0.drop('SMILES', axis=1)

rps = clean_var_set.corr(method='pearson')
df_ps = rps.iloc[0]

ps_list = rps['pIC50'].tolist()
ps_list.remove(1.0)

feature_set = clean_set0.drop(['SMILES', 'pIC50'], axis=1)
Clean_var = feature_set.columns.tolist()

print("ps_list:",ps_list)
print(len(ps_list))

rspm = clean_var_set.corr(method='spearman')
df_spm = rspm.iloc[0]

spm_list = rspm['pIC50'].tolist()
spm_list.remove(1.0)

pic50 = clean_var_set['pIC50'].tolist()
delete2 = []
dcor_list = []
for a in clean_zi_var1:
     i = clean_var_set[a].tolist()
     d = dcor(pic50, i)
     dcor_list.append(d)
     if abs(df_ps[a])<0.2 and abs(df_spm[a])<0.2 and d<0.2:
         delete2.append(a)
dcor_list.remove(1.0)
print("dcor_list:", dcor_list)


df_clean2 = clean_set0.drop(delete2, axis=1)
print(clean_var_set)
print(df_clean2)
Clean_data = df_clean2
print(Clean_data)
Clean_data.to_csv('.\Clean_Data.csv.csv')

