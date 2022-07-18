import pandas as pd
from minepy import MINE
import numpy as np

Set = pd.read_csv('./Clean_Data.csv' , encoding='gb18030', index_col=0)
y_set = Set['pIC50']
feature_set = Set.drop(['SMILES', 'pIC50'], axis=1)

y = np.array(y_set.tolist())
Clean_var = feature_set.columns.tolist()

mine = MINE(alpha=0.6, c=15)
mic = []
for i in Clean_var:
    x = np.array(feature_set[i].tolist())
    mine.compute_score(x, y)
    m = mine.mic()
    mic.append(m)
print(mic)

max_index = pd.Series(mic).sort_values().index[:40]
mic_slect_var = [x for x in Clean_var if Clean_var.index(x) in max_index]

print(Clean_var)
print(max_index)
print(mic_slect_var)