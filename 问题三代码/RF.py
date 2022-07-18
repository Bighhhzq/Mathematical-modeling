from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

gr = pd.read_csv('./clean451.csv', index_col=0, encoding='gb18030')
# print(gr)
feature = ['ATSm2', 'ATSm3', 'BCUTc-1h', 'SCH-6', 'VC-5', 'SP-1', 'ECCEN', 'SHBd',
           'SsCH3', 'SaaO', 'minHBa', 'minaaO', 'maxaaO', 'hmin',
           'LipoaffinityIndex', 'ETA_Beta', 'ETA_Beta_s', 'ETA_Eta_R', 'ETA_Eta_F',
           'ETA_Eta_R_L', 'FMF', 'MDEC-12', 'MDEC-23', 'MLFER_S', 'MLFER_E',
           'MLFER_L', 'TopoPSA', 'MW', 'WTPT-1', 'WPATH']

feature_df = gr[feature]

x = feature_df.values
# x = gr.iloc[:, 6:].values

print(x)
# y_var = ['Caco-2', 'CYP3A4', 'hERG', 'hERG', 'MN']
y_var = ['Caco-2']
for v in y_var:
    y = gr[v]

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, train_size=0.7)
    print('训练集和测试集 shape', x_train.shape, y_train.shape, x_test.shape, y_test.shape)

clf = RandomForestClassifier()
clf = clf.fit(x_train, y_train)

y_predicted = clf.predict(x_test)
accuracy = np.mean(y_predicted == y_test) * 100
print("y_test\n", y_test)
print("y_predicted\n", y_predicted)
print("accuracy:", accuracy)
