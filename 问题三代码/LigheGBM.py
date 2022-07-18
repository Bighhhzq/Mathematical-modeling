import json
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import metrics

try:
    import cPickle as pickle
except BaseException:
    import pickle

gr = pd.read_csv('./clean451.csv', index_col=0, encoding='gb18030')
feature = ['ATSm2', 'ATSm3', 'BCUTc-1h', 'SCH-6', 'VC-5', 'SP-1', 'ECCEN', 'SHBd',
           'SsCH3', 'SaaO', 'minHBa', 'minaaO', 'maxaaO', 'hmin',
           'LipoaffinityIndex', 'ETA_Beta', 'ETA_Beta_s', 'ETA_Eta_R', 'ETA_Eta_F',
           'ETA_Eta_R_L', 'FMF', 'MDEC-12', 'MDEC-23', 'MLFER_S', 'MLFER_E',
           'MLFER_L', 'TopoPSA', 'MW', 'WTPT-1', 'WPATH']

feature_df = gr[feature]
x = feature_df.values

print(x)
# y_var = ['Caco-2', 'CYP3A4', 'hERG', 'hERG', 'MN']
y_var = ['Caco-2']
for v in y_var:
    y = gr[v]

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, train_size=0.7)
    print('训练集和测试集 shape', x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 7,
        'metric': 'multi_error',
        'num_leaves': 120,
        'min_data_in_leaf': 100,
        'learning_rate': 0.06,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.4,
        'lambda_l2': 0.5,
        'min_gain_to_split': 0.2,
        'verbose': -1,
    }
    print('Training...')
    trn_data = lgb.Dataset(x_train, y_train)
    val_data = lgb.Dataset(x_test, y_test)
    clf = lgb.train(params,
                    trn_data,
                    num_boost_round=1000,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=100,
                    early_stopping_rounds=100)
    print('Predicting...')
    y_prob = clf.predict(x_test, num_iteration=clf.best_iteration)
    y_pred = [list(x).index(max(x)) for x in y_prob]
    print("AUC score: {:<8.5f}".format(metrics.accuracy_score(y_pred, y_test)))
