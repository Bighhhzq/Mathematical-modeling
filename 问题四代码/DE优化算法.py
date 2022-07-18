from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline
from xgboost.sklearn import XGBClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


gr = pd.read_csv('E:\\test\jianmo\优化\clean451.csv', index_col=0, encoding='gb18030')
# gr = gr.drop('SMILES', axis=1)
# x = gr.iloc[:, 6:].values
Feature_0 = ['C1SP2', 'SsOH', 'minHBa', 'maxssO', 'ALogP',
              'ATSc1', 'ATSc3', 'ATSc5', 'BCUTc-1l', 'BCUTc-1h',
              'BCUTp-1h', 'mindssC', 'minsssN', 'hmin', 'LipoaffinityIndex',
              'MAXDP2', 'ETA_BetaP_s', 'nHBAcc', 'nBondsD2', 'nsssCH']

Feature_1 = ['ATSm2', 'BCUTc-1h', 'SCH-6', 'VC-5', 'SHBd',
             'SsCH3', 'SaaO', 'minHBa', 'hmin', 'LipoaffinityIndex',
             'FMF', 'MDEC-23', 'MLFER_S', 'WPATH']

Feature_2 = ['apol', 'ATSc1', 'ATSm3', 'SCH-6', 'VCH-7',
             'SP-6', 'SHBd', 'SHsOH', 'SHaaCH', 'minHBa',
             'maxsOH', 'ETA_dEpsilon_D', 'ETA_Shape_P', 'ETA_Shape_Y', 'ETA_BetaP_s',
             'ETA_dBetaP']

Feature_3 = ['ATSc2', 'BCUTc-1l', 'BCUTc-1h', 'BCUTp-1h', 'SHBd',
             'SHother', 'SsOH', 'minHBd', 'minHBa','minaaCH',
             'minaasC', 'maxHBd', 'maxwHBa', 'maxHBint8','maxHsOH',
             'hmin', 'LipoaffinityIndex','ETA_dEpsilon_B', 'ETA_Shape_Y','ETA_EtaP_F',
             'ETA_Eta_R_L', 'MDEO-11', 'WTPT-4', 'minssCH2']

Feature_4 = ['ATSc2', 'BCUTc-1l', 'BCUTp-1l', 'VCH-6', 'SC-5',
             'SPC-6', 'VP-3', 'SHsOH', 'SdO', 'minHBa',
             'minHsOH', 'maxHother', 'maxdO', 'hmin', 'MAXDP2',
             'ETA_dEpsilon_B', 'ETA_Shape_Y', 'ETA_EtaP_F_L','MDEC-23', 'MLFER_A',
             'TopoPSA', 'WTPT-2', 'WTPT-4']

Feature_5 = ['nN', 'ATSc2', 'SCH-7', 'VPC-5', 'SP-6',
             'SHaaCH', 'SssCH2', 'SsssCH', 'SssO', 'minHBa',
             'mindssC', 'maxsCH3', 'maxsssCH', 'maxssO', 'hmin',
             'ETA_dEpsilon_B', 'ETA_dEpsilon_C', 'ETA_Shape_Y','ETA_BetaP', 'ETA_BetaP_s',
             'ETA_EtaP_F', 'ETA_EtaP_B_RC', 'FMF', 'nHBAcc', 'MLFER_E', 'WTPT-4'] #26


featuren_all = ['nN', 'ATSc2', 'SCH-7', 'VPC-5', 'SP-6',
             'SHaaCH', 'SssCH2', 'SsssCH', 'SssO', 'minHBa',
             'mindssC', 'maxsCH3', 'maxsssCH', 'maxssO', 'hmin',
             'ETA_dEpsilon_B', 'ETA_dEpsilon_C', 'ETA_Shape_Y','ETA_BetaP', 'ETA_BetaP_s',
             'ETA_EtaP_F', 'ETA_EtaP_B_RC', 'FMF', 'nHBAcc', 'MLFER_E',
                'WTPT-4','BCUTc-1l','BCUTp-1l','VCH-6','SC-5',
                'SPC-6','VP-3','SHsOH','SdO','minHsOH',
                'maxHother','maxdO','MAXDP2','ETA_EtaP_F_L','MDEC-23',
                'MLFER_A','TopoPSA','WTPT-2','BCUTc-1h','BCUTp-1h',
                'SHBd','SHother','SsOH','minHBd','minaaCH',
                'minaasC','maxHBd','maxwHBa','maxHBint8','maxHsOH',
                'LipoaffinityIndex','ETA_Eta_R_L','MDEO-11','minssCH2','apol',
                'ATSc1','ATSm3','SCH-6','VCH-7','maxsOH',
                'ETA_dEpsilon_D','ETA_Shape_P','ETA_dBetaP','ATSm2','VC-5',
                'SsCH3','SaaO','MLFER_S','WPATH','C1SP2',
                'ALogP','ATSc3','ATSc5','minsssN','nBondsD2',
                'nsssCH']




Set = pd.read_csv('E:\\test\jianmo\优化\clean451.csv', index_col=0, encoding='gb18030')
print(Set)
scaler = StandardScaler()
Set.loc[:, Set.columns != 'SMILES'] = scaler.fit_transform(Set.loc[:, Set.columns != 'SMILES'])
Set_d = Set[featuren_all].copy()
print(Set_d)
print(Set_d.shape)
import numpy as np
print(type(Set_d))
print(np.max(Set_d))
print(np.min(Set_d))
max_list=np.max(Set_d).tolist()
min_list=np.min(Set_d).tolist()
print(max_list)
print(min_list)


################################# "MN"

feature_df = gr[Feature_5].copy()

x = feature_df.values

y_var = ['MN']
for v in y_var:
    y = gr[v]
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, train_size=0.8)
    print('训练集和测试集 shape', x_train.shape, y_train.shape, x_test.shape, y_test.shape)


rf = RandomForestClassifier()
xgboost = XGBClassifier(eval_metric=['logloss', 'auc', 'error'], use_label_encoder=False)
lgbm = LGBMClassifier()

# pipe1 = make_pipeline(StandardScaler(), clf_lr)
pipe6_1 = make_pipeline(StandardScaler(), rf)
pipe6_2 = make_pipeline(StandardScaler(), xgboost)
pipe6_3 = make_pipeline(StandardScaler(), lgbm)

models = [
          ('rf', pipe6_1),
          ('xgb', pipe6_2),
          ('lgbm', pipe6_3)
          ]

ensembel6_4 = VotingClassifier(estimators=models, voting='soft')

from sklearn.model_selection import cross_val_score

all_model = [pipe6_1, pipe6_2, pipe6_3, ensembel6_4]
clf_labels = ['RandomForestClassifier', 'XGBClassifier', "LGBMClassifier", 'Ensemble']

score = cross_val_score(estimator=pipe6_1, X=x_train, y=y_train, cv=10, scoring='roc_auc')
print("ElasticNet score: {:.4f}\n".format(score.mean(), score.std()), )
pipe6_1.fit(x_train, y_train)

score = cross_val_score(estimator=pipe6_2, X=x_train, y=y_train, cv=10, scoring='roc_auc')
print("ElasticNet score: {:.4f}\n".format(score.mean(), score.std()), )
pipe6_2.fit(x_train, y_train)

score = cross_val_score(estimator=pipe6_3, X=x_train, y=y_train, cv=10, scoring='roc_auc')
print("ElasticNet score: {:.4f}\n".format(score.mean(), score.std()), )
pipe6_3.fit(x_train, y_train)

score = cross_val_score(estimator=ensembel6_4, X=x_train, y=y_train, cv=10, scoring='roc_auc')
print("ElasticNet score: {:.4f}\n".format(score.mean(), score.std()), )
ensembel6_4.fit(x_train, y_train)

################################# "HOB"

feature_df = gr[Feature_4].copy()

x = feature_df.values

y_var = ['HOB']
for v in y_var:
    y = gr[v]
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, train_size=0.8)
    print('训练集和测试集 shape', x_train.shape, y_train.shape, x_test.shape, y_test.shape)

rf = RandomForestClassifier()
xgboost = XGBClassifier(eval_metric=['logloss', 'auc', 'error'], use_label_encoder=False)
lgbm = LGBMClassifier()

pipe5_1 = make_pipeline(StandardScaler(), rf)
pipe5_2 = make_pipeline(StandardScaler(), xgboost)
pipe5_3 = make_pipeline(StandardScaler(), lgbm)

models = [
          ('rf', pipe5_1),
          ('xgb', pipe5_2),
          ('lgbm', pipe5_3)
          ]

ensembel5_4 = VotingClassifier(estimators=models, voting='soft')

from sklearn.model_selection import cross_val_score

all_model = [pipe5_1, pipe5_2, pipe5_3, ensembel5_4]
clf_labels = ['RandomForestClassifier', 'XGBClassifier', "LGBMClassifier", 'Ensemble']

score = cross_val_score(estimator=pipe5_1, X=x_train, y=y_train, cv=10, scoring='roc_auc')
print("ElasticNet score: {:.4f}\n".format(score.mean(), score.std()), )
pipe5_1.fit(x_train, y_train)

score = cross_val_score(estimator=pipe5_2, X=x_train, y=y_train, cv=10, scoring='roc_auc')
print("ElasticNet score: {:.4f}\n".format(score.mean(), score.std()), )
pipe5_2.fit(x_train, y_train)

score = cross_val_score(estimator=pipe5_3, X=x_train, y=y_train, cv=10, scoring='roc_auc')
print("ElasticNet score: {:.4f}\n".format(score.mean(), score.std()), )
pipe5_3.fit(x_train, y_train)

score = cross_val_score(estimator=ensembel5_4, X=x_train, y=y_train, cv=10, scoring='roc_auc')
print("ElasticNet score: {:.4f}\n".format(score.mean(), score.std()), )
ensembel5_4.fit(x_train, y_train)

################################# "hERG"

feature_df = gr[Feature_3].copy()

x = feature_df.values

y_var = ['hERG']
for v in y_var:
    y = gr[v]
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, train_size=0.8)
    print('训练集和测试集 shape', x_train.shape, y_train.shape, x_test.shape, y_test.shape)

rf = RandomForestClassifier()
xgboost = XGBClassifier(eval_metric=['logloss', 'auc', 'error'], use_label_encoder=False)
lgbm = LGBMClassifier()

pipe4_1 = make_pipeline(StandardScaler(), rf)
pipe4_2 = make_pipeline(StandardScaler(), xgboost)
pipe4_3 = make_pipeline(StandardScaler(), lgbm)

models = [
          ('rf', pipe4_1),
          ('xgb', pipe4_2),
          ('lgbm', pipe4_3)
          ]

ensembel4_4 = VotingClassifier(estimators=models, voting='soft')

from sklearn.model_selection import cross_val_score

all_model = [pipe4_1, pipe4_2, pipe4_3, ensembel4_4]
clf_labels = ['RandomForestClassifier', 'XGBClassifier', "LGBMClassifier", 'Ensemble']

score = cross_val_score(estimator=pipe4_1, X=x_train, y=y_train, cv=10, scoring='roc_auc')
print("ElasticNet score: {:.4f}\n".format(score.mean(), score.std()), )
pipe4_1.fit(x_train, y_train)

score = cross_val_score(estimator=pipe4_2, X=x_train, y=y_train, cv=10, scoring='roc_auc')
print("ElasticNet score: {:.4f}\n".format(score.mean(), score.std()), )
pipe4_2.fit(x_train, y_train)

score = cross_val_score(estimator=pipe4_3, X=x_train, y=y_train, cv=10, scoring='roc_auc')
print("ElasticNet score: {:.4f}\n".format(score.mean(), score.std()), )
pipe4_3.fit(x_train, y_train)

score = cross_val_score(estimator=ensembel4_4, X=x_train, y=y_train, cv=10, scoring='roc_auc')
print("ElasticNet score: {:.4f}\n".format(score.mean(), score.std()), )
ensembel4_4.fit(x_train, y_train)


################################# "CYP3A4"

feature_df = gr[Feature_2].copy()

x = feature_df.values

y_var = ['CYP3A4']
for v in y_var:
    y = gr[v]
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, train_size=0.8)
    print('训练集和测试集 shape', x_train.shape, y_train.shape, x_test.shape, y_test.shape)

rf = RandomForestClassifier()
xgboost = XGBClassifier(eval_metric=['logloss', 'auc', 'error'], use_label_encoder=False)
lgbm = LGBMClassifier()

pipe3_1 = make_pipeline(StandardScaler(), rf)
pipe3_2 = make_pipeline(StandardScaler(), xgboost)
pipe3_3 = make_pipeline(StandardScaler(), lgbm)

models = [
          ('rf', pipe3_1),
          ('xgb', pipe3_2),
          ('lgbm', pipe3_3)
          ]

ensembel3_4 = VotingClassifier(estimators=models, voting='soft')

from sklearn.model_selection import cross_val_score

all_model = [pipe3_1, pipe3_2, pipe3_3, ensembel3_4]
clf_labels = ['RandomForestClassifier', 'XGBClassifier', "LGBMClassifier", 'Ensemble']

score = cross_val_score(estimator=pipe3_1, X=x_train, y=y_train, cv=10, scoring='roc_auc')
print("ElasticNet score: {:.4f}\n".format(score.mean(), score.std()), )
pipe3_1.fit(x_train, y_train)

score = cross_val_score(estimator=pipe3_2, X=x_train, y=y_train, cv=10, scoring='roc_auc')
print("ElasticNet score: {:.4f}\n".format(score.mean(), score.std()), )
pipe3_2.fit(x_train, y_train)

score = cross_val_score(estimator=pipe3_3, X=x_train, y=y_train, cv=10, scoring='roc_auc')
print("ElasticNet score: {:.4f}\n".format(score.mean(), score.std()), )
pipe3_3.fit(x_train, y_train)

score = cross_val_score(estimator=ensembel3_4, X=x_train, y=y_train, cv=10, scoring='roc_auc')
print("ElasticNet score: {:.4f}\n".format(score.mean(), score.std()), )
ensembel3_4.fit(x_train, y_train)

################################# "Caco-2"

feature_df = gr[Feature_1].copy()

x = feature_df.values

y_var = ['Caco-2']
for v in y_var:
    y = gr[v]
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, train_size=0.8)
    print('训练集和测试集 shape', x_train.shape, y_train.shape, x_test.shape, y_test.shape)

rf = RandomForestClassifier()
xgboost = XGBClassifier(eval_metric=['logloss', 'auc', 'error'], use_label_encoder=False)
lgbm = LGBMClassifier()

pipe2_1 = make_pipeline(StandardScaler(), rf)
pipe2_2 = make_pipeline(StandardScaler(), xgboost)
pipe2_3 = make_pipeline(StandardScaler(), lgbm)

models = [
          ('rf', pipe2_1),
          ('xgb', pipe2_2),
          ('lgbm', pipe2_3)
          ]

ensembel2_4 = VotingClassifier(estimators=models, voting='soft')

from sklearn.model_selection import cross_val_score

score = cross_val_score(estimator=pipe2_1, X=x_train, y=y_train, cv=10, scoring='roc_auc')
print("ElasticNet score: {:.4f}\n".format(score.mean(), score.std()), )
pipe2_1.fit(x_train, y_train)

score = cross_val_score(estimator=pipe2_2, X=x_train, y=y_train, cv=10, scoring='roc_auc')
print("ElasticNet score: {:.4f}\n".format(score.mean(), score.std()), )
pipe2_2.fit(x_train, y_train)

score = cross_val_score(estimator=pipe2_3, X=x_train, y=y_train, cv=10, scoring='roc_auc')
print("ElasticNet score: {:.4f}\n".format(score.mean(), score.std()), )
pipe2_3.fit(x_train, y_train)

score = cross_val_score(estimator=ensembel2_4, X=x_train, y=y_train, cv=10, scoring='roc_auc')
print("ElasticNet score: {:.4f}\n".format(score.mean(), score.std()), )
ensembel2_4.fit(x_train, y_train)

################################# "ERa"
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from mlxtend.regressor import StackingCVRegressor

Set = pd.read_csv('E:\\test\jianmo\预测\Clean_Data.csv', encoding='gb18030', index_col=0)
y_train = Set['pIC50'].copy()
Set.loc[:, Set.columns != 'SMILES'] = scaler.fit_transform(Set.loc[:, Set.columns != 'SMILES'])

x_train = Set[Feature_0].copy()
x_train = x_train.values
print('训练集和测试集 shape', x_train.shape, y_train.shape)

gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                max_depth=4, max_features='sqrt',
                                min_samples_leaf=15, min_samples_split=10,
                                loss='huber', random_state=42)

xgboost = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                       max_depth=3, min_child_weight=0,
                       gamma=0, subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:linear', nthread=-1,
                       scale_pos_weight=1, seed=27,
                       reg_alpha=0.00006)

stack_gen = StackingCVRegressor(regressors=( gbr, xgboost), meta_regressor=xgboost, use_features_in_secondary=True)

pipe1_1 = make_pipeline(StandardScaler(), stack_gen)

from sklearn.model_selection import cross_val_score

score = cross_val_score(estimator=pipe1_1, X=x_train, y=y_train, cv=10, scoring='neg_mean_squared_error')
print("ElasticNet score: {:.4f}\n".format(score.mean(), score.std()), )
pipe1_1.fit(x_train, y_train)

def schaffer(p):
    '''
    This function has plenty of local minimum, with strong shocks
    global minimum at (0,0) with value 0
    '''
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26\
        ,x27,x28,x29,x30,x31,x32,x33,x34,x35,x36,x37,x38,x39,x40,x41,x42,x43 \
        ,x44,x45,x46,x47,x48,x49,x50,x51,x52,x53,x54,x55,x56,x57,x58,x59\
        ,x60,x61,x62,x63,x64,x65,x66,x67,x68\
        ,x69,x70,x71,x72,x73,x74\
        ,x75,x76,x77,x78,x79,x80,x81= p

    feature1 = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26] # MN
    feature2 = [x2,x27,x28,x29,x30,x31,x32,x33,x34,x10,x35,x36,x37,x15,x38,x16,x18,x39,x40,x41,x42,x43,x26] # Hob
    feature3 = [x2,x27,x44,x45,x46,x47,x48,x49,x10,x50,x51,x52,x53,x54,x55,x15,x56,x16,x18,x21,x57,x58,x26,x59] #hERG
    feature4 = [x60,x61,x62,x63,x64,x5,x46,x33,x6,x10,x65,x66,x67,x18,x20,x68] #CYP3A4
    feature5 = [x69,x44,x63,x70,x46,x71,x72,x10,x15,x56,x23,x40,x73,x74]
    feature6 = [x75,x48,x10,x14,x76,x61,x77,x78,x27,x44,x45,x45,x79,x45,x56,x38,x20,x24,x80,x81]

    pipe6_1_score = pipe6_1.predict([feature1])
    pipe6_2_score = pipe6_2.predict([feature1])
    pipe6_3_score = pipe6_3.predict([feature1])
    pipe6_4_score = ensembel6_4.predict([feature1])
    score_mn =pipe6_1_score + pipe6_2_score + pipe6_3_score + pipe6_4_score

    pipe5_1_score = pipe5_1.predict([feature2])
    pipe5_2_score = pipe5_2.predict([feature2])
    pipe5_3_score = pipe5_3.predict([feature2])
    pipe5_4_score = ensembel5_4.predict([feature2])
    score_hob =pipe5_1_score + pipe5_2_score + pipe5_3_score + pipe5_4_score

    pipe4_1_score = pipe4_1.predict([feature3])
    pipe4_2_score = pipe4_2.predict([feature3])
    pipe4_3_score = pipe4_3.predict([feature3])
    pipe4_4_score = ensembel4_4.predict([feature3])
    score_hERG =pipe4_1_score + pipe4_2_score + pipe4_3_score + pipe4_4_score

    pipe3_1_score = pipe3_1.predict([feature4])
    pipe3_2_score = pipe3_2.predict([feature4])
    pipe3_3_score = pipe3_3.predict([feature4])
    pipe3_4_score = ensembel3_4.predict([feature4])
    score_CYP3A4 =pipe3_1_score + pipe3_2_score + pipe3_3_score + pipe3_4_score

    pipe2_1_score = pipe2_1.predict([feature5])
    pipe2_2_score = pipe2_2.predict([feature5])
    pipe2_3_score = pipe2_3.predict([feature5])
    pipe2_4_score = ensembel2_4.predict([feature5])
    score_Caco2 =pipe2_1_score + pipe2_2_score + pipe2_3_score + pipe2_4_score

    pipe1_1_score = pipe1_1.predict([feature6])
    score_ERa =pipe1_1_score

    print(score_ERa)
    print(score_Caco2)
    print(score_CYP3A4)
    print(score_hERG)
    print(score_hob)
    print(score_mn)

    print("总得分", -score_Caco2 + score_CYP3A4 + score_hERG - score_hob + score_mn - score_ERa)

    return -score_Caco2 + score_CYP3A4 + score_hERG - score_hob + score_mn - score_ERa


from 问题四优化.scikitopt.sko.DE import DE

de = DE(func=schaffer, n_dim=81, size_pop=100, max_iter=40, lb=min_list, ub=max_list)

# pso.run()
best_x, best_y = de.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)

import matplotlib.pyplot as plt

plt.plot(de.gbest_y_hist)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

Y_history = pd.DataFrame(de.all_history_Y)
fig, ax = plt.subplots(2, 1)
ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
Y_history.min(axis=1).cummin().plot(kind='line')
plt.show()




