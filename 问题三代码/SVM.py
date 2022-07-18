import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

gr = pd.read_csv('./clean451.csv', index_col=0, encoding='gb18030')
Feature_1 = ['ATSm2', 'BCUTc-1h', 'SCH-6', 'VC-5', 'SHBd', 'SsCH3', 'SaaO', 'minHBa', 'hmin', 'LipoaffinityIndex',
             'FMF', 'MDEC-23', 'MLFER_S', 'WPATH']

Feature_2 = ['apol', 'ATSc1', 'ATSm3', 'SCH-6', 'VCH-7', 'SP-6', 'SHBd', 'SHsOH', 'SHaaCH', 'minHBa',
             'maxsOH', 'ETA_dEpsilon_D', 'ETA_Shape_P', 'ETA_Shape_Y', 'ETA_BetaP_s', 'ETA_dBetaP']

Feature_3 = ['ATSc2', 'BCUTc-1l', 'BCUTc-1h', 'BCUTp-1h', 'SHBd', 'SHother', 'SsOH', 'minHBd', 'minHBa', 'minssCH2',
             'minaaCH', 'minaasC', 'maxHBd', 'maxwHBa', 'maxHBint8', 'maxHsOH', 'hmin', 'LipoaffinityIndex',
             'ETA_dEpsilon_B', 'ETA_Shape_Y',
             'ETA_EtaP_F', 'ETA_Eta_R_L', 'MDEO-11', 'WTPT-4']

Feature_4 = ['ATSc2', 'BCUTc-1l', 'BCUTp-1l', 'VCH-6', 'SC-5', 'SPC-6', 'VP-3', 'SHsOH', 'SdO', 'minHBa',
             'minHsOH', 'maxHother', 'maxdO', 'hmin', 'MAXDP2', 'ETA_dEpsilon_B', 'ETA_Shape_Y', 'ETA_EtaP_F_L',
             'MDEC-23', 'MLFER_A',
             'TopoPSA', 'WTPT-2', 'WTPT-4']

Feature_5 = ['nN', 'ATSc2', 'SCH-7', 'VPC-5', 'SP-6', 'SHaaCH', 'SssCH2', 'SsssCH', 'SssO', 'minHBa',
             'mindssC', 'maxsCH3', 'maxsssCH', 'maxssO', 'hmin', 'ETA_dEpsilon_B', 'ETA_dEpsilon_C', 'ETA_Shape_Y',
             'ETA_BetaP', 'ETA_BetaP_s',
             'ETA_EtaP_F', 'ETA_EtaP_B_RC', 'FMF', 'nHBAcc', 'MLFER_E', 'WTPT-4']

feature_df = gr[Feature_2]
x = feature_df.values
# y_var = ['Caco-2', 'CYP3A4', 'hERG', 'HOB', 'MN']
y_var = ['CYP3A4']
for v in y_var:
    y = gr[v]

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, train_size=0.8)
    print('训练集和测试集 shape', x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    clf = SVC(C=30, kernel='rbf', decision_function_shape='ovo', max_iter=1000)

    ## 模型训练
    clf.fit(x_train, y_train)

    print(clf.score(x_train, y_train))
    print('训练集准确率：', accuracy_score(y_train, clf.predict(x_train)))
    print(clf.score(x_test, y_test))
    print('测试集准确率：', accuracy_score(y_test, clf.predict(x_test)))
