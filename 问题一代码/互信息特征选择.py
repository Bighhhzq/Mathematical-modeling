from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error,r2_score
import numpy as np
# Create the RFE object and rank each pixel
Set = pd.read_csv('E:\\test\jianmo\特征选择\Clean_Data.csv', encoding='gb18030', index_col=0)
print(Set)
def sigmoid(inx):
    return 1.0/(1+np.exp(-inx))
Set_label = Set['pIC50']
print(Set_label)
Set_feature = Set.drop(['SMILES','pIC50'],axis=1)
print(Set_feature)
#Set_feature = Set_feature.loc[:,['ALogP', 'ATSc5', 'nBondsD2', 'nBondsM', 'C2SP3', 'nHBd', 'nHBint2', 'nHBint3', 'nssCH2', 'nsssCH', 'naasC', 'ndsN', 'nssO', 'naaO', 'nssS', 'SdsN', 'SaaO', 'SssS', 'minHBint7', 'minHaaCH', 'mindsN', 'minaaO', 'minssS', 'maxwHBa', 'maxsssCH', 'maxdsN', 'maxaaO', 'ETA_Shape_P', 'ETA_Beta_ns', 'ETA_BetaP_ns', 'ETA_dBeta', 'ETA_Beta_ns_d', 'ETA_BetaP_ns_d', 'nHBDon', 'nHBDon_Lipinski', 'MDEN-23', 'nRing', 'n7Ring', 'nTRing', 'nT7Ring']]
Set_feature = Set_feature.loc[:,['ALogP', 'ATSc1', 'ATSc5', 'BCUTc-1l', 'BCUTc-1h', 'BCUTp-1h', 'SsOH', 'minHBa', 'minwHBa', 'minHsOH', 'hmin', 'LipoaffinityIndex', 'Kier3', 'MDEC-33', 'WTPT-5','nssCH2', 'nsssCH', 'nssO', 'maxaaCH', 'SHBint10']]

print(Set_feature)


x_train, x_test, y_train, y_test = train_test_split(Set_feature, Set_label , test_size=0.3, random_state=42)

from sklearn.ensemble import RandomForestRegressor #集成学习中的随机森林
rfc = RandomForestRegressor(random_state=42)
rfc = rfc.fit(x_train,y_train)

ac_2 = mean_squared_error(y_test,rfc.predict(x_test))
ridge_mae = mean_absolute_error(y_test, rfc.predict(x_test))
ridge_mape = mean_absolute_percentage_error(y_test, rfc.predict(x_test))
ridge_r2 = r2_score(y_test, rfc.predict(x_test))
print("MSE =", ac_2)
print("MAE =", ridge_mae)
print("MAPE =", ridge_mape)
print("r2 =", ridge_r2)

