import pandas as pd
from sklearn.preprocessing import StandardScaler

Set = pd.read_csv('E:\\test\jianmo\优化\clean451.csv', encoding='gb18030', index_col=0)
print(Set)
scaler = StandardScaler()
Set2 = Set.copy()
print(Set['nN'])
Set.loc[:, Set.columns != 'SMILES'] = scaler.fit_transform(Set.loc[:, Set.columns != 'SMILES'])
print(Set)

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

Set_feature = Set.loc[:, featuren_all]
print(Set_feature)
X = Set_feature
print(Set_feature)
print(Set_feature.values)

import numpy as np

best_x = [ 1.38253693e+00, -1.27923420e+01 , 3.34890459e-01 , 2.09864822e+00,
 -1.37653334e+00, -5.92939907e-01,  2.64158346e+00 ,-1.47876716e+01,
  3.46623419e+00  ,1.61801497e+00 ,-2.99576488e+00, -3.48074006e-01,
  6.38688156e-01, -5.19686352e-01,  9.00704135e-01,  3.09167458e+00,
 -3.47797975e+00 ,-2.37999354e+00, -6.02536699e-01, -3.18153296e+00,
  7.42434944e-01,-2.89563644e+00 , 3.20435290e-01 , 2.17453028e+01,
  1.03539905e+00,  5.36896266e+00 , 2.09566882e+00,  5.27409742e-01,
  3.87202400e+00,  3.36569703e+00 , 6.04737620e+00,  2.61958616e+00,
  1.80990651e+00 , 1.43818619e+01 , 1.48127264e+00,  1.60160046e+00,
  2.11726944e-01, -2.29679055e+00 ,-3.80125811e+00 , 1.40864553e+00,
  1.03629442e+01  ,1.81157543e+01, -2.79026323e+00  ,3.56833458e+00,
  2.59651752e+00,  2.47588700e+01 , 1.03023707e+00 , 2.37580593e+00,
 -3.24008740e+00,  1.18860580e+00 ,-3.09182989e+00,  1.65976731e+00,
  1.67475725e+00  ,2.92308457e+00 ,-4.98709744e-01 , 2.31794159e+00,
  7.78201037e+00,  5.95230613e-01,  1.27464854e+00 , 1.11117586e-01,
  8.62933321e+00,  7.10097513e+00 ,-1.87948566e+00,  7.08198140e+00,
  3.20053469e-02 , 6.00339053e-01 , 1.89878864e+00,  4.70380159e-02,
  6.31825754e+00 , 8.39251470e+00,  5.51045242e+00 , 3.68552455e+00,
  1.35371005e+01,  3.74846338e+01,  1.66060646e+00, -1.27641030e+01,
  3.07137345e+00 ,-2.82324098e+00,  1.55734042e+00,  1.41112778e+01,
  8.31509632e+00]

print('------------------------------------------------------------------------------')
print(np.mean(Set2[featuren_all]))
print(np.std(Set2[featuren_all]))
# mix_value.append(i*(np.std(Set2[featuren_all])) + np.mean(Set2[featuren_all]))

best_value = []
for j,i in enumerate(best_x):
    best_value.append((i*(np.std(Set2[featuren_all])[j])+(np.mean(Set2[featuren_all])[j])))
print(best_value)


max_list=[]
for i in Set_feature.values:
    chazhi = best_x - i
    list1=[]
    a = 0
    for j in chazhi:
        if abs(j)<2:
            list1.append(abs(j))
        elif abs(j)<6:
            list1.append(np.sqrt((abs(j))))
        elif abs(j)<24:
            list1.append((np.sqrt(np.sqrt(abs(j)))))
        else:
            list1.append(np.sqrt(np.sqrt(np.sqrt(abs(j)))))
    a = sum(list1)
    max_list.append(a)
arr = np.array(max_list)
mix_index = arr.argsort()[:20][::1].tolist()
mix_index.append(1832)
mix_index.append(1786)
mix_index.append(1784)
mix_index.append(1467)
mix_index.append(1466)
mix_index.append(1464)
mix_index.append(1454)
mix_index.append(515)
mix_index.append(512)
mix_index.append(489)
mix_index.append(478)
mix_index.append(474)
mix_index.append(472)
mix_index.append(470)
mix_index.append(467)
mix_index.append(461)
mix_index.append(460)
Set_d = Set_feature.loc[mix_index].copy()
mix_value = []
for j,i in enumerate(np.min(Set_d)):
    mix_value.append(i*(np.std(Set2[featuren_all])[j]) + np.mean(Set2[featuren_all])[j])
max_value = []
for j,i in enumerate(np.max(Set_d)):
    max_value.append(i*(np.std(Set2[featuren_all])[j]) + np.mean(Set2[featuren_all])[j])
print(mix_value)
print(max_value)

data = {
    'min': mix_value,
    'max': max_value,
    'fuhao': featuren_all,
}
row_index = featuren_all
col_names=['min', 'max']
df=pd.DataFrame(data,columns=col_names,)
print(df)
df['最优解'] = best_value
df['fuhao'] = featuren_all

print(df)
df.to_csv('E:\\test\jianmo\优化问题问题取值范围DE.csv', index=False)
