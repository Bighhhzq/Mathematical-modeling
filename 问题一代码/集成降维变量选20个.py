
import pandas as pd
import heapq

Set = pd.read_csv('E:\\test\jianmo\特征选择\Clean_Data.csv', encoding='gb18030', index_col=0)
print(Set)
Set_label = Set['pIC50']
print(Set_label)
Set_feature = Set.drop(['SMILES','pIC50'],axis=1)
print(Set_feature)
print(Set_feature.columns.tolist())

hu_xx= ['ALogP', 'ATSc5', 'nBondsD2', 'nBondsM', 'C2SP3', 'nHBd', 'nHBint2', 'nHBint3', 'nssCH2', 'nsssCH', 'naasC', 'ndsN', 'nssO', 'naaO', 'nssS', 'SdsN', 'SaaO', 'SssS', 'minHBint7', 'minHaaCH', 'mindsN', 'minaaO', 'minssS', 'maxwHBa', 'maxsssCH', 'maxdsN', 'maxaaO', 'ETA_Shape_P', 'ETA_Beta_ns', 'ETA_BetaP_ns', 'ETA_dBeta', 'ETA_Beta_ns_d', 'ETA_BetaP_ns_d', 'nHBDon', 'nHBDon_Lipinski', 'MDEN-23', 'nRing', 'n7Ring', 'nTRing', 'nT7Ring']
dan_chi2 = ['minHBa', 'ETA_dBeta', 'C1SP2', 'SsOH', 'maxsOH', 'minsOH', 'SsssN',
       'minsssN', 'maxsssN', 'SdO', 'maxdO', 'mindO', 'SHBint10', 'minHBint10',
       'maxHBint10', 'SssNH', 'nAtomLAC', 'maxssNH', 'minssNH', 'SssO',
       'maxssO', 'SaaN', 'minssO', 'nHsOH', 'nHsOH', 'maxaaN', 'minaaN',
       'nsssCH', 'SssCH2', 'ndO', 'SHCsats', 'C2SP3', 'nssO', 'ALogP',
       'nHBint10', 'C1SP3', 'SHBint4', 'nHCsats', 'nBondsD2', 'nssCH2']
dan_huxinxi =  ['SHsOH', 'BCUTc-1l', 'BCUTc-1h', 'minHsOH', 'maxHsOH', 'MLFER_A', 'SHBd', 'Kier3', 'minHBd', 'minHBa', 'hmin', 'ATSc2', 'BCUTp-1h', 'ETA_BetaP_s', 'maxsOH', 'maxHBd', 'ATSc3', 'maxHCsats', 'mindssC', 'minaasC', 'C1SP2', 'minwHBa', 'McGowan_Volume', 'nHBAcc', 'ATSc4', 'VP-6', 'WTPT-5', 'ETA_Alpha', 'SP-2', 'maxssO', 'LipoaffinityIndex',
       'SP-6', 'ATSc1', 'minssCH2', 'VPC-6', 'VP-4', 'VPC-5', 'SsOH', 'MAXDP2',
       'SP-3']
digui_RFE =['ALogP', 'ATSc1', 'ATSc2', 'ATSc3', 'ATSc5', 'BCUTc-1l', 'BCUTc-1h','BCUTp-1h', 'C1SP2', 'SC-3', 'VC-5', 'VP-5', 'ECCEN', 'SHBint10', 'SaaCH', 'SsOH', 'minHBa', 'minHBint4', 'minHsOH', 'mindssC', 'minsssN', 'minsOH', 'minssO', 'maxHsOH', 'maxaaCH', 'maxssO', 'gmax', 'hmin','LipoaffinityIndex', 'MAXDP2', 'ETA_BetaP_s', 'nHBAcc', 'Kier3','MDEC-23', 'MDEC-33', 'MLFER_A', 'MLFER_BH', 'TopoPSA', 'WTPT-5','XLogP']
suiji_senling = ['minsssN', 'MDEC-23', 'LipoaffinityIndex', 'maxHsOH', 'minssO',
       'nHBAcc', 'minHsOH', 'BCUTc-1l', 'maxssO', 'MLFER_A', 'C1SP2',
       'TopoPSA', 'WTPT-5', 'ATSc3', 'ATSc2', 'VC-5', 'mindssC', 'ATSc1',
       'BCUTp-1h', 'BCUTc-1h', 'XLogP', 'MDEC-33', 'Kier3', 'maxsssCH',
       'minsOH', 'hmin', 'minHBa', 'SHsOH', 'minHBint7', 'MAXDP2', 'ATSc5',
       'maxHBa', 'VCH-7', 'ETA_BetaP_s', 'ATSc4', 'SC-3', 'SsOH', 'minssNH',
       'VP-5', 'maxsOH']

l=245
list = [0]*l
for j,i in enumerate(Set_feature.columns.tolist()):
    if i in hu_xx:
        list[j] = list[j]+1
    if i in dan_chi2:
        list[j] = list[j]+1
    if i in dan_huxinxi:
        list[j] = list[j]+1
    if i in digui_RFE:
        list[j] = list[j]+1
    if i in suiji_senling:
        list[j] = list[j]+1

print(list)

print(sorted(list,reverse=True)[:60])
result = map(list.index, heapq.nlargest(60, list))
print(result)

best_21_va=[]

for j,i in enumerate(Set_feature.columns.tolist()):
    if list[j]>=4:
        best_21_va.append(i)

print(best_21_va)

for j,i in enumerate(Set_feature.columns.tolist()):
    if list[j]>=3 and list[j]<4:
        best_21_va.append(i)

print(best_21_va)

for j,i in enumerate(Set_feature.columns.tolist()):
    if list[j]<3 and list[j]>=2:
        best_21_va.append(i)

print(best_21_va)
