import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


Set = pd.read_csv('./tu2.csv' , encoding='gb18030')



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

featuren_pre = ['C1SP2', 'SsOH', 'minHBa', 'maxssO', 'ALogP',
              'ATSc1', 'ATSc3', 'ATSc5', 'BCUTc-1l', 'BCUTc-1h',
              'BCUTp-1h', 'mindssC', 'minsssN', 'hmin', 'LipoaffinityIndex',
              'MAXDP2', 'ETA_BetaP_s', 'nHBAcc', 'nBondsD2', 'nsssCH']




df_var_all = Set[featuren_all]
df_var_pre = Set[featuren_pre]

print(df_var_pre.describe())

df_var_pre.to_csv('./pred.csv')