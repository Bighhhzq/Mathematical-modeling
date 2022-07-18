import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

from matplotlib import rcParams

# matplotlib.use("pgf")
# plt.rcParams['pdf.fonttype'] = 42
# pgf_config = {
#     "font.family":'serif',
#     "font.size": 35,
#     "pgf.rcfonts": False,
#     "text.usetex": True,
#     "pgf.preamble": [
#         r"\usepackage{unicode-math}",
#         r"\setmainfont{Times New Roman}",
#         r"\usepackage{xeCJK}",
#         r"\setCJKmainfont{SimSun}",
#     ],
# }
# rcParams.update(pgf_config)


# rc = {'axes.unicode_minus': False}
# sns.set(context='notebook', style='ticks', font='SimSon', rc=rc)

# Set = pd.read_csv('../Clean_Data.csv' , encoding='gb18030', index_col=0)
# Slect = ['C1SP2', 'SsOH', 'minHBa', 'maxssO', 'ALogP', 'ATSc1', 'ATSc2', 'ATSc3', 'ATSc5', 'BCUTc-1l', 'BCUTc-1h', 'BCUTp-1h', 'minHsOH', 'mindssC', 'minsssN', 'minsOH', 'minssO', 'maxHsOH', 'maxsOH', 'hmin', 'LipoaffinityIndex', 'MAXDP2', 'ETA_BetaP_s', 'nHBAcc', 'Kier3', 'MLFER_A', 'WTPT-5', 'ATSc4', 'nBondsD2', 'C2SP3', 'SC-3', 'VC-5', 'VP-5', 'nssCH2', 'nsssCH', 'nssO', 'SHBint10', 'SHsOH', 'minHBint7', 'minssNH', 'maxsssCH', 'ETA_dBeta', 'MDEC-23', 'MDEC-33', 'TopoPSA', 'XLogP']
# Slect0 = ['C1SP2', 'SsOH', 'minHBa', 'maxssO', 'ALogP', 'ATSc1', 'ATSc2', 'ATSc3', 'ATSc5', 'BCUTc-1l', 'BCUTc-1h', 'BCUTp-1h', 'minHsOH', 'mindssC', 'minsssN', 'minsOH', 'minssO', 'maxHsOH', 'maxsOH', 'hmin', 'LipoaffinityIndex', 'MAXDP2', 'ETA_BetaP_s', 'nHBAcc', 'Kier3', 'MLFER_A', 'WTPT-5', 'nBondsD2', 'nsssCH']
# Slect00 = ['C1SP2', 'SsOH', 'minHBa', 'maxssO', 'ALogP', 'ATSc1', 'ATSc3', 'ATSc5', 'BCUTc-1l', 'BCUTc-1h', 'BCUTp-1h', 'mindssC', 'minsssN', 'hmin', 'LipoaffinityIndex', 'MAXDP2', 'ETA_BetaP_s', 'nHBAcc', 'nBondsD2', 'nsssCH']


Set = pd.read_csv('../clean451.csv' , encoding='gb18030', index_col=0)
feature11 = ['ATSm2', 'BCUTc-1h', 'SCH-6', 'VC-5', 'SHBd','SsCH3', 'SaaO', 'minHBa', 'hmin',
       'LipoaffinityIndex', 'FMF', 'MDEC-23', 'MLFER_S','WPATH']

feature40 = ['ATSc1','ATSc3', 'ATSc5', 'BCUTc-1l', 'BCUTc-1h', 'BCUTp-1l',
       'SCH-6', 'SC-5', 'VC-5', 'VP-3', 'nHsOH', 'SHBd', 'SaasC',
       'SdO', 'minHBa', 'minHsOH', 'minHother', 'minaasC',
       'maxssCH2', 'maxaasC', 'maxdO', 'hmin', 'gmin',
       'LipoaffinityIndex', 'MAXDP2', 'ETA_Shape_P',
       'FMF', 'MDEC-33', 'MLFER_BO', 'TopoPSA', 'WTPT-2', 'WTPT-4']

feature4 = ['ATSc2', 'BCUTc-1l', 'BCUTp-1l', 'VCH-6', 'SC-5', 'SPC-6', 'VP-3',
       'SHsOH', 'SdO', 'minHBa', 'minHsOH',
       'maxHother', 'maxdO', 'hmin', 'MAXDP2',
       'ETA_dEpsilon_B', 'ETA_Shape_Y', 'ETA_EtaP_F_L', 'MDEC-23', 'MLFER_A',
       'TopoPSA', 'WTPT-2', 'WTPT-4']

feature22 =['apol', 'ATSc1', 'ATSm3', 'SCH-6', 'VCH-7', 'SP-6', 'SHBd', 'SHsOH', 'SHaaCH', 'minHBa', 'maxsOH', 'ETA_dEpsilon_D', 'ETA_Shape_P', 'ETA_Shape_Y',
       'ETA_BetaP_s', 'ETA_dBetaP']

feature3 = ['apol', 'ATSc2', 'BCUTc-1l', 'BCUTc-1h', 'BCUTp-1h', 'bpol', 'VP-0',
       'VP-1', 'VP-2', 'CrippenMR', 'ECCEN', 'SHBd', 'SHother', 'SsOH',
       'minHBd', 'minHBa', 'minssCH2', 'minaaCH', 'minaasC', 'maxHBd',
       'maxwHBa', 'maxHBint8', 'maxHsOH', 'maxaaCH', 'maxaasC', 'hmin',
       'LipoaffinityIndex', 'ETA_dEpsilon_B', 'ETA_Shape_Y', 'ETA_EtaP',
       'ETA_EtaP_F', 'ETA_Eta_R_L', 'fragC', 'Kier2', 'Kier3',
       'McGowan_Volume', 'MDEO-11', 'WTPT-1', 'WTPT-4', 'WPATH']

feature5 = ['nN', 'ATSc2', 'SCH-7', 'VCH-7', 'VPC-5', 'VPC-6', 'SP-6', 'SHaaCH',
       'SssCH2', 'SsssCH', 'SssO', 'minHBa', 'mindssC', 'maxsCH3', 'maxsssCH',
       'maxssO', 'hmin', 'ETA_Epsilon_1', 'ETA_Epsilon_2', 'ETA_Epsilon_4',
       'ETA_dEpsilon_A', 'ETA_dEpsilon_B', 'ETA_dEpsilon_C', 'ETA_Shape_Y',
       'ETA_BetaP', 'ETA_BetaP_s', 'ETA_EtaP_F', 'ETA_Eta_F_L', 'ETA_EtaP_F_L',
       'ETA_EtaP_B_RC', 'FMF', 'nHBAcc', 'nHBAcc_Lipinski', 'MLFER_BO',
       'MLFER_S', 'MLFER_E', 'TopoPSA', 'WTPT-3', 'WTPT-4', 'WTPT-5']

feature55 = ['nN', 'ATSc2', 'SCH-7', 'VPC-5', 'SP-6', 'SHaaCH',
       'SssCH2', 'SsssCH', 'SssO', 'minHBa', 'mindssC', 'maxsCH3', 'maxsssCH',
       'maxssO', 'hmin', 'ETA_dEpsilon_B', 'ETA_dEpsilon_C', 'ETA_Shape_Y',
       'ETA_BetaP', 'ETA_BetaP_s', 'ETA_EtaP_F',
       'ETA_EtaP_B_RC', 'FMF', 'nHBAcc', 'MLFER_E', 'WTPT-4']


df1 = Set[feature55]
r = df1.corr()

f, ax = plt.subplots(figsize= [40,30])
sns.heatmap(r, ax=ax, vmax=1,vmin=-1,annot=True,
            cbar_kws={'label': '相关系数'}, cmap='viridis')
plt.xticks(rotation=90)    # 将字体进行旋转
plt.yticks(rotation=360)


# plt.savefig('./问题一待检验热图.pdf', bbox_inches='tight', dpi=600)
# plt.savefig('./feature1.pdf', bbox_inches='tight', dpi=600)
plt.show()