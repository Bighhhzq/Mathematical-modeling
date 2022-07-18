import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from random import sample
plt.rcParams['pdf.fonttype'] = 42
config = {
    "font.family":'Times New Roman',  # 设置字体类型
    "font.size": 80,
}
plt.rcParams.update(config)

Set = pd.read_csv('../data.csv' , encoding='gb18030')
print(Set)
Set.drop('SMILES', axis=1)


Slect = ['ALogP', 'ATSc1', 'ATSc4', 'ATSc5', 'BCUTc-1l', 'BCUTc-1h', 'BCUTp-1h', 'SHsOH', 'SsOH', 'minHBa', 'minwHBa', 'minHsOH', 'minsOH', 'maxHsOH', 'maxssO', 'hmin', 'LipoaffinityIndex', 'Kier3', 'MDEC-33', 'WTPT-3', 'WTPT-5', 'ATSc2', 'ATSc3', 'nBondsD2', 'C2SP3', 'VC-5', 'CrippenLogP', 'nssCH2', 'nsssCH', 'nssO', 'SwHBa', 'SHBint10', 'SaaCH', 'SsssN', 'SssO', 'minsssN', 'maxHBa', 'maxHother', 'maxaaCH', 'maxaasC', 'maxsOH', 'MAXDP2', 'ETA_BetaP_s', 'ETA_dBeta', 'MDEC-23', 'MLFER_A', 'MLFER_BO', 'TopoPSA', 'XLogP']



Clean_vars = Set.columns.tolist()
Clean_vars.remove('pIC50')
# Cat_vars.remove('Condition2')

# sns.set_theme(rc = {'grid.linewidth': 0.5,
#                     'axes.linewidth': 0.75, 'axes.facecolor': '#fff3e9', 'axes.labelcolor': '#6b1000',
#                     # 'figure.facecolor': '#f7e7da',
#                     'xtick.color': '#6b1000', 'ytick.color': '#6b1000'})

with plt.rc_context(rc={'figure.dpi': 600, 'axes.labelsize': 10,
                        'xtick.labelsize': 12, 'ytick.labelsize': 12}):
    fig_0, ax_0 = plt.subplots(3, 3, figsize=(12, 10))
    for idx, (column, axes) in list(enumerate(zip(Slect[0:9], ax_0.flatten()))):
        sns.scatterplot(ax=axes, x=Set[column],
                        y=Set['pIC50'],
                        hue=Set['pIC50'],
                        palette='viridis', alpha=0.7, s=8)

        # Get rid of legend
        axes.legend([], [], frameon=False)


    # Remove empty figures

    else:

        [axes.set_visible(False) for axes in ax_0.flatten()[idx + 1:]]

# plt.tight_layout()
# plt.show()
plt.savefig('./问题一散点图1.pdf', bbox_inches='tight', dpi=600)
