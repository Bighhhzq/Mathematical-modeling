import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from random import sample
plt.rcParams['pdf.fonttype'] = 42
config = {
    "font.family":'Times New Roman',  # 设置字体类型
}
plt.rcParams.update(config)


Set = pd.read_csv('../Clean_Data.csv' , encoding='gb18030', index_col=0)


Set.drop('SMILES', axis=1)
print(Set)


Clean_vars = Set.columns.tolist()
Clean_vars.remove('pIC50')

Cat_vars_low = list(Set.loc[:, (Set.nunique() < 10)].nunique().index)
print(len(Cat_vars_low))
Cat_vars_high = list(Set.loc[:, (Set.nunique() >= 10)].nunique().index)

with plt.rc_context(rc={'figure.dpi': 200, 'axes.labelsize': 8,
                        'xtick.labelsize': 6, 'ytick.labelsize': 6,
                        'legend.fontsize': 6, 'legend.title_fontsize': 6,
                        'axes.titlesize': 9}):
    fig_2, ax_2 = plt.subplots(1, 3, figsize=(8.5, 3.5))

    for idx, (column, axes) in list(enumerate(zip(['nssO','nHsOH', 'nHBDon_Lipinski'], ax_2.flatten()))):

        sns.kdeplot(ax=axes, x=Set['pIC50'],
                    hue=Set[column].astype('category'),
                    common_norm=True,
                    fill=True, alpha=0.2, palette='viridis',
                    linewidth=0.6)

        axes.set_title(str(column), fontsize=9, fontweight='bold', color='#6b1000')

    else:

        [axes.set_visible(False) for axes in ax_2.flatten()[idx + 1:]]

    # Fixing a legend box for a particulal variable

    # ax_2_flat = ax_2.flatten()
    #
    # legend_3 = ax_2_flat[2].get_legend()
    # handles_3 = legend_3.legendHandles
    # legend_3.remove()

    # ax_2_flat[2].legend(handles_3, Set['HouseStyle'].unique(),
    #                     title='HouseStyle', ncol=2)

plt.tight_layout()
plt.show()

plt.savefig('./问题一分布.pdf', dpi=600)

# plt.tight_layout()
# plt.show()

