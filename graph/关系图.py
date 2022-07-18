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

Cat_vars_low = list(Set.loc[:, (Set.nunique() < 6)].nunique().index)
print(len(Cat_vars_low))
Cat_vars_high = list(Set.loc[:, (Set.nunique() >= 10)].nunique().index)

with plt.rc_context(rc = {'figure.dpi': 600, 'axes.labelsize': 12,
                          'xtick.labelsize': 12, 'ytick.labelsize': 12}):

    fig_3, ax_3 = plt.subplots(3, 3, figsize = (15, 15))

    for idx, (column, axes) in list(enumerate(zip(Cat_vars_low[0:9], ax_3.flatten()))):
        order = Set.groupby(column)['pIC50'].mean().sort_values(ascending = True).index
        plt.xticks(rotation=90)
        sns.violinplot(ax = axes, x = Set[column],
                       y = Set['pIC50'],
                       order = order, scale = 'width',
                       linewidth = 0.5, palette = 'viridis',
                       inner = None)

        plt.setp(axes.collections, alpha = 0.3)

        sns.stripplot(ax = axes, x = Set[column],
                      y = Set['pIC50'],
                      palette = 'viridis', s = 1.5, alpha = 0.75,
                      order = order, jitter = 0.07)

        sns.pointplot(ax = axes, x = Set[column],
                      y = Set['pIC50'],
                      order = order,
                      color = '#ff5736', scale = 0.2,
                      estimator = np.mean, ci = 'sd',
                      errwidth = 0.5, capsize = 0.15, join = True)

        plt.setp(axes.lines, zorder = 100)
        plt.setp(axes.collections, zorder = 100)

        if Set[column].nunique() > 5:

            plt.setp(axes.get_xticklabels(), rotation = 90)

    else:

        [axes.set_visible(False) for axes in ax_3.flatten()[idx + 1:]]

plt.savefig('./问题一小提琴.pdf', dpi=600)

# plt.tight_layout()
# plt.show()

