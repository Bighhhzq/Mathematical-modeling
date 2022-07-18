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


Clean_vars = Set.columns.tolist()
Clean_vars.remove('pIC50')
# Cat_vars.remove('Condition2')

# sns.set_theme(rc = {'grid.linewidth': 0.5,
#                     'axes.linewidth': 0.75, 'axes.facecolor': '#fff3e9', 'axes.labelcolor': '#6b1000',
#                     # 'figure.facecolor': '#f7e7da',
#                     'xtick.color': '#6b1000', 'ytick.color': '#6b1000'})

with plt.rc_context(rc={'figure.dpi': 600, 'axes.labelsize': 10,
                        'xtick.labelsize': 12, 'ytick.labelsize': 12}):
    fig_0, ax_0 = plt.subplots(1, 1, figsize=(15, 8))

    sns.scatterplot(ax=ax_0, x=list(range(0,1974)),
                        y=Set['pIC50'],
                        hue=Set['pIC50'],
                        alpha=0.7,)
    my_y_ticks = np.arange(2, 12, 2)
    my_x_ticks = np.arange(0, 2000, 200)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)

    # Get rid of legend
    ax_0.legend([], [], frameon=False)


    # Remove empty figures


# plt.tight_layout()
# plt.show()
plt.savefig('./问题一因变量.pdf', dpi=600)
