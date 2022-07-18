import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# 设置
pd.options.display.notebook_repr_html=False  # 表格显示
plt.rcParams['figure.dpi'] = 75  # 图形分辨率
# sns.set_theme(style='darkgrid')  # 图形主题
sns.set_theme(style='dark')  # 图形主题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


Set1 = pd.read_csv('./tu2.csv' , encoding='gb18030')
x1 = Set1['SsOH']
y1 = Set1['bianhua']

Set2 = pd.read_csv('./tu2_ALogP.csv' , encoding='gb18030')
x2 = Set2['ALogP']
y2 = Set2['bianhua']

Set3 = pd.read_csv('./tu2_ATSc1.csv' , encoding='gb18030')
x3 = Set3['ATSc1']
y3 = Set3['bianhua']

Set4 = pd.read_csv('./tu2_C1SP2.csv' , encoding='gb18030')
x4 = Set4['C1SP2']
y4 = Set4['bianhua']

Set5 = pd.read_csv('./tu2_maxssO.csv' , encoding='gb18030')
x5 = Set5['maxssO']
y5 = Set5['bianhua']

Set6 = pd.read_csv('./tu2_minHBa.csv' , encoding='gb18030')
x6 = Set6['minHBa']
y6= Set6['bianhua']






# Set = pd.read_csv('../nihe.csv' , encoding='gb18030')
# print(Set)

# y0 = Set["pIC50"]
# y1 = Set["REa_9"]

# yq = Set.iloc[0:1976:4]
#
# yq.reset_index(drop=True, inplace=True)
# print(yq)
#
# y0 = yq["y0"]
# y1 = yq["y1"]
# x = range(0,len(a))

plt.figure(figsize=[20,30])

plt.subplot(2, 3, 1)

# plt.scatter(x, y, marker='o')
plt.plot(x1, y1, color='g', marker='', label='pIC50', markersize=5, linewidth="2")
# plt.plot(x, y1, color='b', marker='', label='预测值', markersize=5, linewidth="2")
plt.xlabel('SsOH',fontsize=16)
plt.xticks(x1,())
plt.yticks(fontproperties='Times New Roman', size=16)
plt.xticks(fontproperties='Times New Roman', size=16)
plt.legend(fontsize=15)
# plt.show()

plt.subplot(2, 3, 2)

# plt.scatter(x, y, marker='o')
plt.plot(x2, y2, color='g', marker='', label='pIC50', markersize=5, linewidth="2")
# plt.plot(x, y1, color='b', marker='', label='预测值', markersize=5, linewidth="2")
plt.xlabel('ALogP',fontsize=16)
plt.xticks(x2,())
plt.yticks(fontproperties='Times New Roman', size=16)
plt.xticks(fontproperties='Times New Roman', size=16)
plt.legend(fontsize=15)
# plt.show()

plt.subplot(2, 3, 3)

# plt.scatter(x, y, marker='o')
plt.plot(x3, y3, color='g', marker='', label='pIC50', markersize=5, linewidth="2")
# plt.plot(x, y1, color='b', marker='', label='预测值', markersize=5, linewidth="2")
plt.xlabel('ATSc1',fontsize=16)
plt.xticks(x3,())
plt.yticks(fontproperties='Times New Roman', size=16)
plt.xticks(fontproperties='Times New Roman', size=16)
plt.legend(fontsize=15)
# plt.show()

plt.subplot(2, 3, 4)

# plt.scatter(x, y, marker='o')
plt.plot(x4, y4, color='g', marker='', label='pIC50', markersize=5, linewidth="2")
# plt.plot(x, y1, color='b', marker='', label='预测值', markersize=5, linewidth="2")
plt.xlabel('C1SP2',fontsize=16)
plt.xticks(x4,())
plt.yticks(fontproperties='Times New Roman', size=16)
plt.xticks(fontproperties='Times New Roman', size=16)
plt.legend(fontsize=15)
# plt.show()

plt.subplot(2, 3, 5)

# plt.scatter(x, y, marker='o')
plt.plot(x5, y5, color='g', marker='', label='pIC50', markersize=5, linewidth="2")
# plt.plot(x, y1, color='b', marker='', label='预测值', markersize=5, linewidth="2")
plt.xlabel('maxssO',fontsize=16)
plt.xticks(x5,())
plt.yticks(fontproperties='Times New Roman', size=16)
plt.xticks(fontproperties='Times New Roman', size=16)
plt.legend(fontsize=15)
# plt.show()

plt.subplot(2, 3, 6)

# plt.scatter(x, y, marker='o')
plt.plot(x6, y6, color='g', marker='', label='pIC50', markersize=5, linewidth="2")
# plt.plot(x, y1, color='b', marker='', label='预测值', markersize=5, linewidth="2")
plt.xlabel('minHBa',fontsize=16)
plt.xticks(x6,())
plt.yticks(fontproperties='Times New Roman', size=16)
plt.xticks(fontproperties='Times New Roman', size=16)
plt.legend(fontsize=15)
plt.show()