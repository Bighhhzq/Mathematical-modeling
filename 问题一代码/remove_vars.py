import numpy as np
import pandas as pd

Set = pd.read_csv('./data.csv', encoding='gb18030')
feature = Set.drop('pIC50', axis=1).columns.tolist()
vars_set = Set.drop('SMILES', axis=1)
zi_var = vars_set.drop('pIC50', axis=1).columns.tolist()
print(zi_var)

delete = []
for a in zi_var:
    if ((Set[a] == 0).sum() / 1973) > 0.99:
        delete.append(a)

clean_Set1 = Set.drop(delete, axis=1)
print(clean_Set1)
clean_Set1.to_csv('./clean451.csv', encoding='gb18030')

clean_zi_set1 = vars_set.drop(delete, axis=1)
clean_zi_var1 = clean_zi_set1.columns.tolist()


def three_sigma(Ser1):
    '''
    Ser1：表示传入DataFrame的某一列。
    '''
    rule = (Ser1.mean() - 10 * Ser1.std() > Ser1) | (Ser1.mean() + 10 * Ser1.std() < Ser1)
    index = np.arange(Ser1.shape[0])[rule]
    return index  # 返回落在3sigma之外的行索引值


def delete_out3sigma(Set, var):
    """
    data：待检测的DataFrame
    """
    data1 = Set[var]
    data = (data1 - data1.min()) / (data1.max() - data1.min())
    out_index = []  # 保存要删除的行索引
    for i in range(data.shape[1]):  # 对每一列分别用3sigma原则处理
        index = three_sigma(data.iloc[:, i])
        out_index += index.tolist()
    delete_ = list(set(out_index))
    print('所删除的行索引为：', delete_)
    print(len(delete_))
    Set.drop(delete_, inplace=True)
    return Set


clean_Set0 = delete_out3sigma(clean_Set1, clean_zi_var1)  # 去除异常样本后的结果

clean_Set0.to_csv('./clean_set0.csv', encoding='gb18030')
