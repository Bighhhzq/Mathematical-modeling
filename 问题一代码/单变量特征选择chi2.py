from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error,mean_squared_error,mean_absolute_error,r2_score
import numpy as np
Set = pd.read_csv('E:\\test\jianmo\特征选择\Clean_Data.csv', encoding='gb18030', index_col=0)
print(Set)
def sigmoid(inx):
    return 1.0/(1+np.exp(-inx))
Set_label = Set['pIC50']
print(Set_label)
Set_feature = Set.drop(['SMILES','pIC50'],axis=1)
print(Set_feature)
Set_feature = sigmoid(Set_feature)
print(Set_feature)
x_train, x_test, y_train, y_test = train_test_split(Set_feature, Set_label.astype("int") , test_size=0.3, random_state=42)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import heapq

select_feature = SelectKBest(chi2, k=10).fit(x_train, y_train)

print('Score list:', select_feature.scores_)
print(len(select_feature.scores_))
print(sorted(select_feature.scores_,reverse=True)[:10])
result = map(select_feature.scores_.tolist().index, heapq.nlargest(10, select_feature.scores_.tolist()))
# print(list(result))
best_index = list(result)
print(best_index)
print(Set_feature.columns[best_index])

x_train_2 = select_feature.transform(x_train)
x_test_2 = select_feature.transform(x_test)
#random forest classifier with n_estimators=10 (default)
clf_rf_2 = RandomForestRegressor()
clr_rf_2 = clf_rf_2.fit(x_train_2,y_train)
ac_2 = mean_absolute_percentage_error(y_test,clf_rf_2.predict(x_test_2))
print('Accuracy is: ',ac_2)
ac_2 = mean_squared_error(y_test,clf_rf_2.predict(x_test_2))
ridge_mae = mean_absolute_error(y_test, clf_rf_2.predict(x_test_2))
ridge_mape = mean_absolute_percentage_error(y_test, clf_rf_2.predict(x_test_2))
ridge_r2 = r2_score(y_test, clf_rf_2.predict(x_test_2))
print("MSE =", ac_2)
print("MAE =", ridge_mae)
print("MAPE =", ridge_mape)
print("r2 =", ridge_r2)
