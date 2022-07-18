from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error,mean_squared_error,mean_absolute_error,r2_score
import numpy as np

Set = pd.read_csv('E:\\test\jianmo\特征选择\Clean_Data.csv', encoding='gb18030', index_col=0)
print(Set)
Set_label = Set['pIC50']
print(Set_label)
Set_feature = Set.drop(['SMILES','pIC50'],axis=1)
print(Set_feature)
x_train, x_test, y_train, y_test = train_test_split(Set_feature, Set_label, test_size=0.3, random_state=42)
clf_rf_5 = RandomForestRegressor()
clr_rf_5 = clf_rf_5.fit(x_train,y_train)
ac_2 = mean_squared_error(y_test,clr_rf_5.predict(x_test))
ridge_mae = mean_absolute_error(y_test, clr_rf_5.predict(x_test))
ridge_mape = mean_absolute_percentage_error(y_test, clr_rf_5.predict(x_test))
ridge_r2 = r2_score(y_test, clr_rf_5.predict(x_test))
print("MSE =", ac_2)
print("MAE =", ridge_mae)
print("MAPE =", ridge_mape)
print("r2 =", ridge_r2)


importances = clr_rf_5.feature_importances_
std = np.std([tree.feature_importances_ for tree in clr_rf_5.estimators_], axis=0)

indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# best_index = [136,217,166,146,139,204,126,26,161,221,38,233,240,12,11,47,131,10,28,27,243,218,212,153,137,165,121,103,124,168,14,143,44,186,13,45,115,133,67,159]
# print(Set_feature.columns[best_index])
#
#
# import matplotlib.pyplot as plt
#
# plt.figure(1, figsize=(14, 13))
# plt.title("Feature importances")
# plt.bar(range(x_train.shape[1])[:40], importances[best_index], color="g" , align="center")
# plt.xticks(range(x_train.shape[1])[:40], x_train.columns[best_index],rotation=90)
# plt.xlim([-1, 40])
# plt.show()
