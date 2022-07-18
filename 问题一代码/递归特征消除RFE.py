from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import mean_absolute_percentage_error,mean_squared_error,mean_absolute_error,r2_score

# Create the RFE object and rank each pixel
Set = pd.read_csv('E:\\test\jianmo\特征选择\Clean_Data.csv', encoding='gb18030', index_col=0)
print(Set)
Set_label = Set['pIC50']
print(Set_label)
Set_feature = Set.drop(['SMILES','pIC50'],axis=1)
print(Set_feature)

x_train, x_test, y_train, y_test = train_test_split(Set_feature, Set_label , test_size=0.3, random_state=42)

clf_rf_3 = RandomForestRegressor()
rfe = RFE(estimator=clf_rf_3, n_features_to_select=40, step=1)
rfe = rfe.fit(x_train, y_train)

print('Chosen best 25 feature by rfe:',x_train.columns[rfe.support_])

ac = mean_absolute_percentage_error(y_test,rfe.predict(x_test))
print('Accuracy is: ',ac)
ac_2 = mean_squared_error(y_test,rfe.predict(x_test))
ridge_mae = mean_absolute_error(y_test, rfe.predict(x_test))
ridge_mape = mean_absolute_percentage_error(y_test, rfe.predict(x_test))
ridge_r2 = r2_score(y_test, rfe.predict(x_test))
print("MSE =", ac_2)
print("MAE =", ridge_mae)
print("MAPE =", ridge_mape)
print("r2 =", ridge_r2)