import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


Set = pd.read_csv('..\问题一代码\Clean_Data.csv', encoding='gb18030', index_col=0)
print(Set)
Set_label = Set['pIC50'].copy()


scaler = StandardScaler()
Set.loc[:, Set.columns != 'SMILES'] = scaler.fit_transform(Set.loc[:, Set.columns != 'SMILES'])
print(Set)
Set_unit_index =  ['C1SP2', 'SsOH', 'minHBa', 'maxssO', 'ALogP', 'ATSc1', 'ATSc3', 'ATSc5', 'BCUTc-1l', 'BCUTc-1h', 'BCUTp-1h', 'mindssC', 'minsssN', 'hmin', 'LipoaffinityIndex', 'MAXDP2', 'ETA_BetaP_s', 'nHBAcc', 'nBondsD2', 'nsssCH']
Set_feature = Set.loc[:, Set_unit_index]
print(Set_feature)
print(Set_label)

x_train, x_test, y_train, y_test = train_test_split(Set_feature, Set_label.astype('int') , test_size=0.2, random_state=42)

x_train,y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))


from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dropout

model =Sequential()
model.add(LSTM(64,return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.5))
model.add(LSTM(64, return_sequences= False))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Dense(1))
model.summary()

from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error, mean_squared_error , r2_score
import tensorflow as tf
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mean_squared_error')

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

model.fit(x_train,y_train, batch_size=85, epochs=300, validation_data=(x_test,y_test),validation_batch_size=100)

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
predictions = model.predict(x_test)
print(predictions)
print(predictions.shape)
print(y_test.shape)

ridge_mae = mean_absolute_error(y_test, predictions)
ridge_mape = mean_absolute_percentage_error(y_test, predictions)
ridge_mse = mean_squared_error(y_test, predictions)
ridge_r2 = r2_score(y_test, predictions)
print("Ridge MAE =", ridge_mae)
print("Ridge MAPE =", ridge_mape)
print("Ridge MSE =", ridge_mse)
print("Ridge r2 =", ridge_r2)

Set = pd.read_csv('E:\\test\jianmo\预测\Molecular_test.csv', encoding='gb18030', index_col=0)

print(Set)
scaler = StandardScaler()
Set.loc[:, Set.columns != 'SMILES'] = scaler.fit_transform(Set.loc[:, Set.columns != 'SMILES'])
print(Set)
Set_unit_index =  ['C1SP2', 'SsOH', 'minHBa', 'maxssO', 'ALogP', 'ATSc1', 'ATSc3', 'ATSc5', 'BCUTc-1l', 'BCUTc-1h', 'BCUTp-1h', 'mindssC', 'minsssN', 'hmin', 'LipoaffinityIndex', 'MAXDP2', 'ETA_BetaP_s', 'nHBAcc', 'nBondsD2', 'nsssCH']

Set_feature = Set.loc[:, Set_unit_index]
X = Set_feature.copy()

x_test = np.array(X)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

predictions = model.predict(x_test)

Set_feature['REa_9'] = predictions
Set_feature['REa_6'] = predictions
Set_feature.to_csv('E:\\test\jianmo\Set_feature22.csv', index=False)