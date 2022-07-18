import numpy as np
from datetime import datetime
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import pandas as pd
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

Set = pd.read_csv('..\问题一代码\Clean_Data.csv', encoding='gb18030', index_col=0)
print(Set)
Set_label = Set['pIC50'].copy()
scaler = StandardScaler()
Set.loc[:, Set.columns != 'SMILES'] = scaler.fit_transform(Set.loc[:, Set.columns != 'SMILES'])
print(Set)
Set_unit_index =  ['C1SP2', 'SsOH', 'minHBa', 'maxssO', 'ALogP', 'ATSc1', 'ATSc3', 'ATSc5', 'BCUTc-1l', 'BCUTc-1h', 'BCUTp-1h', 'mindssC', 'minsssN', 'hmin', 'LipoaffinityIndex', 'MAXDP2', 'ETA_BetaP_s', 'nHBAcc', 'nBondsD2', 'nsssCH']

Set_feature = Set.loc[:, Set_unit_index]
print(Set_feature)
X = Set_feature
y = Set_label

kfolds = KFold(n_splits=5, shuffle=True, random_state=42)

def rmsle2(y, y_pred):
    return mean_squared_error(y, y_pred)

def rmsle(y, y_pred):
    return mean_squared_error(y, y_pred)
# build our model scoring function
def cv_rmse(model, X=X):
    rmse = -cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds)
    return rmse


alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]



ridge = make_pipeline(RobustScaler(),
                      RidgeCV(alphas=0.1, cv=kfolds))

lasso = make_pipeline(RobustScaler(),
                      LassoCV(max_iter=1000, alphas=0.0001,
                              random_state=42, cv=kfolds))

elasticnet = make_pipeline(RobustScaler(),
                           ElasticNetCV(max_iter=100, alphas=0.0005,
                                        cv=kfolds, l1_ratio=0.8))

svr = make_pipeline(RobustScaler(),
                    SVR(C=10, coef0=0, degree=1, kernel='rbf', gamma=0.1)) #tiaowan

gbr = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                max_depth=4, max_features='sqrt',
                                min_samples_leaf=15, min_samples_split=10,
                                loss='huber', random_state=42)

lightgbm = LGBMRegressor(objective='regression',
                         num_leaves=6,
                         learning_rate=0.03,
                         n_estimators=300,
                         max_bin=200,
                         bagging_fraction=0.75,
                         bagging_freq=5,
                         bagging_seed=7,
                         feature_fraction=0.2,
                         feature_fraction_seed=7,
                         verbose=-1,
                         # min_data_in_leaf=2,
                         # min_sum_hessian_in_leaf=11
                         )

xgboost = XGBRegressor(learning_rate=0.01, n_estimators=346,
                       max_depth=3, min_child_weight=0,
                       gamma=0, subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:linear', nthread=-1,
                       scale_pos_weight=1, seed=27,
                       reg_alpha=0.00006)

# stack
stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet,
                                            gbr, svr, xgboost, lightgbm),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)

print('TEST score on CV')

score = cv_rmse(ridge)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(lasso)
print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(elasticnet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(svr)
print("SVR score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(lightgbm)
print("Lightgbm score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(gbr)
print("GradientBoosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(xgboost)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(stack_gen)
print("stack_gen score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

print('START Fit')
print(datetime.now(), 'StackingCVRegressor')
stack_gen_model = stack_gen.fit(np.array(X), np.array(y))
print(datetime.now(), 'elasticnet')
elastic_model_full_data = elasticnet.fit(X, y)
print(datetime.now(), 'lasso')
lasso_model_full_data = lasso.fit(X, y)
print(datetime.now(), 'ridge')
ridge_model_full_data = ridge.fit(X, y)
print(datetime.now(), 'svr')
svr_model_full_data = svr.fit(X, y)
print(datetime.now(), 'GradientBoosting')
gbr_model_full_data = gbr.fit(X, y)
print(datetime.now(), 'xgboost')
xgb_model_full_data = xgboost.fit(X, y)
print(datetime.now(), 'lightgbm')
lgb_model_full_data = lightgbm.fit(X, y)


print((stack_gen_model.predict(np.array(X))))

def blend_models_predict(X):
    return ((0.02 * elastic_model_full_data.predict(X)) + \
            (0.02 * lasso_model_full_data.predict(X)) + \
            (0.02 * lasso_model_full_data.predict(X)) + \
            (0.14 * svr_model_full_data.predict(X)) + \
            (0.15 * gbr_model_full_data.predict(X)) + \
            (0.15 * xgb_model_full_data.predict(X)) + \
            (0.15 * lgb_model_full_data.predict(X)) + \
            (0.35 * stack_gen_model.predict(np.array(X))))

def blend_models_predict(X):
    return ((0.15 * rf_model_full_data.predict(X)) + \
            (0.15 * svr_model_full_data.predict(X)) + \
            (0.15 * gbr_model_full_data.predict(X)) + \
            (0.15 * xgb_model_full_data.predict(X)) + \
            (0.15 * lgb_model_full_data.predict(X)) + \
            (0.25 * stack_gen_model.predict(np.array(X))))

from sklearn.metrics import mean_absolute_percentage_error,mean_squared_error,mean_absolute_error,r2_score

def all_duliang(model,X=X, y=y):
    ac_2 = mean_squared_error(y, model.predict(X))
    ridge_mae = mean_absolute_error(y, model.predict(X))
    ridge_mape = mean_absolute_percentage_error(y, model.predict(X))
    ridge_r2 = r2_score(y, model.predict(X))
    print("MSE =", ac_2)
    print("MAE =", ridge_mae)
    print("MAPE =", ridge_mape)
    print("r2 =", ridge_r2)
def all_duliang2(model,X=X, y=y):
    ac_2 = mean_squared_error(y, model(X))
    ridge_mae = mean_absolute_error(y, model(X))
    ridge_mape = mean_absolute_percentage_error(y, model(X))
    ridge_r2 = r2_score(y, model(X))
    print("MSE =", ac_2)
    print("MAE =", ridge_mae)
    print("MAPE =", ridge_mape)
    print("r2 =", ridge_r2)


print(blend_models_predict(X))
print('RMSLE score on train data:')
print(rmsle(y, blend_models_predict(X)))

print("elastic_model_full_data:")
all_duliang(model = elastic_model_full_data , X=X, y =y)
print("lasso_model_full_data:")
all_duliang(model = lasso_model_full_data , X=X, y =y)
print("ridge_model_full_data:")
all_duliang(model = ridge_model_full_data , X=X, y =y)
print("rf_model_full_data:")
all_duliang(model = rf_model_full_data , X=X, y =y)
print("svr_model_full_data:")
all_duliang(model = svr_model_full_data , X=X, y =y)
print("gbr_model_full_data:")
all_duliang(model = gbr_model_full_data , X=X, y =y)
print("xgb_model_full_data:")
all_duliang(model = xgb_model_full_data , X=X, y =y)
print("lgb_model_full_data:")
all_duliang(model = lgb_model_full_data , X=X, y =y)
print("stack_gen_model:")
all_duliang(model = stack_gen_model , X=X, y =y)
print("blend_models_9_zhong:")
all_duliang2(model = blend_models_predict , X=X, y =y)
print("blend_models_6_zhong:")



# 预测test上面的样本
Set = pd.read_csv('E:\\test\jianmo\预测\Molecular_test.csv', encoding='gb18030', index_col=0)

scaler = StandardScaler()
Set.loc[:, Set.columns != 'SMILES'] = scaler.fit_transform(Set.loc[:, Set.columns != 'SMILES'])
Set_unit_index =  ['C1SP2', 'SsOH', 'minHBa', 'maxssO', 'ALogP', 'ATSc1', 'ATSc3', 'ATSc5', 'BCUTc-1l', 'BCUTc-1h', 'BCUTp-1h', 'mindssC', 'minsssN', 'hmin', 'LipoaffinityIndex', 'MAXDP2', 'ETA_BetaP_s', 'nHBAcc', 'nBondsD2', 'nsssCH']

Set_feature = Set.loc[:, Set_unit_index]
X = Set_feature.copy()

Set_feature['REa_9'] = blend_models_predict(X)
Set_feature.to_csv('.\Set_feature_stacking.csv', index=False)
