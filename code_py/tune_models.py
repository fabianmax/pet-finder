import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
from code_py.define_models import ModelXGB, ModelLGB, ModelCat, cv

import xgboost as xgb
import catboost as cat

# ============================
# Data loading and preparation
# ============================

# Load data
df_train = pd.read_pickle('data/prepared/train.pkl')

# Features
features = ['age',
            'breed1',
            'breed2',
            'gender',
            'color1',
            'color2',
            'color3',
            'maturitysize',
            'furlength',
            'vaccinated',
            'dewormed',
            'sterilized',
            'health',
            'quantity',
            'fee',
            'state',
            'videoamt',
            'photoamt',
            'description_length',
            'sentiment_magnitude',
            'sentiment_score']

# Target
target = 'adoptionspeed_bin'

# Make binary target
df_train.loc[:, target] = np.where(df_train.loc[:, 'adoptionspeed'] <= 2, 1, 0)

# Build feature matrix and labels
y = df_train.loc[:, target].ravel()
X = df_train.drop(target, axis=1)
X = X.loc[:, features]

# Split train, test
X_train, X_test, y_train, y_test = train_test_split(X, y)


# =================
# Fit linear model
# =================

log_best = LogisticRegression(C=1e8, solver='lbfgs').fit(X_train, y_train)

log_p = log_best.predict_proba(X_test)

log_auc_score = roc_auc_score(y_test, log_p[:, 1])


# =================
# Fit XGBoost model
# =================

params = {'n_estimators': [100, 200, 300, 500],
          'max_depth': [1, 3, 6, 9, 12],
          'reg_lambda': [0.0, 0.01, 0.001],
          'objective': ['binary:logistic'],
          'silent': 1}

xgb_best, xgb_params_best, xgb_cv_result = cv(X_train,
                                              y_train,
                                              n_folds=5,
                                              model_class=ModelXGB(),
                                              eval_fn=log_loss,
                                              params=params)

xgb_p = xgb_best.predict(xgb.DMatrix(X_test))

xgb_auc_score = roc_auc_score(y_test, xgb_p)


# =================
# Fit LightGM model
# =================

params = {'num_iterations': [100, 200, 300, 500],
          'num_leaves': list(np.power(2, [1, 3, 6, 9, 12])),
          'reg_lambda': [0.0, 0.01, 0.001],
          'objective': 'binary',
          'verbose': -1}

lgb_best, lgb_params_best, lgb_cv_result = cv(X_train,
                                              y_train,
                                              n_folds=5,
                                              model_class=ModelLGB(),
                                              eval_fn=log_loss,
                                              params=params)

lgb_p = lgb_best.predict(X_test)

lgb_auc_score = roc_auc_score(y_test, lgb_p)


# =================
# Fit CatBoost model
# =================

params = {'iterations': [100, 200, 300],
          'depth': [1, 3, 6, 9],
          'l2_leaf_reg': [0.0, 0.01, 0.001],
          'loss_function': 'Logloss'}

cat_best, cat_params_best, cat_cv_result = cv(X_train,
                                              y_train,
                                              n_folds=5,
                                              model_class=ModelCat(),
                                              eval_fn=log_loss,
                                              params=params)

cat_p = cat_best.predict(cat.Pool(X_test))

cat_auc_score = roc_auc_score(y_test, cat_p)

# ==============
# Compare models
# ==============

print('AUC logistic regression: {}'.format(round(log_auc_score, 3)))
print('AUC XGBoost: {}'.format(round(xgb_auc_score, 3)))
print('AUC LightGBM: {}'.format(round(lgb_auc_score, 3)))
print('AUC CatBoost: {}'.format(round(cat_auc_score, 3)))

