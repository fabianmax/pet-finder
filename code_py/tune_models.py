import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from code_py.define_models import ModelXGB, ModelLGB, ModelCat, cv

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


# XGB test
params = {'n_estimators': [50, 100, 200],
          'max_depth': [1, 3, 6, 9],
          'reg_lambda': [0.0, 0.01, 0.001],
          'objective': ['binary:logistic'],
          'num_class': [5],
          'silent': 1}

xgb_best, xgb_params_best, xgb_cv_result = cv(X_train,
                                              y_train,
                                              n_folds=5,
                                              model_class=ModelXGB(),
                                              eval_fn=log_loss,
                                              params=params)


# LGB Test
params = {'n_estimators': [50, 100, 200],
          'num_leaves': list(np.power(2, [1, 3, 6, 9])),
          'reg_lambda': [0.0, 0.01, 0.001],
          'objective': 'binary',
          'num_class': 5,
          'verbose': -1}

lgb_best, lgb_params_best, lgb_cv_result = cv(X_train,
                                              y_train,
                                              n_folds=5,
                                              model_class=ModelLGB(),
                                              eval_fn=log_loss,
                                              params=params)


# Cat Test
params = {'iterations': [50, 100, 200],
          'depth': [1, 3, 6, 9],
          'l2_leaf_reg': [0.0, 0.01, 0.001],
          'loss_function': 'Logloss'}

cat_best, cat_params_best, cat_cv_result = cv(X_train,
                                              y_train,
                                              n_folds=5,
                                              model_class=ModelCat(),
                                              eval_fn=log_loss,
                                              params=params)


