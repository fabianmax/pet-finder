import lightgbm as lgb
import xgboost as xgb
import catboost as cat
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score

from itertools import product
from tqdm import tqdm


def qwk(actual, prediction):
    """
    Quadratic weighted kappa scoring function
    :param actual:
    :param prediction:
    :return:
    """
    return cohen_kappa_score(actual, prediction, weights='quadratic')


def expand_grid(dictionary: dict) -> pd.DataFrame:
    """
    Expand grid (similar to R's expand.grid())
    :param dictionary:
    :return:
    """

    # Make sure each dict value is an iterable list
    dictionary = {key: get_iterable_list(value) for (key, value) in dictionary.items()}

    # Return cartesian product of dict values
    return pd.DataFrame([row for row in product(*dictionary.values())],
                        columns=dictionary.keys())


def cartesian(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    rows = product(df1.iterrows(), df2.iterrows())

    df = pd.DataFrame(left.append(right) for (_, left), (_, right) in rows)
    return df.reset_index(drop=True)


def get_iterable_list(x) -> list:
    """
    Function to turn single object into iterable list
    Checks whether x is list and wraps x into list if not
    :param x:
    :return:
    """
    if isinstance(x, list):
        return x
    else:
        return [x]


class ModelXGB:
    """
    XGBoost model class
    """

    def __init__(self):

        self.mod = None

    def train(self, X_train, y_train, X_valid=None, y_valid=None, params=None, **kwargs):

        train_data = xgb.DMatrix(X_train, y_train)

        watchlist = [(train_data, 'train')]

        if X_valid is not None and y_valid is not None:
            valid_data = xgb.DMatrix(X_valid, y_valid)
            watchlist.append((valid_data, 'valid'))
            stopping_rounds = 10
        else:
            stopping_rounds = None

        self.mod = xgb.train(params=params,
                             dtrain=train_data,
                             evals=watchlist,
                             early_stopping_rounds=stopping_rounds,
                             maximize=False,
                             verbose_eval=0,
                             **kwargs)

        return self.mod

    def predict(self, X_test):

        test_data = xgb.DMatrix(X_test)
        y_hat = self.mod.predict(test_data)

        return y_hat


class ModelLGB:
    """
    LightGBM model class
    """

    def __init__(self):

        self.mod = None

    def train(self, X_train, y_train, X_valid=None, y_valid=None, params=None, **kwargs):

        train_data = lgb.Dataset(X_train, y_train)

        watchlist = [train_data]

        if X_valid is not None and y_valid is not None:
            valid_data = lgb.Dataset(X_valid, y_valid)
            watchlist.append(valid_data)
            stopping_rounds = 10
        else:
            stopping_rounds = None

        self.mod = lgb.train(params=params,
                             train_set=train_data,
                             valid_sets=watchlist,
                             early_stopping_rounds=stopping_rounds,
                             verbose_eval=0,
                             **kwargs)

        return self.mod

    def predict(self, X_test):

        y_hat = self.mod.predict(X_test)
        # Prob to class
        y_hat = np.argmax(y_hat, axis=1)

        return y_hat


class ModelCat:
    """
    CatBoost model class
    """

    def __init__(self):

        self.mod = None

    def train(self, X_train, y_train, X_valid=None, y_valid=None, params=None, **kwargs):

        train_data = cat.Pool(X_train, y_train, cat_features=None)

        watchlist = [train_data]

        if X_valid is not None and y_valid is not None:
            valid_data = cat.Pool(X_valid, y_valid, cat_features=None)
            watchlist.append(valid_data)
            stopping_rounds = 10
        else:
            stopping_rounds = None

        self.mod = cat.CatBoost(params=params)
        self.mod.fit(X=train_data,
                     eval_set=watchlist,
                     early_stopping_rounds=stopping_rounds,
                     verbose=False,
                     **kwargs)

        return self.mod

    def predict(self, X_test):

        test_data = cat.Pool(X_test, cat_features=None)
        y_hat = self.mod.predict(test_data)
        # Prob to class
        y_hat = np.argmax(y_hat, axis=1)

        return y_hat


def cv(X, y, model_class, eval_fn, n_folds=5, params=None, eval_max=True, **kwargs):
    """
    Cross-Validation function including hyperparameter grid search
    :param X: Input training data
    :param y: Input label
    :param model_class: Model object
    :param eval_fn: Evaluation function
    :param n_folds: Number of cv-folds
    :param params: Additional parameters for Model object
    :param eval_max: Bool indicating if eval_fn should be maximized
    :return: tuple
    """

    # Save name of hyper-parameters
    param_names = list(params.keys())

    # Expand param dict to DataFrame
    param_grid = expand_grid(params)
    param_grid['param_set_id'] = param_grid.index
    param_grid['cv_score'] = np.nan

    # Create grid including folds
    param_fold_grid = cartesian(param_grid, pd.DataFrame({'fold_id': np.arange(n_folds).tolist()}))

    # Create fold indices
    skv = StratifiedKFold(n_splits=n_folds)
    folds = skv.split(X, y)

    # Loop over folds
    for i, (train_index, test_index) in enumerate(folds):

        print('Started training on fold ' + str(i+1) + '/' + str(n_folds))

        # Split data in folds
        if isinstance(X, pd.DataFrame):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
        else:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        # Loop over parameters
        with tqdm() as pbar:
            for j, row in param_grid.iterrows():

                # Convert current parameter set to dict
                current_param_set_id = row.param_set_id
                param_candidate_set = row[param_names].to_dict()

                # Train model
                model_class.train(X_train, y_train, params=param_candidate_set, **kwargs)

                # Predict on hold-out
                y_hat = model_class.predict(X_test)

                # Score hold-out and save result
                score = eval_fn(y_test, y_hat)
                idx = (param_fold_grid['param_set_id'] == current_param_set_id) & (param_fold_grid['fold_id'] == i)
                param_fold_grid.loc[idx, 'cv_score'] = score

                # Update  processbar
                pbar.update(1)

    # Aggregate results
    cv_result = param_fold_grid.groupby(param_names).agg({'cv_score': 'mean'}).reset_index()

    # Find best parameters according to cv
    if eval_max:
        best_params = cv_result.iloc[cv_result['cv_score'].idxmax(), :]
    else:
        best_params = cv_result.iloc[cv_result['cv_score'].idxmin(), :]
    best_params = best_params[param_names].to_dict()

    # Fit final model on entire data
    print('Fitting final model')
    best_model = model_class.train(X, y, params=best_params)

    return best_model, best_params, param_fold_grid

