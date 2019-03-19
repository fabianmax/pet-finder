import catboost as cat

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import cohen_kappa_score, make_scorer

class ModelCat:
    """
    CatBoost Model
    """

    def __init__(self):

        self.classifier = cat.CatBoostClassifier(loss_function='MultiClass')
        self.cv = None
        self.cv_results_ = None
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None

    def grid_search(self, X_train, y_train, param_grid, folds=5, cores=4):

        kappa_scorer = make_scorer(cohen_kappa_score, greater_is_better=True, weights='quadratic')

        self.cv = GridSearchCV(estimator=self.classifier,
                               param_grid=param_grid,
                               scoring=kappa_scorer,
                               cv=StratifiedKFold(n_splits=folds, shuffle=True),
                               n_jobs=cores,
                               verbose=3)

        self.cv.fit(X=X_train, y=y_train)

        self.cv_results_ = self.cv.cv_results_
        self.best_estimator_ = self.cv.best_estimator_
        self.best_params_ = self.cv.best_params_
        self.best_score_ = self.cv.best_score_

    def predict(self, X_test):

        y_hat = self.best_model.predict(X_test)

        return y_hat

