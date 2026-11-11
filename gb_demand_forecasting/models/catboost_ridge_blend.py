#import the necessary libraries
import numpy as np

from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

TRUSTED_TYPES = [
    "gb_demand_forecasting.models.catboost_ridge_blend.CatBoostRidgeBlend",
    "catboost.core.CatBoostRegressor",
    "numpy.dtype",
]
CATBOOST_DEFAULTS = dict(iterations=5600, learning_rate=0.05, depth=4, l2_leaf_reg=3,
                         rsm=0.7, random_seed=42, verbose=0, allow_writing_files=False)


class CatBoostRidgeBlend(RegressorMixin,BaseEstimator):
    def __init__(self, cat_features, ridge_weight=0.25, ridge_alpha=10.0,
                 baseline_col="prev_6weeks_lag_noholiday", catboost_params=None):
        self.cat_features = cat_features
        self.ridge_weight = ridge_weight
        self.ridge_alpha = ridge_alpha
        self.baseline_col = baseline_col
        self.catboost_params = catboost_params

    def _as_text(self, X):
        X = X.copy()
        for column in self.cat_features:
            X[column] = X[column].astype(str)
        return X

    def fit(self, X, y, sample_weight=None):
        text = self._as_text(X)
        params = {**CATBOOST_DEFAULTS, **(self.catboost_params or {})}
        self.catboost_ = CatBoostRegressor(**params).fit(
            text, y, sample_weight=sample_weight, cat_features=list(self.cat_features))
        prep = ColumnTransformer(
            [("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
              list(self.cat_features))],
            remainder=SimpleImputer(strategy="median"))
        self.ridge_ = Pipeline([("prep", prep), ("scale", StandardScaler()),
                                ("model", Ridge(alpha=self.ridge_alpha))])
        self.ridge_.fit(text, np.asarray(y) - X[self.baseline_col].to_numpy(),
                        model__sample_weight=sample_weight)
        return self

    def predict(self, X):
        text = self._as_text(X)
        catboost = self.catboost_.predict(text)
        ridge = self.ridge_.predict(text) + X[self.baseline_col].to_numpy()
        return (1 - self.ridge_weight) * catboost + self.ridge_weight * ridge


