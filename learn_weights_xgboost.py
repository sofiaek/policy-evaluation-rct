"""
Learn p(s|x) using xgboost
"""
import logging
import numpy as np
import pandas as pd

from xgboost import XGBClassifier


class LearnWeightsXgboostRct:
    def __init__(
        self,
        df_rct,
        df_test,
        get_weight_decision_rct,
        x_name=["x"],
        s_name="s",
        use_cv=True,
        is_nhanes=False,
    ):
        self.get_weight_decision_rct = get_weight_decision_rct

        self.x_name = x_name
        self.s_name = s_name

        df = pd.concat([df_rct, df_test])
        s = df[self.s_name]
        self.weight = len(s[s == 0]) / len(s[s == 1])

        if use_cv:
            self.model_sampling = self.tune_xgboost(df)
        else:
            self.model_sampling = self.train_sampling_model(df, is_nhanes)

    def get_p_s_x(self, s, x):
        proba = self.model_sampling.predict_proba(x)
        return proba[np.arange(0, len(x)), s]

    def get_weight_sampling(self, x):
        p_s1_x = self.get_p_s_x(1, x)
        p_s0_x = self.get_p_s_x(0, x)
        weight = p_s1_x / p_s0_x
        return weight

    def get_weight(self, x, a, a_i):
        return self.get_weight_sampling(x) * self.get_weight_decision_rct(
            x=x, a=a, a_i=a_i
        )

    def train_sampling_model(self, df, is_nhanes):
        logging.debug("Training model in LearnWeightsLogistic")
        x = df[self.x_name].to_numpy()
        s = df[self.s_name]
        if is_nhanes:
            params = {
                "objective": "binary:logistic",
                "scale_pos_weight": self.weight,
                "subsample": 0.4,
                "n_estimators": 100,
                "min_child_weight": 1,
                "gamma": 0,
                "max_depth": 1,
                "learning_rate": 0.1,
                "colsample_bytree": 0.4,
            }

        else:
            params = {
                "objective": "binary:logistic",
                "subsample": 0.6,
                "scale_pos_weight": self.weight,
                "n_estimators": 100,
                "min_child_weight": 1,
                "max_depth": 2,
                "learning_rate": 0.05,
                "colsample_bytree": 0.8,
                "colsample_bylevel": 0.4,
            }

        clf = XGBClassifier()
        clf.set_params(**params)
        clf.fit(x, s)

        return clf

    def tune_xgboost(self, df):
        from xgboost import XGBClassifier
        from sklearn.model_selection import RandomizedSearchCV

        x = df[self.x_name].to_numpy()
        s = df[self.s_name]
        weight = len(s[s == 0]) / len(s[s == 1])

        clf_xgb = XGBClassifier(objective="binary:logistic")

        param_dist = {
            "learning_rate": [0.05, 0.1],
            "max_depth": [1, 2, 3],
            "subsample": np.arange(0.2, 0.7, 0.1),
            "colsample_bytree": np.arange(0.2, 0.7, 0.1),
            "n_estimators": [50, 100, 200],
            "min_child_weight": [1],
            "scale_pos_weight": [weight],
        }

        clf = RandomizedSearchCV(
            clf_xgb,
            param_distributions=param_dist,
            n_iter=200,
            scoring="log_loss",
            error_score=0,
            verbose=0,
            n_jobs=-1,
            random_state=42,
        )

        search = clf.fit(x, s)
        print("XGB result")
        print(search.best_params_)
        print(search.best_score_)

        return search.best_estimator_
