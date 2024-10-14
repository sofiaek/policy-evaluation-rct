"""
Learn p(s|x) using logistic regression.
"""
import logging
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression


class LearnWeightsLogisticRct:
    def __init__(
        self, df_rct, df_test, get_weight_decision_rct, x_name=["x"], s_name="s"
    ):
        self.get_weight_decision_rct = get_weight_decision_rct

        self.x_name = x_name
        self.s_name = s_name

        df = pd.concat([df_rct, df_test])
        self.model_sampling = self.train_sampling_model(df)

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

    def train_sampling_model(self, df):
        logging.debug("Training model in LearnWeightsLogistic")
        x = df[self.x_name].to_numpy()
        s = df[self.s_name]

        clf = LogisticRegression(class_weight="balanced").fit(x, s)
        return clf
