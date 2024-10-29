"""
Test coverage.

"""

import os
import logging
import numpy as np
import pandas as pd
import pyreadr
import save_utils
import plotting

from numpy.random import default_rng
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from main_policy import create_parser

from learn_weights_logistic import LearnWeightsLogisticRct
from learn_weights_xgboost import LearnWeightsXgboostRct
from robust_policy import RobustPolicyCurve

from matplotlib import pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "lines.linewidth": 2
})


#  python main_nhanes.py --weight_models xgboost --gamma_list 1 1.5 2 --y_max 20 --decision 0 1 --name nhanes


def get_weight(x, a, a_i):
    weight_decision = 1
    weight_sampling = np.where(x > 35, (1 - 0.2) / 0.2, (1 - 0.8) / 0.8).reshape(-1)
    weight_sampling = np.where(a != a_i, 0, weight_sampling)
    return weight_sampling * weight_decision


def get_weight_ones(**kwargs):
    weight = np.where(kwargs["a"] != kwargs["a_i"], 0, 1)
    return weight


def test_loss(loss_list, loss_test):
    n = np.zeros(len(loss_list))
    loss_arr = np.array(loss_list)
    for loss_i in loss_test:
        n[loss_i > loss_arr] += 1

    alpha_test = n / len(loss_test)
    return np.array(alpha_test)


def read_and_prepare_data(args, rng, remove=""):
    # Read data
    result = pyreadr.read_r("data/nhanes.fish.rda")
    df_fish = result["nhanes.fish"]

    x_name_orig = [
        "gender",
        "age",
        "income",
        "income.missing",
        "race",
        "education",
        "smoking.ever",
        "smoking.now",
    ]

    df = df_fish.dropna()

    # select categorical and numerical features
    num_ix = ["age", "income", "smoking.now"]
    cat_ix = ["gender", "income.missing", "smoking.ever", "education"]

    # df["income"] = np.log(df["income"] + 1)

    # Create synthetic
    y_name = "o.LBXTHG"
    steps = [
        ("one_hot", OneHotEncoder(handle_unknown="ignore"), cat_ix),
        ("min_max", MinMaxScaler(), num_ix),
    ]
    ct = ColumnTransformer(steps)
    model = GradientBoostingRegressor(n_estimators=100)
    pipe = Pipeline(steps=[("t", ct), ("m", model)])

    pipe.fit(
        df[x_name_orig][df["fish.level"] == "high"],
        df[y_name][df["fish.level"] == "high"],
    )
    df["y1"] = pipe.predict(df[x_name_orig])
    df["y1"] = np.where(df["y1"] < 0, 0, df["y1"])

    pipe.fit(
        df[x_name_orig][df["fish.level"] == "low"],
        df[y_name][df["fish.level"] == "low"],
    )
    df["y0"] = pipe.predict(df[x_name_orig])

    scaler = StandardScaler()
    df.loc[:, num_ix] = scaler.fit_transform(df[num_ix])

    # Set a
    df["a"] = rng.binomial(1, np.ones(len(df)) * 0.5)
    df["y"] = np.where(df["a"] == 0, df["y0"], df["y1"])
    y_name = "y"

    # Split the data set in RCT and RWD
    p_age = 1 / (1 + np.exp(-df["age"]))
    p_income = 1 / (1 + np.exp(-df["income"]))

    p = (
        0.25 * p_age
        + 0.25 * p_income
        + 0.25 * np.where(df["gender"] == 1, 1 - 0.2, 1 - 0.4)
        + 0.25 * np.where(df["smoking.ever"] == 1, 1 - 0.2, 1 - 0.4)
    )
    df["s"] = rng.binomial(1, p)

    x_names_true = ["age", "income", "gender_2", "smoking.ever_True"]

    # Transform x-categorical
    df_one_hot = pd.get_dummies(df[cat_ix], columns=cat_ix, drop_first=True)
    df = pd.concat([df, df_one_hot], axis=1)

    cat_ix.remove("smoking.ever")
    if remove in num_ix:
        num_ix.remove(remove)
    if remove in cat_ix:
        cat_ix.remove(remove)
    df_one_hot = pd.get_dummies(df[cat_ix], columns=cat_ix, drop_first=True)
    x_names = num_ix + df_one_hot.columns.tolist()

    df_rct = df[df["s"] == 0]
    df_test = df[df["s"] == 1]

    return df_rct, df_test, x_names, x_names_true, y_name


def main(read_and_prepare_data, get_weight, xlim_list):
    parser = create_parser()
    args = parser.parse_args()

    out_dir = save_utils.save_logging_and_setup(args)

    quant_arr = np.linspace(0.05, 0.95, 19)
    rng = default_rng(args.seed)

    df_rct, df_rwd, x_names, x_names_true, y_name = read_and_prepare_data(
        args, rng, args.skip_x
    )

    coverage_rct = [np.zeros(len(quant_arr)) for _ in args.gamma_list]

    weight_func_list = []
    name_list = []
    for weight_model in args.weight_models:
        if weight_model == "logistic":
            weight_func_list += [
                LearnWeightsLogisticRct(df_rct, df_rwd, get_weight_ones, x_names)
            ]
            name_list += ["Logistic"]
        elif weight_model == "xgboost":
            weight_func_list += [
                LearnWeightsXgboostRct(
                    df_rct,
                    df_rwd,
                    get_weight_ones,
                    x_names,
                    use_cv=False,
                    is_nhanes=True,
                )
            ]
            name_list += ["XGBoost"]

    conformal_rct = RobustPolicyCurve(
        args.y_max, weight_func_list[0].get_weight, x_names, y_name
    )
    conformal_rct_true = RobustPolicyCurve(args.y_max, get_weight, x_names_true, y_name)

    plotting.save_calibration_curves(
        out_dir, df_rct, df_rwd, x_names, weight_func_list, name_list, "s", args.name
    )

    for a_i in [0, 1]:
        loss_list_rct = []
        alpha_list_rct = []

        df_rct_loss = df_rct[df_rct["a"] == a_i]
        df_rct_weight = df_rct[df_rct["a"] == 1 - a_i]

        logging.debug("Results:")
        for i_gamma, gamma in enumerate(args.gamma_list):
            loss_n1, alpha_n1, loss_array = conformal_rct.get_robust_quantiles_policy(
                quant_arr, df_rct_loss, df_rct_weight, gamma, a_i
            )
            loss_list_rct += [loss_n1]
            alpha_list_rct += [alpha_n1]
            coverage_rct[i_gamma] += test_loss(loss_array, df_rwd[y_name].to_numpy())
            logging.info(
                "Gamma = {}, coverage = {}".format(gamma, coverage_rct[i_gamma])
            )

        loss_rct = np.sort(df_rct[df_rct["a"] == a_i]["y"].to_numpy())
        alpha_rct = np.cumsum(1 / (len(loss_rct)) * np.ones(len(loss_rct)))
        loss_list_rct += [loss_rct]
        alpha_list_rct += [alpha_rct[::-1]]

        loss_rwd = np.sort(df_rwd[df_rwd["a"] == a_i]["y"].to_numpy())
        alpha_rwd = np.cumsum(1 / (len(loss_rwd)) * np.ones(len(loss_rwd)))
        loss_list_rct += [loss_rwd]
        alpha_list_rct += [alpha_rwd[::-1]]

        np.savez(os.path.join(out_dir, "loss_list_rct_{}".format(a_i)), *loss_list_rct)
        np.savez(
            os.path.join(out_dir, "alpha_list_rct_{}".format(a_i)), *alpha_list_rct
        )
        np.save(os.path.join(out_dir, "coverage_rct_{}".format(a_i)), coverage_rct)

    name = args.name
    plotting.plot_loss(out_dir, out_dir, [0], "{}0".format(name), xlim_list)
    plotting.plot_loss(out_dir, out_dir, [1], "{}1".format(name), xlim_list)

    plotting.plot_loss(out_dir, out_dir, [0, 1], "{}".format(name), xlim_list)
    plotting.plot_loss_compare(out_dir, out_dir, xlim_list[-1])


if __name__ == "__main__":
    xlim_list = [[0, 2], [0, 16]]
    main(read_and_prepare_data, get_weight, xlim_list)
