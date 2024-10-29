import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import logging
import os

import numpy as np
import argparse
import plotting
import save_utils

from numpy.random import default_rng
from sklearn.model_selection import train_test_split
from generate_data_policy import DataGenPolicy
from robust_policy import RobustPolicyCurve
from learn_weights_logistic import LearnWeightsLogisticRct
from learn_weights_xgboost import LearnWeightsXgboostRct

from matplotlib import pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "lines.linewidth": 2
})


def test_loss(loss_list, loss_test):
    n = np.zeros(len(loss_list))
    loss_arr = np.array(loss_list)
    for loss_i in loss_test:
        n[loss_i > loss_arr] += 1

    alpha_test = n / len(loss_test)
    return np.array(alpha_test)


def create_parser():
    """ """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=2, help="random seed (default: %(default)s)"
    )
    parser.add_argument(
        "--weight_models",
        default=["logistic", "xgboost"],
        nargs="+",
        choices=["true", "logistic", "xgboost"],
        help="model to estimate the weights",
    )
    parser.add_argument(
        "--decision",
        type=int,
        default=[1],
        nargs="+",
        help="the policies to compare (default: %(default)s)",
    )
    parser.add_argument(
        "--gamma_list",
        type=float,
        default=[1.0, 1.2, 2.0],
        nargs="+",
        help="list of gamma (default: %(default)s)",
    )
    parser.add_argument("--y_max", type=float, default=100, help="maximum y value")
    parser.add_argument(
        "--name", default="", help="name of experiment (default: %(default)s)"
    )
    parser.add_argument(
        "--skip_x", default="", help="do not use covariate (default: %(default)s)"
    )

    parser.add_argument(
        "--n_mc",
        type=int,
        default=1,
        help="number of monte carlo simulations (default: %(default)s)",
    )

    # add model specific args
    parser = save_utils.add_specific_args(parser)

    return parser


def run_experiment(i, args, out_dir, a, quant_arr, coverage_rct, coverage_rct_no_shift):
    rng = default_rng(args.seed + i)

    data_gen = DataGenPolicy(args, rng)
    df, x_name = data_gen.get_rct_data(args.n_train_rct)

    if args.skip_x in x_name:
        if args.weight_models == ["true"]:
            raise ValueError("Cannot skip covariate when using true weights")
        x_name.remove(args.skip_x)

    df_rct, df_sampling = train_test_split(
        df, train_size=0.5, random_state=rng.integers(10000)
    )

    df_rct_loss = df_rct[df_rct["a"] == a]
    df_rct_weight = df_rct[df_rct["a"] == 1 - a]

    df_rwd = data_gen.get_test_data(args.n_train_rwd, a)
    df_test = data_gen.get_test_data(args.n_test, a)

    weight_func_list = []
    name_list = []

    for weight_model in args.weight_models:
        if weight_model == "logistic":
            weight_func_list += [
                LearnWeightsLogisticRct(
                    df_sampling, df_rwd, data_gen.get_weight_decision_rct, x_name
                )
            ]
            name_list += ["Logistic"]
        elif weight_model == "xgboost":
            weight_func_list += [
                LearnWeightsXgboostRct(
                    df_sampling,
                    df_rwd,
                    data_gen.get_weight_decision_rct,
                    x_name,
                    use_cv=False,
                )
            ]
            name_list += ["XGBoost"]
        elif weight_model == "true":
            weight_func_list += [data_gen]
            name_list += ["True"]

    if i == 0 and len(args.decision) == 1:
        gamma = args.gamma_list[-1]

        plotting.save_calibration_curves(
            out_dir, df_rct, df_rwd, x_name, weight_func_list, name_list, "s", args.name
        )

        if args.skip_x == "":
            plotting.plot_odds_dist(
                out_dir,
                df_rct,
                df_rwd,
                gamma,
                x_name,
                weight_func_list,
                data_gen.get_weight_sampling,
                name_list,
                args.name,
            )

            plotting.plot_weight_distribution(
                out_dir, df_rct, weight_func_list, name_list, x_name, True, args.name
            )
            plotting.plot_weight_distribution(
                out_dir,
                df_rct,
                weight_func_list,
                name_list,
                x_name,
                False,
                "{}_appx".format(args.name),
            )

    loss_list_rct = []
    alpha_list_rct = []

    for i_weight, weight_func in enumerate(weight_func_list):
        conformal_rct = RobustPolicyCurve(args.y_max, weight_func.get_weight, x_name)
        if name_list[i_weight] == "Logistic":
            loss_base, alpha_base, loss_array_base = conformal_rct.get_quantiles_policy(
                quant_arr, df_rct, a
            )

        logging.debug("Results:")
        for i_gamma, gamma in enumerate(args.gamma_list):
            loss_n1, alpha_n1, loss_array = conformal_rct.get_robust_quantiles_policy(
                quant_arr, df_rct_loss, df_rct_weight, gamma, a
            )
            loss_list_rct += [loss_n1]
            alpha_list_rct += [alpha_n1]
            coverage_rct[i_weight][i_gamma] += test_loss(
                loss_array, df_test["y"].to_numpy()
            )
            logging.info(
                "Gamma = {}, coverage = {}".format(
                    gamma, coverage_rct[i_weight][i_gamma] / (i + 1)
                )
            )

    loss_rct, alpha_rct, loss_array = conformal_rct.get_quantiles_rct(
        quant_arr, df_rct, a
    )
    coverage_rct_no_shift[0] += (
        test_loss(loss_array, df_test["y"].to_numpy()) / args.n_mc
    )
    loss_list_rct += [loss_rct]
    alpha_list_rct += [alpha_rct]
    benchmark = False
    try:
        loss_base, alpha_base, loss_array_base
    except NameError:
        var_exists = False
    else:
        coverage_rct_no_shift[1] += (
            test_loss(loss_array_base, df_test["y"].to_numpy()) / args.n_mc
        )
        loss_list_rct += [loss_base]
        alpha_list_rct += [alpha_base]
        benchmark = True

    return (
        loss_list_rct,
        alpha_list_rct,
        coverage_rct,
        coverage_rct_no_shift,
        name_list,
        benchmark,
    )


def main():
    parser = create_parser()
    parser = DataGenPolicy.add_model_specific_args(parser)
    args = parser.parse_args()

    out_dir = save_utils.save_logging_and_setup(args)

    quant_arr = np.linspace(0.05, 0.95, 19)

    coverage_rct = [
        [np.zeros(len(quant_arr)) for _ in args.gamma_list] for _ in args.weight_models
    ]
    coverage_rct_no_shift = [np.zeros(len(quant_arr)) for _ in range(2)]

    name_list = []
    for a in args.decision:
        for i in range(args.n_mc):
            if i % 50 == 1:
                print(i)
            (
                loss_list_rct,
                alpha_list_rct,
                coverage_rct,
                coverage_rct_no_shift,
                name_list,
                benchmark,
            ) = run_experiment(
                i, args, out_dir, a, quant_arr, coverage_rct, coverage_rct_no_shift
            )

            if i == 0:
                np.savez(
                    os.path.join(out_dir, "loss_list_rct_{}".format(a)), *loss_list_rct
                )
                np.savez(
                    os.path.join(out_dir, "alpha_list_rct_{}".format(a)),
                    *alpha_list_rct
                )

        coverage_rct = [
            [coverage_rct[j][i] / args.n_mc for i in range(len(args.gamma_list))]
            for j in range(len(args.weight_models))
        ]

    np.save(os.path.join(out_dir, "coverage_rct"), coverage_rct)
    np.save(os.path.join(out_dir, "coverage_rct_no_shift"), coverage_rct_no_shift)
    np.save(os.path.join(out_dir, "quant_arr"), quant_arr)
    np.save(os.path.join(out_dir, "name_list"), name_list)


if __name__ == "__main__":
    main()
