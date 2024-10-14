import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scipy.stats import multivariate_normal


class DataGenPolicy:
    def __init__(self, args, rng):
        self.rng = rng

        self.mu_y = args.mu_y
        self.sigma_y = args.sigma_y

        self.rv_s0 = multivariate_normal(
            np.zeros(len(args.mu_x_s1)), np.diag(np.ones(len(args.mu_x_s1))), seed=rng
        )
        self.rv_s1 = multivariate_normal(
            args.mu_x_s1, np.diag(args.sigma_x_s1), seed=rng
        )

        self.rv_u_s0 = multivariate_normal(
            np.zeros(len(args.mu_u_s1)),
            np.diag(np.ones(len(args.sigma_u_s1))),
            seed=rng,
        )
        self.rv_u_s1 = multivariate_normal(
            args.mu_u_s1, np.diag(args.sigma_u_s1), seed=rng
        )

    def get_x_s(self, n, s):
        x = 0
        if s == 0:
            x = self.rv_s0.rvs(size=n)
        elif s == 1:
            x = self.rv_s1.rvs(size=n)
        return x

    def get_u_s(self, n, s):
        u = 0
        if s == 0:
            u = self.rv_u_s0.rvs(size=n).reshape(n, -1)
        elif s == 1:
            u = self.rv_u_s1.rvs(size=n).reshape(n, -1)
        return u

    def get_random_a(self, n):
        a = self.rng.integers(2, size=n)
        return a

    @staticmethod
    def get_p_ax_policy(n, a_i):
        return a_i * np.ones(n)

    def get_p_s_x(self, s, x):
        if s == 0:
            return self.rv_s0.pdf(x).reshape(-1) / (
                self.rv_s0.pdf(x).reshape(-1) + self.rv_s1.pdf(x).reshape(-1)
            )
        elif s == 1:
            return self.rv_s1.pdf(x).reshape(-1) / (
                self.rv_s0.pdf(x).reshape(-1) + self.rv_s1.pdf(x).reshape(-1)
            )

    def get_weight_sampling(self, x):
        p_s0_x = self.rv_s0.pdf(x)
        p_s1_x = self.rv_s1.pdf(x)
        weight = p_s1_x / p_s0_x
        return weight.reshape(-1)

    @staticmethod
    def get_weight_decision_rct(**kwargs):
        p_a_x = 0.5
        p_policy_a_x = np.where(kwargs["a"] == kwargs["a_i"], 1, 0)
        weight = p_policy_a_x / p_a_x
        return weight.reshape(-1)

    def get_weight(self, x, a, a_i):
        return self.get_weight_sampling(x) * self.get_weight_decision_rct(a=a, a_i=a_i)

    def get_rct_data(self, n):
        df, x_names = self.__generate_data(n, s=0)
        return df, x_names

    def get_test_data(self, n, a):
        df, __ = self.__generate_data(n, s=1, a=a)
        return df

    def __generate_data(self, n, s, **kwargs):
        x = self.get_x_s(n, s)
        u = self.get_u_s(n, s)
        a = 0
        if s == 0:
            a = self.get_random_a(n)
        elif s == 1:
            p_ax = self.get_p_ax_policy(n, kwargs["a"])
            a = self.rng.binomial(1, p_ax)

        noise = self.rng.normal(self.mu_y, self.sigma_y, n)
        y = np.where(
            a == 0, 1 + x[:, 1] + noise, x[:, 0] ** 2 + x[:, 1] + u[:, 0] + noise
        )
        df = pd.DataFrame({"y": y, "s": s, "a": a})
        x_names = []
        for i in range(x.shape[1]):
            df["X_{}".format(i)] = x[:, i]
            x_names += ["X_{}".format(i)]

        for i in range(u.shape[1]):
            df["U_{}".format(i)] = u[:, i]

        return df, x_names

    @staticmethod
    def plot_data(df_s0, df_s1):
        color_scatter = ["darkseagreen", "palevioletred"]
        fig = plt.figure()
        plt.scatter(df_s0["x0"], df_s0["x1"], s=1, c=color_scatter[1])
        plt.scatter(df_s1["x0"], df_s1["x1"], s=1, c=color_scatter[0])
        plt.legend(["s0", "s1"])
        return fig

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DataGen")

        parser.add_argument(
            "--mu_x_s1",
            type=float,
            default=[0, 0],
            nargs="+",
            help="covariate mean for the test data",
        )
        parser.add_argument(
            "--sigma_x_s1",
            type=float,
            default=[1, 1],
            nargs="+",
            help="covariate sigma for the test data",
        )

        parser.add_argument(
            "--mu_u_s1",
            type=float,
            default=[0, 0],
            nargs="+",
            help="covariate mean for the test data",
        )
        parser.add_argument(
            "--sigma_u_s1",
            type=float,
            default=[1, 1],
            nargs="+",
            help="covariate sigma for the test data",
        )

        parser.add_argument(
            "--mu_y", type=float, default=0, help="noise mean for the outcome"
        )
        parser.add_argument(
            "--sigma_y", type=float, default=1, help="noise sigma for the outcome"
        )

        parser.add_argument(
            "--n_train_rct",
            type=int,
            default=1000,
            help="number of samples for training (default: %(default)s)",
        )
        parser.add_argument(
            "--n_train_rwd",
            type=int,
            default=1000,
            help="number of samples for training (default: %(default)s)",
        )
        parser.add_argument(
            "--n_test",
            type=int,
            default=1000,
            help="number of samples for testing (default: %(default)s)",
        )
        return parent_parser
