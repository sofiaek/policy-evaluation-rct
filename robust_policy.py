"""
Can handle model errors of the weights on the form:
    1/gamma < weight(x)/weight_est(x) < gamma
"""

import numpy as np


class RobustPolicyCurve:
    def __init__(self, loss_max, get_weight, x_name=["x"], y_name="y"):
        self.loss_max = loss_max
        self.func_get_weight = get_weight
        self.x_name = x_name
        self.y_name = y_name

    def get_score_one_sided(self, df):
        y = df[self.y_name].to_numpy()
        score = y
        return score

    def get_weight_bounds(self, x, a, a_i, gamma):
        weight = self.func_get_weight(x, a, a_i)
        weight_low = 1 / gamma * weight
        weight_high = gamma * weight
        return weight_low, weight_high

    def get_robust_quantiles_policy(self, quant_list, df, df_beta, gamma, a_i):
        # Calculate one-sided score
        loss = self.get_score_one_sided(df)

        # Extract necessary data from the main dataframe
        x = df[self.x_name].to_numpy()
        a = df["a"].to_numpy()

        # Get weight bounds based on the main dataframe
        weight_low, weight_high = self.get_weight_bounds(x, a, a_i, gamma)

        # Extract necessary data from the beta dataframe
        x_beta = df_beta[self.x_name].to_numpy()
        a_beta = df_beta["a"].to_numpy()

        # Get weight bounds based on the beta dataframe
        _, weight_high_beta = self.get_weight_bounds(x_beta, a_i, a_i, gamma)

        # Filter out zero weights
        non_zero_indices = weight_low != 0
        loss = loss[non_zero_indices]
        weight_high = weight_high[non_zero_indices]
        weight_low = weight_low[non_zero_indices]

        # Sort arrays based on the loss
        sort_idx = np.argsort(loss)
        sort_idx = np.append(sort_idx, 0)  # Dummy, to avoid append later
        loss_n1 = loss[sort_idx]
        weight_low = weight_low[sort_idx]
        weight_high = weight_high[sort_idx]
        weight_high[-1] = 0
        weight_low[-1] = 0

        # Calculate cumulative weights
        weight_cum_low = np.cumsum(weight_low)
        weight_cum_high = np.cumsum(weight_high[::-1])[::-1]

        # Set last element to maximum loss
        loss_n1[-1] = self.loss_max

        # Get weights and cumulative weights for beta values
        beta_n1, weight_beta_n1 = self.get_weight_beta(weight_high_beta)
        idx_beta = 0
        alpha_n1 = np.ones(len(loss_n1))

        # Calculate cumulative weights with beta values
        weight_cum_0 = weight_cum_low + np.append(weight_cum_high[1:], 0)
        weight_cum_low_end = weight_cum_low[-1]

        # Iterate over beta values
        while (
            idx_beta < 3
        ):  # len(beta_n1)/2: # and beta_n1[idx_beta] <= 1 - np.min(quant_list):
            weight_inf = weight_beta_n1[idx_beta]

            weight_cum_n1 = weight_cum_0 + weight_inf
            weight_cum_low[-1] = weight_cum_low_end + weight_inf

            F_hat_n1 = weight_cum_low / weight_cum_n1
            beta = beta_n1[idx_beta]
            alpha_temp = 1 - (1 - beta) * F_hat_n1
            alpha_temp = np.where(beta < alpha_temp, alpha_temp, 1)
            alpha_n1 = np.where(alpha_temp < alpha_n1, alpha_temp, alpha_n1)
            idx_beta += 1

        # Set values based on conditions
        alpha_n1 = np.where(alpha_n1 == 1, 0, alpha_n1)
        loss_n1 = np.where(alpha_n1 == 0, self.loss_max, loss_n1)

        # Initialize array for quantiles
        loss_array = np.full(len(quant_list), self.loss_max)

        # Iterate over quantiles to find corresponding losses
        for m, quant_alpha in enumerate(quant_list):
            idx = np.argmax(alpha_n1 < quant_alpha)
            if idx == 0:
                loss_array[m] = self.loss_max
            else:
                loss_array[m] = loss_n1[idx - 1]

        return loss_n1, alpha_n1, loss_array

    def get_quantiles_policy(self, quant_list, df, a_i):
        loss = self.get_score_one_sided(df)
        x = df[self.x_name].to_numpy()
        a = df["a"].to_numpy()
        weight, _ = self.get_weight_bounds(x, a, a_i, gamma=1)

        non_zero_indices = weight != 0
        loss = loss[non_zero_indices]
        weight = weight[non_zero_indices]

        sort_idx = np.argsort(loss)
        loss = loss[sort_idx]
        weight = weight[sort_idx]

        alpha = np.cumsum(weight[::-1])[::-1] / np.sum(weight)

        loss_array = np.full(len(quant_list), self.loss_max)

        for m, quant_alpha in enumerate(quant_list):
            idx = np.argmax(alpha < quant_alpha)
            if idx == 0:
                loss_array[m] = self.loss_max
            else:
                loss_array[m] = loss[idx - 1]

        return loss, alpha, loss_array

    def get_quantiles_rct(self, quant_list, df_all, a_i):
        loss_rct = np.sort(df_all[df_all["a"] == a_i]["y"].to_numpy())
        alpha_rct = np.cumsum(1 / len(loss_rct) * np.ones(len(loss_rct)))[::-1]
        loss_array = np.full(len(quant_list), self.loss_max)

        for m, quant_alpha in enumerate(quant_list):
            idx = np.argmax(alpha_rct < quant_alpha)
            if idx == 0:
                loss_array[m] = self.loss_max
            else:
                loss_array[m] = loss_rct[idx - 1]

        return loss_rct, alpha_rct, loss_array

    def get_mean_policy(self, df, a_i):
        y = df[df["a"] == a_i]["y"].to_numpy()
        weight = self.func_get_weight(
            df[self.x_name].to_numpy(), df["a"].to_numpy(), a_i
        )
        weight = weight[df["a"] == a_i]
        return np.sum(y * weight) / np.sum(weight)

    @staticmethod
    def get_weight_beta(weight_all):
        n = len(weight_all)
        weight_n1 = 1 / (n + 1) * np.ones(n + 1)
        alpha_n1 = np.cumsum(weight_n1)
        weight_n1 = np.append(weight_all, 1000)
        weight_n1 = np.sort(weight_n1)[::-1]
        return alpha_n1, weight_n1
