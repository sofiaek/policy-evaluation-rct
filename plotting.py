# %% Plot result
import os
import json

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as mtick

sns.set(font_scale=1.5, rc={"text.usetex": True})
sns.set_style("white")
plt.rcParams["axes.grid"] = True
levels = 50


def center_subtitles(legend):
    """Centers legend labels with alpha=0"""
    vpackers = legend.findobj(matplotlib.offsetbox.VPacker)
    for vpack in vpackers[:-1]:  # Last vpack will be the title box
        vpack.align = "left"
        for hpack in vpack.get_children():
            draw_area, text_area = hpack.get_children()
            for collection in draw_area.get_children():
                alpha = collection.get_alpha()
                # sizes = collection.get_sizes()
                if alpha == 0:  # or all(sizes == 0):
                    draw_area.set_visible(False)
    return legend


def plot_odds_dist(
    out_dir,
    df_rct,
    df_test,
    gamma,
    x_name,
    weight_list,
    get_weight_sampling_true,
    name_list,
    name="",
):
    from matplotlib import colors

    size = 2
    frac = 0.5

    def get_odds_sampling(pos, weighter):
        x, y, z = pos.shape
        odds_map = np.zeros((x, y))
        for x_i in range(x):
            for y_i in range(y):
                x_test = pos[x_i, y_i, :].reshape(-1, 2)
                odds_map[x_i, y_i] = get_weight_sampling_true(
                    x_test
                ) / weighter.get_weight_sampling(x_test)
        return np.log10(odds_map), np.min(odds_map), np.max(odds_map)

    def get_odds_sampling_x(x, weighter, gamma):
        weight = weighter.get_weight_sampling(x.to_numpy())
        odds = get_weight_sampling_true(x.to_numpy()) / weight
        odds_wrong = np.where(np.logical_and(odds > 1 / gamma, odds < gamma), 0, size)
        odds_ok = np.where(odds_wrong == 0, size, 0)
        return (
            x[np.logical_and(odds > 1 / gamma, odds < gamma)],
            x[np.logical_or(odds <= 1 / gamma, odds >= gamma)],
        )  # odds_wrong, odds_ok

    if "True" in name_list:
        idx = name_list.index("True")
        weight_list = weight_list[:idx] + weight_list[idx + 1 :]
        name_list = name_list[:idx] + name_list[idx + 1 :]

    x, y = np.mgrid[-3:3.1:0.1, -3:3.1:0.1]
    pos = np.dstack((x, y))

    n_subplots = len(weight_list)
    fig, ax = plt.subplots(1, n_subplots, figsize=(4 * n_subplots, 4), squeeze=False)
    ax = ax[0]

    weights_maps = []
    min_weight = -2
    max_weight = 2

    for i in range(n_subplots):
        weight_map_i, min_weight_i, max_weight_i = get_odds_sampling(
            pos, weight_list[i]
        )
        weights_maps += [weight_map_i]

    df_plot_rct = df_rct.sample(frac=frac, random_state=1)
    norm = colors.TwoSlopeNorm(vmin=min_weight, vcenter=0.0, vmax=max_weight)
    for i in range(n_subplots):
        cnt = ax[i].contourf(
            x, y, weights_maps[i], levels=levels, norm=norm, cmap="seismic"
        )

        for c in cnt.collections:
            c.set_edgecolor("face")

        df_right, df_wrong = get_odds_sampling_x(
            df_plot_rct[x_name], weight_list[i], gamma
        )
        ax[i].scatter(df_wrong[x_name[0]], df_wrong[x_name[1]], s=size, c="black")
        ax[i].scatter(
            df_right[x_name[0]], df_right[x_name[1]], s=size, c="mediumseagreen"
        )

        ax[i].axis([-3, 3, -3, 3])
        ax[i].set_title(name_list[i])
        ax[i].set_xlabel(r"$x_0$")

    plt.tight_layout()

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="seismic"),
        aspect=20,
        pad=0.02,
        anchor=(0, 1.0),
        ax=ax,
    )

    label = []
    for label_i in cbar.ax.get_yticklabels():
        text_i = label_i.get_text()
        leadingText = text_i.split("{")[0]
        trailingText = text_i.split("}")[1]
        text_i = text_i.replace("$\\mathdefault{", "")
        text_i = text_i.replace("}$", "")
        label += [leadingText + "{10^{" + text_i + "}}" + trailingText]

    cbar.ax.set_yticks(cbar.ax.get_yticks().tolist())
    cbar.ax.set_yticklabels(label)
    ax[0].set_ylabel(r"$x_1$")
    plt.savefig(
        os.path.join(out_dir, "odds_dist_{}.png".format(name)), bbox_inches="tight"
    )
    plt.savefig(
        os.path.join(out_dir, "odds_dist_{}.pdf".format(name)), bbox_inches="tight"
    )


def save_calibration_curves(
    out_dir, df_rct, df_test, x_name, weight_list, name_list, s_name="s", name=""
):
    import pandas as pd

    df = pd.concat([df_rct, df_test])
    s = df[s_name]
    x = df[x_name]

    odds_true_list = []
    odds_nominal_list = []
    prob_est_list = []
    for weighter, name in zip(weight_list, name_list):
        if name == "True":
            continue
        prob_est = weighter.get_p_s_x(1, x)
        odds_est = prob_est / (1 - prob_est)
        sort_id = np.argsort(odds_est)
        n_bins = 5

        # Calculate the mean value of each bin
        bins = len(df) // n_bins

        # Remove end points based on n_average
        odds_est = odds_est[sort_id]
        odds_est_bin = np.array(
            [np.mean(odds_est[i : i + bins]) for i in range(0, bins * n_bins, bins)]
        )

        s_np = s.to_numpy()
        s_np = s_np[sort_id]

        odds_true_bin_no_adj = np.array(
            [np.mean(s_np[i : i + bins]) for i in range(0, bins * n_bins, bins)]
        )
        odds_true_bin = odds_true_bin_no_adj / (
            (len(df_test) / len(df_rct)) * (1 - odds_true_bin_no_adj)
        )

        prob_est_list += [prob_est]
        odds_true_list += [odds_true_bin]
        odds_nominal_list += [odds_est_bin]

    np.save(os.path.join(out_dir, "prob_est"), prob_est_list)
    np.save(os.path.join(out_dir, "calibration_prob_true"), odds_true_list)
    np.save(os.path.join(out_dir, "calibration_prob_pred"), odds_nominal_list)


def plot_weight_curves(
    out_dir, prob_est_missing_list, prob_est_list, name_list, name=""
):
    plt.figure(figsize=(7, 5))
    color_synthetic = ["mediumseagreen", "mediumblue", "mediumseagreen", "mediumblue"]
    linestyle_synthetic = ["-", "-", "--", "--"]

    for n, (prob_est, prob_missing) in enumerate(
        zip(prob_est_list, prob_est_missing_list)
    ):
        odds_all = prob_est / (1 - prob_est)
        sort_id = np.argsort(odds_all)
        odds_missing = prob_missing.flatten() / (1 - prob_missing.flatten())
        div = odds_all[sort_id] / odds_missing[sort_id]

        if name == "nhanes":
            plt.ecdf(div, complementary=True)
        else:
            plt.ecdf(
                div,
                complementary=True,
                color=color_synthetic[n],
                ls=linestyle_synthetic[n],
            )

    if name == "nhanes":
        gamma_list = [1, 1.5, 2]
    else:
        gamma_list = [1, 2, 3]

    gamma_c_list = ["black", "dimgrey", "darkgrey"]
    gamma_ls = [":", ":", ":"]
    for i, gamma in enumerate(gamma_list):
        plt.plot([gamma, gamma], [0, 1], gamma_c_list[i], ls=gamma_ls[i])
    for i, gamma in enumerate(gamma_list):
        plt.plot([1 / gamma, 1 / gamma], [0, 1], gamma_c_list[i], ls=gamma_ls[i])

    # Modifiy lables
    label = name_list.copy()
    if name == "nhanes":
        plt.xlim([0, 2.5])
        label += [r"$\Gamma = 1$", r"$\Gamma = 1.5$", r"$\Gamma = 2$"]
    else:
        plt.xlim([0, 4])
        label += [r"$\Gamma = 1$", r"$\Gamma = 2$", r"$\Gamma = 3$"]

    plt.legend(label, title=r"Omitted covariate $X_k$", title_fontsize=14, fontsize=14)
    plt.xlabel(r"$\widehat{\mathrm{odds}}(X_{-k},X_k)/\widehat{\mathrm{odds}}(X_{-k})$")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.grid(False)

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "hist_weight_{}.png".format(name)), bbox_inches="tight"
    )
    plt.savefig(
        os.path.join(out_dir, "hist_weight_{}.pdf".format(name)), bbox_inches="tight"
    )


def plot_calibration_curves(
    out_dir, prob_true_list, prob_pred_list, name_list, name=""
):
    plt.figure(figsize=(7, 5))
    color = [
        "mediumseagreen",
        "mediumblue",
        "green",
        "cornflowerblue",
        "mediumseagreen",
    ]
    marker = ["o", "s", "o", "s"]
    linestyle = ["-", "-", "--", "--"]
    max_plot = 0
    for n, (odds_true, odds_pred) in enumerate(zip(prob_true_list, prob_pred_list)):
        sort_id = np.argsort(odds_pred)
        plt.plot(
            odds_pred[sort_id],
            odds_true[sort_id],
            color[n],
            marker=marker[n],
            markersize=5,
            ls=linestyle[n],
        )
        if max_plot < np.max(odds_pred):
            max_plot = np.max(odds_pred)

    max_plot = 2 * max_plot

    gamma_list = [1, 1.5, 2]
    gamma_c_list = ["black", "dimgrey", "darkgrey"]
    for i, gamma in enumerate(gamma_list):
        plt.plot([0, max_plot], [0, 1 / gamma * max_plot], gamma_c_list[i], ls="dotted")

    for i, gamma in enumerate(gamma_list):
        plt.plot([0, max_plot], [0, gamma * max_plot], gamma_c_list[i], ls="dotted")
    label = name_list.copy()
    label += [r"$\Gamma = 1$", r"$\Gamma = 1.5$", r"$\Gamma = 2$"]
    plt.ylim([0, 4.2])
    plt.xlim([0, 4])
    plt.legend(label, fontsize=14)
    plt.xlabel("Nominal odds")
    plt.ylabel("True odds")

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "calibration_{}.png".format(name)), bbox_inches="tight"
    )
    plt.savefig(
        os.path.join(out_dir, "calibration_{}.pdf".format(name)), bbox_inches="tight"
    )


def plot_weight_distribution(
    out_dir, df_rct, weight_list, name_list, x_name, inline=True, name=""
):
    from matplotlib import colors

    frac = 0.35

    size_title = 18
    size_label = 18
    size_ticks = 16
    if inline:
        size_title = 25
        size_label = 26
        size_ticks = 24

    def get_weight_contour(pos, get_weight_sampling):
        x, y, z = pos.shape
        weight_map = np.zeros((x, y))
        for x_i in range(x):
            for y_i in range(y):
                x_test = pos[x_i, y_i, :].reshape(-1, 2)
                weight_map[x_i, y_i] = np.log10(get_weight_sampling(x_test))
        return weight_map, np.min(weight_map), np.max(weight_map)

    x, y = np.mgrid[-3:3.1:0.1, -3:3.1:0.1]
    pos = np.dstack((x, y))

    n_subplots = len(weight_list)
    fig, ax = plt.subplots(1, n_subplots, figsize=(4 * n_subplots, 4), squeeze=False)
    plt.tick_params(axis="both", which="major", labelsize=size_label)
    ax = ax[0]

    weights_maps = []
    min_weight = np.inf
    max_weight = -np.inf
    for i in range(n_subplots):
        weight_map_i, min_weight_i, max_weight_i = get_weight_contour(
            pos, weight_list[i].get_weight_sampling
        )

        min_weight = min_weight_i if min_weight_i < min_weight else min_weight
        max_weight = max_weight_i if max_weight_i > max_weight else max_weight

        weights_maps += [weight_map_i]

    min_weight = -6
    max_weight = 2
    norm = colors.TwoSlopeNorm(vmin=min_weight, vcenter=0.0, vmax=max_weight)

    for i in range(n_subplots):
        cnt = ax[i].contourf(
            x, y, weights_maps[i], levels=levels, norm=norm, cmap="seismic"
        )

        for c in cnt.collections:
            c.set_edgecolor("face")

        df_plot = df_rct.sample(frac=frac, random_state=1)
        ax[i].scatter(df_plot["X_0"], df_plot["X_1"], s=1, c="k")
        ax[i].axis([-3, 3, -3, 3])
        ax[i].set_title(name_list[i], fontsize=size_title)
        ax[i].set_xlabel(r"$x_0$", fontsize=size_label)
        ax[i].tick_params(axis="both", which="major", labelsize=size_ticks)

    plt.tight_layout()
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="seismic"), ax=ax)
    label = []

    for label_i in cbar.ax.get_yticklabels():
        text_i = label_i.get_text()
        leadingText = text_i.split("{")[0]
        trailingText = text_i.split("}")[1]
        text_i = text_i.replace("$\\mathdefault{", "")
        text_i = text_i.replace("}$", "")
        label += [leadingText + "{10^{" + text_i + "}}" + trailingText]
    cbar.ax.set_yticks(cbar.ax.get_yticks().tolist(), labels=label, fontsize=size_label)
    cbar.ax.set_yticklabels(label)
    ax[0].set_ylabel(r"$x_1$", fontsize=size_label)

    plt.savefig(
        os.path.join(out_dir, "weight_dist_{}.png".format(name)), bbox_inches="tight"
    )
    plt.savefig(
        os.path.join(out_dir, "weight_dist_{}.pdf".format(name)), bbox_inches="tight"
    )


def plot_loss_policy(
    ax,
    plot_settings,
    load_dir,
    a_i,
    args,
    plot_data=False,
    plot_benchmark=False,
    fill_between=False,
):
    loss_list_rct = np.load(os.path.join(load_dir, "loss_list_rct_{}.npz".format(a_i)))
    alpha_list_rct = np.load(
        os.path.join(load_dir, "alpha_list_rct_{}.npz".format(a_i))
    )

    if plot_data:
        ax.step(
            loss_list_rct[
                "arr_{}".format(len(plot_settings["names"]) * len(args["gamma_list"]))
            ],
            1
            - alpha_list_rct[
                "arr_{}".format(len(plot_settings["names"]) * len(args["gamma_list"]))
            ],
            color="indianred",
            ls="solid",
            where="post",
        )

    if plot_benchmark:
        ax.step(
            loss_list_rct[
                "arr_{}".format(
                    len(plot_settings["names"]) * len(args["gamma_list"]) + 1
                )
            ],
            1
            - alpha_list_rct[
                "arr_{}".format(
                    len(plot_settings["names"]) * len(args["gamma_list"]) + 1
                )
            ],
            color="indianred",
            ls="solid",
            where="post",
        )

    for weight_i, name_i in enumerate(plot_settings["names"]):
        for i, gamma in enumerate(args["gamma_list"]):
            if name_i != "True":
                ax.step(
                    loss_list_rct[
                        "arr_{}".format(weight_i * len(args["gamma_list"]) + i)
                    ],
                    1
                    - alpha_list_rct[
                        "arr_{}".format(weight_i * len(args["gamma_list"]) + i)
                    ],
                    color=plot_settings["colors"][weight_i],
                    ls=plot_settings["ls"][i],
                    where="post",
                )

    if fill_between:
        for weight_i, name_i in enumerate(plot_settings["names"]):
            for i, gamma in enumerate(args["gamma_list"]):
                if name_i != "True" and i > 0:
                    ax.fill_between(
                        loss_list_rct[
                            "arr_{}".format(weight_i * len(args["gamma_list"]) + i)
                        ],
                        1
                        - alpha_list_rct[
                            "arr_{}".format(weight_i * len(args["gamma_list"]) + i - 1)
                        ],
                        1
                        - alpha_list_rct[
                            "arr_{}".format(weight_i * len(args["gamma_list"]) + i)
                        ],
                        color=plot_settings["colors"][weight_i],
                        step="post",
                        alpha=0.1,
                    )
    ax.set_ylabel(r"$1-\alpha$", fontsize=plot_settings["font_size"])
    ax.set_xlabel(r"$\ell_{\alpha}^{\Gamma}$", fontsize=plot_settings["font_size"])
    ax.set_xlim([-3, 10])
    ax.set_ylim([0, 1])
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.grid(True)


def plot_coverage(
    ax,
    plot_settings,
    load_dir,
    args,
    plot_benchmark=False,
    plot_rct=False,
    save_fig=False,
):
    def plot_one_trial_coverage(alpha, color, ls):
        ax.plot(
            quant_arr, quant_arr - alpha, color=color, ls=ls, marker="o", markersize=4
        )

    # Read parameters
    alpha_est_tot = np.load(os.path.join(load_dir, "coverage_rct.npy"))
    quant_arr = np.load(os.path.join(load_dir, "quant_arr.npy"))

    if plot_benchmark:
        alpha_est = np.load(os.path.join(load_dir, "coverage_rct_no_shift.npy"))
        ax.plot(
            quant_arr,
            quant_arr - alpha_est[1],
            color="indianred",
            ls="solid",
            marker="o",
            markersize=4,
        )

    if plot_rct:
        alpha_est = np.load(os.path.join(load_dir, "coverage_rct_no_shift.npy"))
        plot_one_trial_coverage(alpha_est[0], "indianred", "solid")

    for weight_i, name_i in enumerate(plot_settings["names"]):
        for i, gamma in enumerate(args["gamma_list"]):
            if name_i != "True":
                plot_one_trial_coverage(
                    alpha_est_tot[weight_i][i],
                    plot_settings["colors"][weight_i],
                    plot_settings["ls"][i],
                )

    sns.lineplot(ax=ax, x=[0, 1], y=[0, 0], color="k", ls=plot_settings["ls"][2])

    ax.set_xlabel(r"Target $\alpha$", fontsize=plot_settings["font_size"])
    ax.set_ylabel(r"Miscoverage gap", fontsize=plot_settings["font_size"])
    ax.set_xlim([0, 1])
    ax.grid(True)

    if save_fig:
        plt.savefig(os.path.join(load_dir, "alpha.pdf"), bbox_inches="tight")
        plt.savefig(os.path.join(load_dir, "alpha.png"), bbox_inches="tight")


def plot_loss_and_coverage_paper(load_dir, out_dir, save_fig=True):
    with open(os.path.join(load_dir, "config.json")) as f:
        args = json.load(f)

    plot_settings = get_plot_settings()
    plot_settings["names"] = np.load(os.path.join(load_dir, "name_list.npy"))

    a_i = args["decision"][-1]

    fig, axes = plt.subplots(1, 1, figsize=(7, 5), squeeze=False)
    axes = axes[0]

    plot_loss_policy(
        axes[0],
        plot_settings,
        load_dir,
        a_i,
        args,
        plot_benchmark=True,
        fill_between=True,
    )
    legend = ["Benchmark"]
    for name in plot_settings["names"]:
        if name != "True":
            for gamma in args["gamma_list"]:
                legend += [r"{}, $\Gamma={}$".format(name, gamma)]
    axes[0].legend(legend, fontsize=plot_settings["font_size_legend"])

    plt.tight_layout()
    if save_fig:
        plt.savefig(
            os.path.join(out_dir, "loss_{}.pdf".format(args["name"])),
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(out_dir, "loss_{}.png".format(args["name"])),
            bbox_inches="tight",
        )

    # Plot coverage curves
    fig, axes = plt.subplots(1, 1, figsize=(7, 5), squeeze=False)
    axes = axes[0]
    plot_coverage(axes[0], plot_settings, load_dir, args, plot_benchmark=True)
    axes[0].set_ylim([-0.25, 0.3])
    plt.tight_layout()

    if save_fig:
        plt.savefig(
            os.path.join(out_dir, "coverage_{}.pdf".format(args["name"])),
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(out_dir, "coverage_{}.png".format(args["name"])),
            bbox_inches="tight",
        )


def plot_loss_and_coverage_first(load_dir, out_dir, save_fig=True):
    with open(os.path.join(load_dir, "config.json")) as f:
        args = json.load(f)

    plot_settings = get_plot_settings()
    plot_settings["colors"][1] = "mediumblue"
    plot_settings["names"] = np.load(os.path.join(load_dir, "name_list.npy"))
    a_i = 1

    loss_list_rct = np.load(os.path.join(load_dir, "loss_list_rct_{}.npz".format(a_i)))
    alpha_list_rct = np.load(
        os.path.join(load_dir, "alpha_list_rct_{}.npz".format(a_i))
    )

    fig, axes = plt.subplots(1, 1, figsize=(7, 5), squeeze=False)
    axes = axes[0]

    axes[0].step(
        loss_list_rct[
            "arr_{}".format(len(plot_settings["names"]) * len(args["gamma_list"]))
        ],
        1
        - alpha_list_rct[
            "arr_{}".format(len(plot_settings["names"]) * len(args["gamma_list"]))
        ],
        color="indianred",
        ls="solid",
        where="post",
    )
    plot_loss_policy(
        axes[0], plot_settings, load_dir, a_i, args, plot_data=False, fill_between=True
    )

    axes[0].legend(
        ["RCT", r"${\Gamma = 1}$", r"${\Gamma = 1.5}$", r"${\Gamma = 2}$"],
        fontsize=plot_settings["font_size_legend"],
    )
    axes[0].set_xlabel(r"$\ell_{\alpha}$", fontsize=plot_settings["font_size"])
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.tight_layout()

    if save_fig:
        plt.savefig(os.path.join(out_dir, "loss_first.pdf"), bbox_inches="tight")
        plt.savefig(os.path.join(out_dir, "loss_first.png"), bbox_inches="tight")

    # Plot coverage curves
    alpha_est_tot = np.load(os.path.join(load_dir, "coverage_rct_no_shift.npy"))
    quant_arr = np.load(os.path.join(load_dir, "quant_arr.npy"))

    fig, axes = plt.subplots(1, 1, figsize=(7, 5), squeeze=False)
    axes = axes[0]
    plot_coverage(axes[0], plot_settings, load_dir, args, plot_rct=True)
    plt.tight_layout()

    if save_fig:
        plt.savefig(os.path.join(out_dir, "coverage_first.pdf"), bbox_inches="tight")
        plt.savefig(os.path.join(out_dir, "coverage_first.png"), bbox_inches="tight")


def plot_loss(load_dir, out_dir, a_list, name, xlim, save_fig=True):
    with open(os.path.join(load_dir, "config.json")) as f:
        args = json.load(f)

    plot_settings = get_plot_settings()
    plot_settings["names"] = ["dummy"]

    fig, axes = plt.subplots(
        1, len(a_list), figsize=(7 * len(a_list), 5), squeeze=False
    )
    axes = axes[0]

    c = plot_settings["colors"]
    for i, a_i in enumerate(a_list):
        plot_settings["colors"] = [c[a_i]]
        plot_loss_policy(
            axes[i],
            plot_settings,
            load_dir,
            a_i,
            args,
            plot_data=True,
            fill_between=True,
        )
        legend = ["RCT"]
        for gamma in args["gamma_list"]:
            legend += [r"$\Gamma={}$".format(gamma)]
        axes[i].legend(legend, fontsize=plot_settings["font_size_legend"])
        axes[i].set_xlabel(
            r"$\ell_{\alpha}^{\Gamma} \; $[$\mu$g/L]",
            fontsize=plot_settings["font_size"],
        )
        axes[i].set_xlim(xlim[a_i])
    plt.tight_layout()

    if save_fig:
        plt.savefig(
            os.path.join(out_dir, "loss_{}.png".format(name)), bbox_inches="tight"
        )
        plt.savefig(
            os.path.join(out_dir, "loss_{}.pdf".format(name)), bbox_inches="tight"
        )


def plot_loss_compare(load_dir, out_dir, xlim, xgboost=False, save_fig=True):
    with open(os.path.join(load_dir, "config.json")) as f:
        args = json.load(f)

    plot_settings = get_plot_settings()
    plot_settings["names"] = ["dummy"]

    fig, axes = plt.subplots(1, 1, figsize=(7, 5), squeeze=False)
    axes = axes[0]
    if xgboost:
        c = ["palevioletred", plot_settings["colors"][2]]
    else:
        c = ["palevioletred", plot_settings["colors"][2]]
    for a_i in args["decision"]:
        plot_settings["colors"] = [c[a_i]]
        plot_loss_policy(
            axes[0],
            plot_settings,
            load_dir,
            a_i,
            args,
            plot_data=False,
            fill_between=True,
        )
    if xgboost:
        axes[0].set_xlabel(
            r"$\ell_{\alpha}^{\Gamma}$", fontsize=plot_settings["font_size"]
        )
    else:
        axes[0].set_xlabel(
            r"$\ell_{\alpha}^{\Gamma} \; $[$\mu$g/L]",
            fontsize=plot_settings["font_size"],
        )
    axes[0].set_xlim(xlim)

    h = []
    h += [plt.plot([], [], label=r"$\pi_0$", color=c[0])[0]]
    h += [plt.plot([], [], label=r"$\pi_1$", color=c[1])[0]]
    h += [plt.plot([], [], alpha=0, label=r"$\Gamma$")[0]]
    h += [
        plt.plot([], [], label=r"${}$".format(label_i), color="k", ls=ls)[0]
        for label_i, ls in zip(args["gamma_list"], plot_settings["ls"])
    ]

    leg = plt.legend(fontsize=plot_settings["font_size_legend"])
    center_subtitles(leg)
    plt.tight_layout()

    if save_fig:
        plt.savefig(
            os.path.join(out_dir, "loss_compare_{}.png".format(args["name"])),
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(out_dir, "loss_compare_{}.pdf".format(args["name"])),
            bbox_inches="tight",
        )


def get_plot_settings():
    plot_settings = {
        "font_size": 20,
        "font_size_legend": 15,
        "colors": [
            "mediumseagreen",
            "mediumseagreen",
            "mediumblue",
            "palevioletred",
            "royalblue",
        ],
        "ls": ["solid", "dashed", "dotted"],
    }
    return plot_settings
