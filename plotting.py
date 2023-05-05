import os

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from omegaconf import OmegaConf

file_dir = os.path.dirname(__file__)
plt.style.use(["science", "ieee"])
plt.rcParams.update(
    {
        # "font.family": "serif",
        "font.serif": "Times New Roman",
        # "font.weight": "normal",
        # "figure.autolayout": False,
        # "figure.titlesize": 40,
        # "lines.linewidth": 6,
        "figure.dpi": 300,
        # "font.size": 8,
        # "figure.figsize": (4, 3),
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{bm,amsmath}",
    },
)
figsize = plt.rcParams["figure.figsize"]

dir = "./multirun/manopt_unconstrained/2023-04-24/15-47-00/"
multi_obj_factors_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7000000000000001, 0.8, 0.9, 1.0]
num_points_per_iter_list = [1, 10, 50, 100]


def plot_with_std(ax, x, arr, color, tol=0.3, label=None):
    arr_mean = arr.mean(axis=0)
    arr_std = arr.std(axis=0)
    ax.plot(x, arr_mean, color=color, label=label)
    ax.fill_between(x, arr_mean - tol * arr_std, arr_mean + tol * arr_std, color=color, edgecolor=None, alpha=0.3)


if __name__ == "__main__":
    colors = {
        1: "black",
        10: "green",
        30: "dodgerblue",
        50: "red",
        100: "violet",
    }

    result_dir = os.path.join(file_dir, "results")
    plots_dir = os.path.join(file_dir, "plots")
    result_to_plot = np.load(f"{result_dir}/result_to_plot.npz")
    cfg = OmegaConf.load(f"{result_dir}/result_to_plot_cfg.yaml")

    objval_cvg_list = result_to_plot[
        "objval_cvg_list"
    ]  # num_monte_carlo x len_multifactors x len_stochpoints x max_iterations
    gradnorm_cvg_list = result_to_plot["gradnorm_cvg_list"]
    term1_list = result_to_plot["term1_list"]  # num_monte_carlo x len_multifactors x len_stochpoints
    term2_list = result_to_plot["term2_list"]

    factor_idx = 5
    trade_off_factor = multi_obj_factors_list[factor_idx]
    fig, ax = plt.subplots()
    for i, num_points in enumerate(num_points_per_iter_list):
        ax.plot(
            np.arange(cfg.max_iterations),
            objval_cvg_list[:, factor_idx, i].mean(0),
            color=colors[num_points],
            label=f"$N={num_points}$",
        )
    ax.set_xlabel("number of iterations")
    ax.set_ylabel("objective function value")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{plots_dir}/obj_convergence_factor{trade_off_factor}.pdf", bbox_inches="tight")
    fig.show()

    fig, ax = plt.subplots()
    for i, num_points in enumerate(num_points_per_iter_list):
        plot_with_std(
            ax,
            np.arange(cfg.max_iterations),
            gradnorm_cvg_list[:, factor_idx, i],
            color=colors[num_points],
            label=f"$N={num_points}$",
        )
    ax.set_xlabel("number of iterations")
    ax.set_ylabel("Riemannian gradient norm")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{plots_dir}/gradnorm_convergence_factor{trade_off_factor}.pdf", bbox_inches="tight")
    fig.show()

    n_idx = 3
    npoints = num_points_per_iter_list[n_idx]
    fig, ax = plt.subplots(figsize=(3.5, 2))
    ax.plot(multi_obj_factors_list, term1_list[:, :, n_idx].mean(0), color="blue", label=r"sensing metric")
    # ax.set_xscale("log")
    ax.set_xlabel(r"trade-off factor $\alpha$")
    ax.set_ylabel(r"objective value of $\hat{f}_s(\bm{W})$")
    ax2 = ax.twinx()
    ax2.plot(multi_obj_factors_list, term2_list[:, :, n_idx].mean(0), color="red", label=r"communications metric")
    ax2.set_ylabel(r"objective value of $\hat{f}_c(\bm{W})$")
    # ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles.extend(handles2)
    labels.extend(labels2)
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{plots_dir}/tradeoff_npoints{npoints}.pdf", bbox_inches="tight")
    fig.show()

    fig, ax = plt.subplots()

    for i, num_points in enumerate(num_points_per_iter_list):
        if i == 0:
            continue
        ax.plot(
            term1_list[:, :, i].mean(0),
            term2_list[:, :, i].mean(0),
            color=colors[num_points],
            label=f"$N={num_points}$",
        )

    ax.set_xlabel(r"$\hat{f}_s(\bm{W})$")
    ax.set_ylabel(r"$\hat{f}_c(\bm{W})$")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{plots_dir}/rate_region.pdf", bbox_inches="tight")
    fig.show()
