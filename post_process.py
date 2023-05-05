import os

import hydra
import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

from src.metrics import obj_bfim
from src.prepare_sim import get_sampler_list
from src.stochastic_manopt import ComplexSphere

dir = "./multirun/manopt_unconstrained/2023-04-27/11-36-26/"  # change it to your multirun results path

hydra.core.global_hydra.GlobalHydra.instance().clear()
initialize(version_base=None, config_path=dir)
mcfg = compose(config_name="multirun", return_hydra_config=True)

multi_obj_factors_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7000000000000001, 0.8, 0.9, 1.0]
num_points_per_iter_list = [1, 10, 50, 100]

data_path = os.path.join("./data", f"mpc_config_set_paths_{mcfg.num_paths}_montecarlo_{mcfg.num_monte_carlo}.npy")
mpc_config_set = np.load(data_path, allow_pickle=True)
len_multifactors = len(multi_obj_factors_list)
len_stochpoints = len(num_points_per_iter_list)

# the values of the first term, the second term of the objective function, and the objective function itself
# after optimization
term1_list, term2_list, objval_list = (
    np.zeros((mcfg.num_monte_carlo, len_multifactors, len_stochpoints)) for _ in range(3)
)

# list of the precoder values after optimization
w_opt_list = np.zeros(
    (mcfg.num_monte_carlo, len_multifactors, len_stochpoints, mcfg.num_ant_tx, mcfg.num_streams),
    dtype=complex,
)

# list of the the convergence of objective function and gradient norm
objval_cvg_list, gradnorm_cvg_list = (
    np.zeros((mcfg.num_monte_carlo, len_multifactors, len_stochpoints, mcfg.max_iterations)) for _ in range(2)
)

for m, lam in enumerate(multi_obj_factors_list):
    for n, sp in enumerate(num_points_per_iter_list):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        path = dir + f"multi_obj_factor={lam},num_points_per_iter={sp}/"
        initialize(version_base=None, config_path=path + ".hydra")
        cfg = compose(config_name="config")
        results = np.load(path + "results.npy", allow_pickle=True)

        S = ComplexSphere(cfg.num_ant_tx, cfg.num_streams)

        for k, mpc_config in enumerate(mpc_config_set):
            print(
                "-" * 20
                + f" {m+1}/{len_multifactors} - {n+1}/{len_stochpoints} - {k+1}/{len(mpc_config_set)} "
                + "-" * 20,
            )
            re = results[k]

            w_opt = re["manopt_result"].point
            w_opt_list[k, m, n] = w_opt

            # some simulations might stop before reaching the maximum number of iterations
            if len(re["objval_convergence"]) < cfg.max_iterations:
                pad_len = cfg.max_iterations - len(re["objval_convergence"])
                obj_cvg = np.pad(re["objval_convergence"], (0, pad_len), "edge")
                gradnorm_cvg = np.pad(re["gradient_norm_convergence"], (0, pad_len), "edge")
            else:
                obj_cvg = re["objval_convergence"][: cfg.max_iterations]
                gradnorm_cvg = re["gradient_norm_convergence"][: cfg.max_iterations]

            objval_cvg_list[k, m, n] = obj_cvg
            gradnorm_cvg_list[k, m, n] = gradnorm_cvg

            # approximate objective values, using 100 points
            cfg.num_points_per_iter = 100

            params_sampler_list = get_sampler_list(cfg, mpc_config)
            params_list = [sampler.sample((cfg.num_points_per_iter,)) for sampler in params_sampler_list]
            w_opt_r = torch.from_numpy(w_opt.real)
            w_opt_i = torch.from_numpy(w_opt.imag)
            r_opt = torch.cat((w_opt_r.flatten(), w_opt_i.flatten()), dim=0)

            w_opt_torch = torch.from_numpy(w_opt).clone().requires_grad_(True)
            term1, term2, objval = obj_bfim(
                X=w_opt_torch,
                params_list=params_list,
                mpc_configs=mpc_config,
                cfg=cfg,
            )

            term1_list[k, m, n] = term1.numpy(force=True)
            term2_list[k, m, n] = term2.numpy(force=True)
            objval_list[k, m, n] = objval.numpy(force=True)

OmegaConf.save(mcfg, "./results/result_to_plot_cfg.yaml")
np.savez(
    "./results/result_to_plot",
    term1_list=term1_list,
    term2_list=term2_list,
    objval_list=objval_list,
    w_opt_list=w_opt_list,
    objval_cvg_list=objval_cvg_list,
    gradnorm_cvg_list=gradnorm_cvg_list,
)
