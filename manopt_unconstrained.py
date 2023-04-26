import logging
import os

import hydra
import numpy as np
import torch
from joblib import Parallel, delayed
from omegaconf import DictConfig

from src.metrics import obj_bfim
from src.ofdmchannel import MultiPathChannelConfig
from src.prepare_sim import generate_mpc_config_set, get_sampler_list
from src.stochastic_manopt import ComplexSphere, StochasticREPMSConjugateGradient

torch.set_default_dtype(torch.float64)
log = logging.getLogger(__name__)


def simulation(cfg: DictConfig, mpc_configs: MultiPathChannelConfig, num_iter: int):
    log.info("-" * 20 + f"simulation {num_iter} / {cfg.num_monte_carlo} begins" + "-" * 20)
    manifold = ComplexSphere(cfg.num_ant_tx, cfg.num_streams)

    def cost(X, params_list):
        return -obj_bfim(X=X, params_list=params_list, mpc_configs=mpc_configs, cfg=cfg)[-1]

    params_sampler_list = get_sampler_list(cfg, mpc_configs)

    optimizer = StochasticREPMSConjugateGradient(
        num_points_per_iter=cfg.num_points_per_iter,
        max_iterations=cfg.max_iterations,
        verbosity=cfg.verbose,
    )
    result = optimizer.run(manifold=manifold, base_problem=cost, param_sampler_list=params_sampler_list)
    log.info(f"[simulation {num_iter} / {cfg.num_monte_carlo}]" + f"result: {result['manopt_result'].cost}")
    log.info("-" * 20 + f"simulation {num_iter} / {cfg.num_monte_carlo} finished" + "-" * 20)
    return result


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    file_path = os.path.dirname(__file__)
    data_path = os.path.join(
        file_path,
        "data",
        f"mpc_config_set_paths_{cfg.num_paths}_montecarlo_{cfg.num_monte_carlo}.npy",
    )
    results_path = os.path.join(cfg.output_dir, "results.npy")
    try:
        # dataset already generated in advance
        mpc_config_set = np.load(data_path, allow_pickle=True)
    except Exception:
        generate_mpc_config_set(cfg, data_path)
        mpc_config_set = np.load(data_path, allow_pickle=True)

    results = Parallel(n_jobs=cfg.num_jobs)(
        delayed(simulation)(cfg, mpc_configs, n) for n, mpc_configs in enumerate(mpc_config_set)
    )
    np.save(results_path, np.array(results, dtype=object))


if __name__ == "__main__":
    main()
