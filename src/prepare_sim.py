import math

import numpy as np
import torch
from omegaconf import DictConfig
from torch.distributions import MultivariateNormal

from .ofdmchannel import MultiPathChannelConfig

CONST_INV_SQRT_2 = 1 / math.sqrt(2)


def from_numpy(x):
    return torch.from_numpy(x).to(torch.float64)


def generate_mpc_config_set(cfg: DictConfig, path):
    mpc_config_set = []
    for _ in range(cfg.num_monte_carlo):
        mpc_config_set.append(MultiPathChannelConfig.random_generate(num_paths=cfg.num_paths))
    mpc_config_set = np.array(mpc_config_set, dtype=object)  # type: ignore
    np.save(path, mpc_config_set)


def get_sampler_list(cfg: DictConfig, mpc_configs):
    dist_br = MultivariateNormal(
        loc=from_numpy(mpc_configs.path_gains.real),
        scale_tril=torch.eye(cfg.num_paths) * cfg.path_gain.sigma * CONST_INV_SQRT_2,
    )
    dist_bi = MultivariateNormal(
        loc=from_numpy(mpc_configs.path_gains.imag),
        scale_tril=torch.eye(cfg.num_paths) * cfg.path_gain.sigma * CONST_INV_SQRT_2,
    )
    dist_tau = MultivariateNormal(
        loc=from_numpy(mpc_configs.path_delays),
        scale_tril=torch.eye(cfg.num_paths) * cfg.delay.sigma,
    )
    dist_fD = MultivariateNormal(
        loc=from_numpy(mpc_configs.doppler_shifts),
        scale_tril=torch.eye(cfg.num_paths) * cfg.doppler.sigma,
    )
    dist_aoas = MultivariateNormal(
        loc=from_numpy(mpc_configs.aoas),
        scale_tril=torch.eye(cfg.num_paths) * cfg.aoa.sigma,
    )
    dist_aods = MultivariateNormal(
        loc=from_numpy(mpc_configs.aods),
        scale_tril=torch.eye(cfg.num_paths) * cfg.aod.sigma,
    )

    return [dist_br, dist_bi, dist_tau, dist_fD, dist_aoas, dist_aods]


def get_prior_cov(cfg: DictConfig):
    c_xi = torch.zeros(cfg.num_paths * 6)
    c_xi[: 2 * cfg.num_paths] = cfg.path_gain.sigma * CONST_INV_SQRT_2
    c_xi[2 * cfg.num_paths : 3 * cfg.num_paths] = cfg.delay.sigma
    c_xi[3 * cfg.num_paths : 4 * cfg.num_paths] = cfg.doppler.sigma
    c_xi[4 * cfg.num_paths : 5 * cfg.num_paths] = cfg.aoa.sigma
    c_xi[5 * cfg.num_paths : 6 * cfg.num_paths] = cfg.aod.sigma
    C_xi = torch.diag(c_xi**2)

    c_s = torch.ones(cfg.num_streams) * cfg.data_symbol.sigma * CONST_INV_SQRT_2
    C_s = torch.diag(c_s**2)

    return C_xi, C_s


def get_scale_matrix_J(cfg: DictConfig):
    num_params = 6
    param_scales = torch.zeros((num_params, cfg.num_paths))
    param_scales[:2, :] = 1 if cfg.path_gain.name in cfg.param_to_est else 0
    param_scales[2, :] = 1 / cfg.subcarrier_spacing if cfg.delay.name in cfg.param_to_est else 0
    param_scales[3, :] = cfg.subcarrier_spacing if cfg.doppler.name in cfg.param_to_est else 0
    param_scales[4, :] = 1 if cfg.aoa.name in cfg.param_to_est else 0
    param_scales[5, :] = 1 if cfg.aod.name in cfg.param_to_est else 0
    param_scales = param_scales.flatten()
    J = torch.zeros((num_params * cfg.num_paths, len(torch.nonzero(param_scales))))

    count = 0
    for i in range(J.shape[0]):
        if param_scales[i] != 0:
            J[i, count] = param_scales[i]
            count += 1
    return J
