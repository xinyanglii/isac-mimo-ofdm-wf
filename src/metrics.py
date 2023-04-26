import numpy as np
import opt_einsum as oe
import torch

from .antenna_arrays import UniformLinearArray
from .ofdmchannel import MultiPathChannelConfig, OFDMBeamSpaceChannel, OFDMConfig
from .prepare_sim import get_prior_cov, get_scale_matrix_J
from .utils import db2lin, herm


def fim_c_xi(
    mpc_configs: MultiPathChannelConfig,
    ofdm_configs: OFDMConfig,
    tx_array: UniformLinearArray,
    rx_array: UniformLinearArray,
    w: torch.Tensor,
    channel_mat: torch.Tensor,
    sig2_noise: float = 1,
    wavelength: float = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_ant_rx, num_ant_tx, num_carriers, num_symbols = channel_mat.shape
    assert (num_ant_rx == rx_array.num_antennas) and (num_ant_tx == tx_array.num_antennas)
    if not isinstance(sig2_noise, torch.Tensor):
        sig2_noise = torch.ones((num_carriers, num_symbols), dtype=torch.complex128) * sig2_noise
    assert sig2_noise.shape == (num_carriers, num_symbols)

    At = torch.from_numpy(tx_array.steering_matrix(grid=mpc_configs.aods, wavelength=wavelength, axis=1)).clone()
    Dt = torch.from_numpy(tx_array.steering_mat_der(grid=mpc_configs.aods, wavelength=wavelength, axis=1)).clone()

    Ar = torch.from_numpy(rx_array.steering_matrix(grid=mpc_configs.aoas, wavelength=wavelength, axis=1)).clone()
    Dr = torch.from_numpy(rx_array.steering_mat_der(grid=mpc_configs.aoas, wavelength=wavelength, axis=1)).clone()

    At = At.conj()
    Dt = Dt.conj()

    tgrid, fgrid = torch.meshgrid(
        torch.arange(num_symbols) * ofdm_configs.symbol_time,
        torch.arange(num_carriers) * ofdm_configs.subcarrier_spacing,
        indexing="xy",
    )
    tgrid_ = oe.contract("nk,l->nkl", tgrid, torch.as_tensor(mpc_configs.doppler_shifts))
    fgrid_ = oe.contract("nk,l->nkl", fgrid, torch.as_tensor(mpc_configs.path_delays))
    Omega_nk = torch.exp(1j * 2 * torch.pi * (tgrid_ - fgrid_))
    G_nk = oe.contract("...nk,...nkl->...nkl", -1j * 2 * torch.pi * fgrid, Omega_nk)
    F_nk = oe.contract("...nk,...nkl->...nkl", 1j * 2 * torch.pi * tgrid, Omega_nk)
    path_gains = torch.as_tensor(mpc_configs.path_gains)

    C_x = w @ herm(w)

    BG_nk = oe.contract("nkl, l -> nkl", G_nk, path_gains)
    BF_nk = oe.contract("nkl, l -> nkl", F_nk, path_gains)
    BO_nk = oe.contract("nkl, l -> nkl", Omega_nk, path_gains)

    Lambda_nk = torch.concat((Omega_nk, 1j * Omega_nk, BG_nk, BF_nk, BO_nk, BO_nk), dim=-1)

    T = torch.concat((At, At, At, At, At, Dt), dim=1)
    R = torch.concat((Ar, Ar, Ar, Ar, Dr, Ar), dim=1)

    term1 = oe.contract(
        "nk, nkl, lt, ti, ij, nkj -> lj",
        2 / sig2_noise,
        Lambda_nk.conj(),
        herm(T),
        C_x.conj(),
        T,
        Lambda_nk,
    )
    term2 = herm(R) @ R
    fim_mat = torch.mul(term1, term2).real

    return fim_mat


def fim_c_s(w, HH, sig2_noise):
    nt, ns = w.shape
    M = np.prod(HH.shape[2:])
    wHHw = oe.contract("rm, rtnk, ts -> msnk", w.conj(), HH, w).reshape(ns, ns, -1)
    fim_complex = wHHw.permute(2, 0, 1) / sig2_noise * 2
    fim_real = torch.zeros((M, 2 * ns, 2 * ns))
    for i in range(M):
        temp = fim_complex[i]
        fim_real[i, :ns, :ns] = temp.real
        fim_real[i, :ns, ns:] = (1j * temp).real
        fim_real[i, ns:, :ns] = (-1j * temp).real
        fim_real[i, ns:, ns:] = temp.real

    return fim_complex, fim_real


def fim_p(C_xi, C_s):
    Gamma = torch.linalg.inv(C_xi)
    I_s = torch.linalg.inv(C_s)
    return Gamma, I_s


def obj_bfim(X, params_list, mpc_configs, cfg):
    ofdm_configs = OFDMConfig(
        subcarrier_spacing=cfg.subcarrier_spacing,
        num_guard_carriers=(0, 0),
        Nfft=cfg.num_carriers,
    )

    snrlin = db2lin(cfg.snrdb)
    power_noise = cfg.power_signal / snrlin
    noise_sigma2 = power_noise

    tx_array = UniformLinearArray(num_antennas=cfg.num_ant_tx)
    rx_array = UniformLinearArray(num_antennas=cfg.num_ant_rx)

    w = X * np.sqrt(cfg.power_signal)
    expec_HH = 0
    expec_fimc_xi = 0

    new_br, new_bi, new_tau, new_fD, new_aoas, new_aods = params_list
    new_tau = abs(new_tau)  # if sampled delay is negative, make it positive
    num_samples = len(new_br)
    new_b = new_br + 1j * new_bi

    scale_matrix_J = get_scale_matrix_J(cfg)

    for i in range(num_samples):
        newmpc = MultiPathChannelConfig(
            path_gains=new_b[i].numpy(),
            path_delays=new_tau[i].numpy(),
            doppler_shifts=new_fD[i].numpy(),
            aoas=new_aoas[i].numpy(),
            aods=new_aods[i].numpy(),
        )
        ofdmbschann = OFDMBeamSpaceChannel(
            mpc_configs=newmpc,
            ofdm_config=ofdm_configs,
            tx_array=tx_array,
            rx_array=rx_array,
        )
        Hf = ofdmbschann.get_channel(num_carriers=cfg.num_carriers, num_symbols=cfg.num_symbols)
        Hf_tensor = torch.from_numpy(Hf).clone()
        Hf_tensor = torch.fft.ifftshift(Hf_tensor, dim=-2)
        expec_HH = expec_HH + torch.einsum("rt..., rm... -> tm...", Hf_tensor.conj(), Hf_tensor)

        fimc_xi = fim_c_xi(
            mpc_configs=newmpc,
            ofdm_configs=ofdm_configs,
            tx_array=tx_array,
            rx_array=rx_array,
            w=w,
            channel_mat=Hf_tensor,
            sig2_noise=noise_sigma2,
        )
        expec_fimc_xi = expec_fimc_xi + fimc_xi

    if not num_samples:
        ofdmbschann = OFDMBeamSpaceChannel(
            mpc_configs=mpc_configs,
            ofdm_config=ofdm_configs,
            tx_array=tx_array,
            rx_array=rx_array,
        )
        Hf = ofdmbschann.get_channel(num_carriers=cfg.num_carriers, num_symbols=cfg.num_symbols)
        Hf_tensor = torch.from_numpy(Hf).clone()
        Hf_tensor = torch.fft.ifftshift(Hf_tensor, dim=-2)
        expec_HH = torch.einsum("rt..., rm... -> tm...", Hf_tensor.conj(), Hf_tensor)
        fimc_xi = fim_c_xi(
            mpc_configs=mpc_configs,
            ofdm_configs=ofdm_configs,
            tx_array=tx_array,
            rx_array=rx_array,
            w=w,
            channel_mat=Hf_tensor,
            sig2_noise=noise_sigma2,
        )
        expec_fimc_xi = fimc_xi
        num_samples = 1

    C_xi, C_s = get_prior_cov(cfg)
    Gamma, I_s = fim_p(C_xi, C_s)
    expec_fim_xi = expec_fimc_xi / num_samples + Gamma
    expec_HH = expec_HH / num_samples
    fim_complex, fim_real = fim_c_s(w, expec_HH, noise_sigma2)

    num_re = cfg.num_carriers * cfg.num_symbols
    term1 = torch.logdet(herm(scale_matrix_J) @ expec_fim_xi @ scale_matrix_J)
    term2 = torch.linalg.slogdet(fim_complex + I_s)[1].real.sum() * 2
    return term1, term2, (cfg.multi_obj_factor * term1 + (1 - cfg.multi_obj_factor) / num_re * term2)
