import sys
from os.path import abspath, dirname

path = dirname(dirname(abspath(__file__)))
sys.path.append(path)
import numpy as np
import torch
from hydra import compose, initialize

from src.antenna_arrays import UniformLinearArray
from src.metrics import fim_c_s, fim_c_xi, fim_p
from src.ofdmchannel import MultiPathChannelConfig, OFDMBeamSpaceChannel, OFDMConfig
from src.prepare_sim import get_prior_cov, get_scale_matrix_J
from src.utils import db2lin, herm


def test_fim():
    with initialize(version_base=None, config_path="."):
        cfg = compose(config_name="test_config")

    mpc_configs = MultiPathChannelConfig.random_generate(cfg.num_paths)
    ofdm_configs = OFDMConfig(
        subcarrier_spacing=cfg.subcarrier_spacing,
        num_guard_carriers=(0, 0),
        Nfft=cfg.num_carriers,
    )
    tx_array = UniformLinearArray(num_antennas=cfg.num_ant_tx)
    rx_array = UniformLinearArray(num_antennas=cfg.num_ant_rx)

    w = torch.randn(cfg.num_ant_tx, cfg.num_streams) + 1j * torch.randn(cfg.num_ant_tx, cfg.num_streams)
    x = w[..., None, None].repeat(1, 1, cfg.num_carriers, cfg.num_symbols)
    assert torch.allclose(x[:, :, 0, 0], w)

    ofdmbschann = OFDMBeamSpaceChannel(
        mpc_configs=mpc_configs,
        ofdm_config=ofdm_configs,
        tx_array=tx_array,
        rx_array=rx_array,
    )
    Hf = ofdmbschann.get_channel(num_carriers=cfg.num_carriers, num_symbols=cfg.num_symbols)
    Hf_shift = np.fft.ifftshift(Hf, axes=-2)
    Hf_tensor = torch.from_numpy(Hf_shift).clone()

    snrlin = db2lin(cfg.snrdb)
    power_noise = cfg.power_signal / snrlin
    noise_sigma2 = power_noise

    J = get_scale_matrix_J(cfg)

    fim_w = fim_c_xi(mpc_configs, ofdm_configs, tx_array, rx_array, w, Hf_tensor, noise_sigma2)

    assert np.allclose(fim_w, herm(fim_w))
    assert herm(J) @ fim_w @ J is not None

    HH = torch.einsum("rtnk, rlnk -> tlnk", Hf_tensor.conj(), Hf_tensor)
    fim_complex, fim_real = fim_c_s(w, HH, noise_sigma2)
    assert torch.allclose(torch.slogdet(fim_complex)[1] * 2, torch.slogdet(fim_real)[1], rtol=1e-2)
    C_xi, C_s = get_prior_cov(cfg)
    logdet_s = torch.linalg.slogdet(fim_complex + torch.linalg.inv(C_s))

    assert torch.allclose(C_xi, torch.diag(C_xi.diag()))
    assert torch.allclose(logdet_s[0].imag, torch.tensor(0.0)) and (logdet_s[0].real > 0).all()

    eyem = torch.eye(cfg.num_streams)
    I1, I2 = fim_p(eyem, eyem)
    assert torch.allclose(I1, I2)
    assert torch.allclose(eyem, I1)
