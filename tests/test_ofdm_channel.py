import sys
from os.path import dirname, abspath
path = dirname(dirname(abspath(__file__)))
sys.path.append(path)

import pytest
import numpy as np
from src.ofdmchannel import OFDMConfig, MultiPathChannelConfig, OFDMBeamSpaceChannel
from src.antenna_arrays import UniformLinearArray

@pytest.mark.parametrize("Nfft", [31, 64] + [1, 2])
@pytest.mark.parametrize("num_guard_carriers", [(0,0), (1,3)])
def test_ofdm_config(Nfft, num_guard_carriers):
    scs = 15e3
    if Nfft <= sum(num_guard_carriers):
        with pytest.raises(Exception):
            ofdmconf = OFDMConfig(
                subcarrier_spacing=scs,
                num_guard_carriers=num_guard_carriers,
                Nfft=Nfft,
            )
    else:
        ofdmconf = OFDMConfig(subcarrier_spacing=scs, num_guard_carriers=num_guard_carriers, Nfft=Nfft)
        assert ofdmconf is not None

        num_data_carriers = Nfft - sum(num_guard_carriers)
        assert num_data_carriers == ofdmconf.num_data_carriers
        

@pytest.mark.parametrize("num_paths", [1, 3, 15])
def test_mpc_config(num_paths):
    rconfig = MultiPathChannelConfig.random_generate(num_paths=num_paths)
    assert (
        len(rconfig.path_gains)
        == len(rconfig.path_delays)
        == len(rconfig.doppler_shifts)
        == len(rconfig.aoas)
        == len(rconfig.aods)
    )

    b = np.random.randn(num_paths) + 1j * np.random.randn(num_paths)
    tau = np.random.uniform(0, 1e-7, num_paths)
    fD = np.random.uniform(0, 2e3, num_paths)
    aoas = np.random.uniform(-np.pi / 2, np.pi / 2, num_paths)
    aods = np.random.uniform(-np.pi / 2, np.pi / 2, num_paths)

    config = MultiPathChannelConfig(b, tau, fD, aoas, aods)
    assert (
        len(config.path_gains)
        == len(config.path_delays)
        == len(config.doppler_shifts)
        == len(config.aoas)
        == len(config.aods)
        == num_paths
    )
    
    
@pytest.mark.parametrize("num_paths", [1,3,5])
@pytest.mark.parametrize("num_ant_tx", [1,3,8])
@pytest.mark.parametrize("num_ant_rx", [1,3,8])
@pytest.mark.parametrize("num_carriers", [1,52,128])
@pytest.mark.parametrize("num_symbols", [1,3,14])
def test_beam_space_channel(num_paths, num_ant_tx, num_ant_rx, num_carriers, num_symbols):
    scs = 15e3
    sampling_time = 1 / scs / num_carriers
    symbol_duration = 1 / scs

    ofdmconf = OFDMConfig(subcarrier_spacing=scs, num_guard_carriers=(0, 0), Nfft=num_carriers)

    tx_array = UniformLinearArray(num_antennas=num_ant_tx)
    rx_array = UniformLinearArray(num_antennas=num_ant_rx)

    mpc_configs = MultiPathChannelConfig.random_generate(
        num_paths=num_paths,
        sampling_time=sampling_time,
    )

    obschannel = OFDMBeamSpaceChannel(
        mpc_configs=mpc_configs,
        ofdm_config=ofdmconf,
        tx_array=tx_array,
        rx_array=rx_array,
    )
    H = obschannel.get_channel(num_carriers=num_carriers, num_symbols=num_symbols)

    H_wo_shift = np.fft.ifftshift(H, axes=-2)
    At = [tx_array.steering_vector(aod) for aod in mpc_configs.aods]
    Ar = [rx_array.steering_vector(aoa) for aoa in mpc_configs.aoas]

    for n in range(num_carriers):
        for k in range(num_symbols):
            H_true = 0
            for path in range(num_paths):
                H_true += (
                    mpc_configs.path_gains[path]
                    * np.exp(
                        1j
                        * 2
                        * np.pi
                        * (
                            mpc_configs.doppler_shifts[path] * symbol_duration * k
                            - mpc_configs.path_delays[path] * scs * n
                        ),
                    )
                    * np.outer(Ar[path], At[path].conj())
                )

            assert np.allclose(H_wo_shift[:, :, n, k], H_true)