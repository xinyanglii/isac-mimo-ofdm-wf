from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.constants import c

from .antenna_arrays import UniformLinearArray


@dataclass
class OFDMConfig:
    subcarrier_spacing: float = 15e3
    cp_frac: float = 0.07
    num_guard_carriers: tuple[int, int] = (4, 3)
    Nfft: int = 64

    def __post_init__(self):
        assert self.subcarrier_spacing > 0
        assert 0 <= self.cp_frac <= 1
        assert len(self.num_guard_carriers) == 2 and all(isinstance(x, int) and x >= 0 for x in self.num_guard_carriers)
        assert isinstance(self.Nfft, int) and self.Nfft > sum(self.num_guard_carriers)

    @property
    def cp_len(self):
        return int(np.ceil(self.Nfft * self.cp_frac))

    @property
    def num_samples_per_sym(self):
        return self.Nfft + self.cp_len

    @property
    def num_data_carriers(self):
        return self.Nfft - sum(self.num_guard_carriers)

    @property
    def symbol_time(self):
        return 1 / self.subcarrier_spacing

    @property
    def sampling_time(self):
        return self.symbol_time / self.Nfft


class MultiPathChannelConfig:
    def __init__(
        self,
        path_gains: np.ndarray | list[complex],
        path_delays: np.ndarray | list[float],
        doppler_shifts: np.ndarray | list[float],
        aoas: np.ndarray | list,
        aods: np.ndarray | list,
    ) -> None:
        """Multi-path channel configuration, specifed by a number of multi-path parameters, i.e.,
           path gains, path delays, doppler shifts, angle of arrivals and angle of departures. All
           parameters should have the same length, which is the number of path.

        Args:
            path_gains (np.ndarray | list[complex]): array or list of path gains
            path_delays (np.ndarray | list[float]): array or list of path gains
            doppler_shifts (np.ndarray | list[float]): array or list of doppler shits
            aoas (np.ndarray | list): array or list of angle of arrivals, in radians
            aods (np.ndarray | list): array or list of angle of departures, in radians
        """
        super().__init__()
        assert len(path_gains) == len(path_delays) == len(doppler_shifts) == len(aoas) == len(aods)
        self._num_paths = len(path_gains)
        self.path_delays = path_delays  # type: ignore
        self.path_gains = path_gains  # type: ignore
        self.doppler_shifts = doppler_shifts  # type: ignore
        self.aoas = aoas  # type: ignore
        self.aods = aods  # type: ignore

    @property
    def num_paths(self) -> int:
        return len(self.path_gains)

    @property
    def path_gains(self) -> np.ndarray:
        return self._path_gains

    @path_gains.setter
    def path_gains(self, b: np.ndarray | list[complex]) -> None:
        assert self._num_paths == len(b)
        assert all([np.isscalar(x) for x in b])
        self._path_gains = np.asarray(b)

    @property
    def path_delays(self) -> np.ndarray:
        return self._path_delays

    @path_delays.setter
    def path_delays(self, tau: np.ndarray | list[float]) -> None:
        assert self._num_paths == len(tau)
        assert all([np.isscalar(x) for x in tau])
        assert all([x >= 0 for x in tau])
        self._path_delays = np.asarray(tau)

    @property
    def doppler_shifts(self) -> np.ndarray:
        return self._doppler_shifts

    @doppler_shifts.setter
    def doppler_shifts(self, ds: np.ndarray | list[float]) -> None:
        assert self._num_paths == len(ds)
        assert all([np.isscalar(x) for x in ds])
        self._doppler_shifts = np.asarray(ds)

    @property
    def aoas(self) -> np.ndarray:
        return self._aoas

    @aoas.setter
    def aoas(self, theta: np.ndarray | list) -> None:
        self._aoas = np.asarray(theta)

    @property
    def aods(self) -> np.ndarray:
        return self._aods

    @aods.setter
    def aods(self, phi: np.ndarray | list[float]) -> None:
        self._aods = np.asarray(phi)

    @staticmethod
    def random_generate(
        num_paths: int,
        sampling_time: float | None = None,
        carrier_frequency: float | None = None,
    ):
        assert isinstance(num_paths, int) and num_paths > 0
        sampling_time = sampling_time if sampling_time else 1 / (1024 * 15e3)
        assert sampling_time > 0
        fc = carrier_frequency if carrier_frequency else 3e9
        assert fc > 0
        path_gains = np.random.normal(size=num_paths) + 1j * np.random.normal(size=num_paths)

        aods = np.random.uniform(-np.pi / 2, np.pi / 2, size=num_paths)
        aoas = np.random.uniform(-np.pi / 2, np.pi / 2, size=num_paths)

        d = np.random.uniform(10, 800, size=num_paths)
        path_delays = d / c
        v = np.random.uniform(0, 80, size=num_paths)
        doppler_shifts = v * fc / c

        mpc_configs = MultiPathChannelConfig(
            path_gains=path_gains,
            path_delays=path_delays,
            doppler_shifts=doppler_shifts,
            aoas=aoas,
            aods=aods,
        )

        return mpc_configs


class OFDMBeamSpaceChannel:
    def __init__(
        self,
        mpc_configs: MultiPathChannelConfig,
        ofdm_config: OFDMConfig,
        tx_array: UniformLinearArray,
        rx_array: UniformLinearArray,
    ) -> None:
        """Beam space OFDM channel model for ULA Tx/Rx array
        Args:
            mpc_configs (MultiPathChannelConfig): multi-path configurations, should contains
                                        path gains, path delays, doppler shifts, aoas and aods
            ofdm_config (OFDMConfig): OFDM configurations
            tx_array (UniformLinearArray): transmit ULA antenna array
            rx_array (UniformLinearArray): receive ULA antenna array
        """
        super().__init__()
        self.mpc_configs = mpc_configs
        self.ofdm_config = ofdm_config
        self.tx_array = tx_array
        self.rx_array = rx_array

    @property
    def tx_array(self) -> UniformLinearArray:
        return self._tx_array

    @tx_array.setter
    def tx_array(self, txa: UniformLinearArray) -> None:
        assert isinstance(txa, UniformLinearArray)
        self._tx_array = txa

    @property
    def rx_array(self) -> UniformLinearArray:
        return self._rx_array

    @rx_array.setter
    def rx_array(self, rxa: UniformLinearArray) -> None:
        assert isinstance(rxa, UniformLinearArray)
        self._rx_array = rxa

    @property
    def mpc_configs(self) -> MultiPathChannelConfig:
        return self._mpc_configs

    @mpc_configs.setter
    def mpc_configs(self, mpcc: MultiPathChannelConfig) -> None:
        assert isinstance(mpcc, MultiPathChannelConfig)
        self._mpc_configs = mpcc

    @property
    def ofdm_config(self) -> OFDMConfig:
        return self._ofdm_config

    @ofdm_config.setter
    def ofdm_config(self, ofdmcon: OFDMConfig) -> None:
        assert isinstance(ofdmcon, OFDMConfig)
        self._ofdm_config = ofdmcon

    def get_channel(
        self,
        num_carriers: int,
        num_symbols: int,
        return_multipath: bool = False,
        wavelength: float = 1,
    ) -> np.ndarray:
        """return the channel matrix for each OFDM resource elements

        Args:
            num_carriers (int): number of carriers
            num_symbols (int): number of symbols
            return_multipath (bool, optional): whether to return channel multipath components. Defaults to False.
            wavelength (float): signal wavelength to generate antenna steering vectors. Defaults to 1.

        Returns:
            np.ndarray: of shape [num_ant_rx, num_ant_tx, (num_paths), num_carriers, num_symbols], note that
                        the DC component (n=0) is located at the middle of OFDM grid
        """
        Atx = self.tx_array.steering_matrix(
            grid=self.mpc_configs.aods,
            wavelength=wavelength,
        )  # num_paths x num_ants_tx
        Arx = self.rx_array.steering_matrix(
            grid=self.mpc_configs.aoas,
            wavelength=wavelength,
        )  # num_paths x num_ants_rx

        A = np.einsum("...l, ...li, ...lj->...lij", self.mpc_configs.path_gains, Arx, Atx.conj())  # type: ignore

        tt, ff = np.meshgrid(
            np.arange(num_symbols) * self.ofdm_config.symbol_time,
            np.arange(num_carriers) * self.ofdm_config.subcarrier_spacing,
        )
        tt_ = np.einsum("...nk,...l->...nkl", tt, self.mpc_configs.doppler_shifts)
        ff_ = np.einsum("...nk,...l->...nkl", ff, self.mpc_configs.path_delays)
        omega_nk = np.exp(1j * 2 * np.pi * (tt_ - ff_))
        Hf = np.einsum(
            "...nkl,...lij->...ijlnk",
            omega_nk,
            A,
        )  # [num_ant_rx x num_ant_tx x num_paths x num_carriers x num_symbols]
        Hf = np.fft.fftshift(Hf, axes=-2)  # make DC component in the middle
        if not return_multipath:
            Hf = np.einsum("...ijlnk -> ijnk", Hf)
        return Hf
