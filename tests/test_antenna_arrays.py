import sys
from os.path import dirname, abspath
path = dirname(dirname(abspath(__file__)))
sys.path.append(path)

import pytest
import numpy as np
from src.utils import freq2wavelen
from src.antenna_arrays import UniformLinearArray


@pytest.mark.parametrize("num_ant_tx", [1, 3, 8])
@pytest.mark.parametrize("d", [0.1, 0.3, 0.5])
@pytest.mark.parametrize("axis", [0, 1])
def test_uniform_linear_array(num_ant_tx, d, axis):
    fc = 6e9
    wavelength = freq2wavelen(fc)
    num_ants = num_ant_tx
    ant_spacing = d * wavelength

    arr = UniformLinearArray(
        num_antennas=num_ants,
        antenna_spacing=ant_spacing,
    )

    vec = np.arange(num_ants)

    def ulasv(x):
        return np.exp(-1j * 2 * np.pi * d * vec * np.sin(x))

    def ulasvder(x):
        return -1j * 2 * np.pi * d * vec * np.cos(x) * ulasv(x)

    angle = np.random.uniform(-np.pi / 2, np.pi / 2)

    stv = arr.steering_vector(angle=angle, wavelength=wavelength)
    stv_der = arr.steering_vec_der(angle=angle, wavelength=wavelength)

    stv_true = ulasv(angle)
    stv_der_true = ulasvder(angle)
    assert stv.shape == stv_true.shape
    assert np.allclose(stv_true, stv, atol=1e-6)
    assert np.allclose(stv_der_true, stv_der, atol=1e-6)

    grid = np.random.uniform(-np.pi / 2, np.pi / 2, 10)

    stm = arr.steering_matrix(grid, wavelength=wavelength, axis=axis)
    sv = [ulasv(angle) for angle in grid]
    stm_true = np.stack(sv, axis=axis)
    assert np.allclose(stm, stm_true)

    stm_der = arr.steering_mat_der(grid=grid, wavelength=wavelength, axis=axis)
    stm_der_true = [ulasvder(angle) for angle in grid]
    stm_der_true = np.stack(stm_der_true, axis=axis)
    assert np.allclose(stm_der, stm_der_true, atol=1e-5)