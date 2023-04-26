from typing import List, Union

import numpy as np


class UniformLinearArray:
    def __init__(
        self,
        num_antennas: int,
        antenna_spacing: float = 0.5,  # in unit of meter!
    ) -> None:
        """uniform linear antenna array (ULA), in which antennas are placed in a line
            with the same spacing distance
        Args:
            num_antennas (int): number of antennas
            antenna_spacing (float): in unit of meter, default to 0.5
        """
        self.antenna_spacing = antenna_spacing
        self.element_locations = -self.antenna_spacing * np.arange(num_antennas)
        self.num_antennas = num_antennas

    def steering_vector(
        self,
        angle: float,
        wavelength: float = 1,
    ) -> np.ndarray:
        sin_ = np.sin(angle)
        return np.exp(1j * 2 * np.pi * sin_ * self.element_locations / wavelength)

    def steering_vec_der(
        self,
        angle: float,
        wavelength: float = 1,
    ) -> np.ndarray:
        stv = self.steering_vector(angle=angle, wavelength=wavelength)
        cos = np.cos(angle)
        stvd = 1j * 2 * np.pi * (self.element_locations / wavelength) * cos * stv
        return stvd

    def steering_matrix(
        self,
        grid: Union[List[float], np.ndarray],
        wavelength: float = 1.0,
        axis: int = 0,
    ) -> np.ndarray:
        """Generate steering matrix given angle grid and stacking axis

        Args:
            grid (Union[List[float], np.ndarray]): Grid of angles on which steering vectors are computed
            wavelength (float, optional): Signal wavelength. Defaults to 1..
            axis (int, optional): axis along which steering vectors are stacked. Defaults to 0.

        Returns:
            np.ndarray: stacked steering matrix
        """
        steering_list = [self.steering_vector(a, wavelength) for a in grid]
        steering_matrix = np.stack(steering_list, axis=axis)
        return steering_matrix

    def steering_mat_der(
        self,
        grid: Union[List[float], np.ndarray],
        wavelength: float = 1.0,
        axis: int = 0,
    ) -> np.ndarray:
        """Generate the derivative steering matrix with respect to given angles

        Args:
            grid (Union[List[float], np.ndarray]): Grid of angles, grid[i] is the i-the angle
            wavelength (float, optional): Signal wavelength. Defaults to 1..
            axis (int, optional): axis along which steering vectors are stacked. Defaults to 0.

        Returns:
            np.ndarray: stacked derivative steering matrix, of size [..., 2] where the last dimension denotes
                the derivative with respect to azimuth and elevation angle respectively
        """
        stvd_list = [self.steering_vec_der(a, wavelength) for a in grid]
        stmd = np.stack(stvd_list, axis=axis)
        return stmd
