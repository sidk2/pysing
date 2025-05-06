"""Implements an Ising model on a lattice"""

from typing import Callable

import numpy as np


class IsingLattice:
    def __init__(
        self,
        lattice_size: int,
        J: np.ndarray,
        h: np.ndarray,
        T: float = 1.0,
        starting_config: np.ndarray | None = None,
    ):
        self.lattice_size: np.ndarray = lattice_size
        self.J: np.ndarray = J
        self.h: np.ndarray = h
        self.lattice: np.ndarray = (
            starting_config
            if starting_config is not None
            else np.random.choice([-1, 1], size=(lattice_size, lattice_size))
        )
        self.T = T

    def energy(self) -> float:
        """
        Calculate the energy of the lattice configuration.

        Returns:
            float: The energy of the current lattice configuration, computed as
            the negative sum of interactions between lattice spins, with interaction
            matrix J, and the external magnetic field h.
        """

        return -self.lattice @ self.J @ self.lattice - self.lattice @ self.h

    def magnetization(self) -> float:
        """
        Calculate the magnetization of the lattice configuration.

        Returns:
            float: The magnetization of the current lattice configuration, computed as
            the sum of all spins in the lattice.
        """
        return np.sum(self.lattice)

    def step(
        self,
        update_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray],
    ) -> np.ndarray:
        """
        Update the lattice configuration using the provided update function.

        Provides a convenient API for implementing different update algorithms.

        Args:
            update_fn (callable): The update function to use.

        Returns:
            np.ndarray: The updated lattice configuration.
        """

        self.lattice = update_fn(self.lattice, self.J, self.h, self.T)
        
        return self.lattice

    def update_temperature(self, T: float) -> None:
        """
        Update the temperature of the lattice.

        Args:
            T (float): The new temperature.

        Returns:
            None
        """
        self.T = T
