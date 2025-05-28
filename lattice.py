"""Implements an Ising model on a lattice"""

from typing import Callable, List, Any

import numpy as np
import networkx as nx
import scipy.sparse as sp


class IsingLattice:
    def __init__(
        self,
        lattice_size: int,
        J: np.ndarray,
        h: np.ndarray,
        T: float = 1.0,
        starting_config: np.ndarray | None = None,
        periodic_bc: bool = True,
    ):
        self.num_spins: np.ndarray = lattice_size
        self.J: np.ndarray = J
        self.h: np.ndarray = h
        self.lattice: np.ndarray = (
            starting_config
            if starting_config is not None
            else np.random.choice([-1, 1], size=self.num_spins)
        )
        self.T = T
        self.num_colors : int = 0
        self.coloring : List[int] = np.array(self.compute_coloring())
        
        self.J = sp.csr_matrix(self.J)
        
    def compute_coloring(self) -> List[int]:
        '''Given the weighted adjacency matrix, J compute the coloring'''
        if isinstance(self.J, np.ndarray): self.J = sp.csr_matrix(self.J)
        G = nx.from_scipy_sparse_array(self.J)
        coloring = nx.coloring.greedy_color(G)
        
        self.num_colors = len(set(coloring.values()))
        
        return [coloring[v] for v in range(self.num_spins)]
         

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
        update_fn: Callable[[Any], np.ndarray],
    ) -> np.ndarray:
        """
        Update the lattice configuration using the provided update function.

        Provides a convenient API for implementing different update algorithms.

        Args:
            update_fn (callable): The update function to use.

        Returns:
            np.ndarray: The updated lattice configuration.
        """

        self.lattice = update_fn(self)
        
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
