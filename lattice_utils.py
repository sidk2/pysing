from typing import List

import networkx as nx
import numpy as np

import lattice


def init_nearest_neighbours(
    lattice_dimensions: List[int],
    coupling_strength: float,
    external_field: float,
    T: float = 1.0,
    starting_config: np.ndarray | None = None,
    periodic: bool = True,
) -> lattice.IsingLattice:
    """
    Initialize a lattice with nearest neighbour interactions.

    Args:
        lattice_dimensions (List[int]): The dimensions of the lattice.
        coupling_strength (float): The strength of the nearest neighbour interactions.
        external_field (float): The strength of the external field.

    Returns:
        lattice.Lattice: A lattice with nearest neighbour interactions.
    """
    num_spins: int = np.prod(lattice_dimensions)
    g: nx.Graph = nx.grid_graph(lattice_dimensions, periodic=periodic)
    coupling_matrix: np.ndarray = coupling_strength * nx.to_scipy_sparse_array(g)
    bias_vector: np.ndarray = external_field * np.ones(num_spins)

    return lattice.IsingLattice(num_spins, coupling_matrix, bias_vector, T, starting_config)