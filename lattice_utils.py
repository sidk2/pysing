from typing import List, Sequence
import math
import cmath

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

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


def init_anisotropic_lattice(
    lattice_dimensions: Sequence[int],
    coupling_strengths: float | Sequence[float],
    external_field: float,
    T: float = 1.0,
    starting_config: np.ndarray | None = None,
    periodic: bool = True,
) -> lattice.IsingLattice:
    """
    Initialize a (possibly anisotropic) Ising lattice with nearest-neighbour couplings.

    Args:
        lattice_dimensions (Sequence[int]):
            The number of spins along each axis, e.g. [Lx, Ly, Lz, ...].
        coupling_strengths (float or Sequence[float]):
            If a single float is given, uses that J for all axes (isotropic).
            If a sequence is given, its length must match len(lattice_dimensions),
            and each entry Ji is the coupling along axis i.
        external_field (float):
            Uniform external field (bias) applied to every spin.
        T (float, optional):
            Temperature; passed to the Lattice constructor. Defaults to 1.0.
        starting_config (ndarray, optional):
            Initial spin configuration. If None, it will be randomized inside the IsingLattice.
        periodic (bool, optional):
            Whether to use periodic boundary conditions. Defaults to True.

    Returns:
        lattice.IsingLattice:
            An Ising lattice object whose J-matrix encodes the specified anisotropy.
    """
    # normalize coupling_strengths to a list of per-axis J's
    ndim = len(lattice_dimensions)
    if isinstance(coupling_strengths, (int, float)):
        Js = [float(coupling_strengths)] * ndim
    else:
        Js = list(coupling_strengths)
        if len(Js) != ndim:
            raise ValueError(
                f"Expected {ndim} coupling strengths, got {len(Js)}."
            )

    # total number of spins
    num_spins = int(np.prod(lattice_dimensions))

    # build the grid graph
    G = nx.grid_graph(dim=lattice_dimensions, periodic=periodic)

    # assign each edge a weight corresponding to its lattice direction
    for u, v in G.edges():
        # find which axis differs by ±1
        diff = np.array(v, dtype=int) - np.array(u, dtype=int)
        axis = int(np.nonzero(np.abs(diff))[0][0])
        G[u][v]['weight'] = Js[axis]

    # extract sparse coupling matrix J_ij
    coupling_matrix = nx.to_scipy_sparse_array(G, weight='weight')

    # uniform external field
    bias_vector = external_field * np.ones(num_spins)

    return lattice.IsingLattice(
        num_spins,
        coupling_matrix,
        bias_vector,
        T,
        starting_config
    )

def init_hexagonal_lattice(
    lattice_dimensions: Sequence[int],
    coupling_strength: float,
    external_field: float,
    T: float = 1.0,
    starting_config: np.ndarray | None = None,
    periodic: bool = True,
) -> lattice.IsingLattice:
    G = nx.hexagonal_lattice_graph(lattice_dimensions[0], lattice_dimensions[1], periodic=periodic)
    coupling_matrix = coupling_strength * nx.to_scipy_sparse_array(G)
    bias_vector = external_field * np.zeros(coupling_matrix.shape[0])
    return lattice.IsingLattice(coupling_matrix.shape[0], coupling_matrix, bias_vector, T, starting_config)