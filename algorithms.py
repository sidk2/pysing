import numpy as np

import lattice

def glauber_dynamics(model: lattice.IsingLattice) -> np.ndarray:
    """
    Perform the Glauber dynamics algorithm on the lattice.

    Args:
        spin_glass (lattice.IsingLattice): The lattice to perform Glauber dynamics sampling on.

    Returns:
        np.ndarray: The lattice after performing Glauber dynamics sampling.
    """

    # Select a random spin to trial flip
    i = np.random.randint(0, len(model.lattice))

    # Calculate the energy change if we flip the spin
    neighbors = np.nonzero(model.J[i, :])[0]
    energy_change = (
        2 * model.lattice[i] * (np.sum(model.lattice[neighbors]) + model.h[i])
    )

    p_flip = 1 / (np.exp(energy_change / model.T) + 1)

    # Flip the spin with probability p_flip
    if np.random.rand() < p_flip:
        model.lattice[i] *= -1

    return model.lattice


def blocked_glauber(ising: lattice.IsingLattice) -> np.ndarray:
    N = ising.num_spins
    s = ising.lattice.ravel()
    h_flat = ising.h.ravel()

    # 1) compute ΔE for every site:
    Js = ising.J.dot(s) # Compute Σ_j J_ij s_j
    deltaE = 2 * s * (h_flat + Js)  # ΔE_i = 2 s_i (h_i + Σ_j J_ij s_j)

    # 2) pick one color class at random:
    c = np.random.randint(ising.num_colors)
    mask = ising.coloring == c

    # 3) compute Glauber flip‐probabilities for this color:
    p_flip = 1.0 / (1.0 + np.exp(deltaE[mask] / ising.T))

    # 4) decide flips in parallel:
    rand = np.random.rand(mask.sum())
    to_flip = rand < p_flip

    # 5) apply flips:
    s_new = s.copy()
    idxs = np.nonzero(mask)[0]
    s_new[idxs[to_flip]] *= -1

    return s_new
