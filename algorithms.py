import numpy as np
import numba

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

def wolff_dynamics(model: lattice.IsingLattice) -> np.ndarray:    
    N = model.num_spins
    i = np.random.randint(N)  # Seed site
    spin_value = model.lattice[i]
    
    checked_pairs = set()

    stack = [i]
    model.lattice[i] *= -1  

    beta = 1.0 / model.T

    while stack:
        site = stack.pop()
        
        neighbors = np.nonzero(model.J[site, :])[1]
        for neighbor in neighbors:
            if (site, neighbor) not in checked_pairs and (neighbor, site) not in checked_pairs:    
                checked_pairs.add((site, neighbor))
                checked_pairs.add((neighbor, site))
                if model.lattice[neighbor] * spin_value > 0:
                    J_ij = model.J[site, neighbor]
                    p_add = 1 - np.exp(-2 * beta * J_ij)                    # p_add = 0.5
                    if np.random.rand() < p_add:
                        model.lattice[neighbor] *= -1
                        stack.append(neighbor)

    return model.lattice    

# def swendsen_wang(ising: lattice.IsingLattice) -> np.ndarray:
#     """
#     One Swendsen–Wang update on an IsingLattice. Uses Union–Find algorithm for fast cluster finding.

#     Args:
#         ising: IsingLattice instance, with
#             - ising.lattice: 1D array of spins ±1, length N
#             - ising.J: CSR sparse adjacency matrix of couplings (shape N×N)
#             - ising.T: temperature

#     Returns:
#         Updated spin configuration (also stored back in ising.lattice).
#     """
#     spins = ising.lattice
#     J = ising.J.tocsr()       # CSR format
#     beta = 1.0 / ising.T

#     N = ising.num_spins
#     parent = np.arange(N)     # union-find parent pointers

#     def find(i):
#         # path‐compressed find
#         while parent[i] != i:
#             parent[i] = parent[parent[i]]
#             i = parent[i]
#         return i

#     def union(i, j):
#         ri, rj = find(i), find(j)
#         if ri != rj:
#             parent[rj] = ri

#     # For each edge (i,j) with coupling Jij, if spins[i]==spins[j], bond with prob p=1−exp(−2*Jij/T)
#     rowptr, cols, data = J.indptr, J.indices, J.data
#     for i in range(N):
#         start, end = rowptr[i], rowptr[i+1]
#         for idx in range(start, end):
#             j = cols[idx]
#             if j <= i:
#                 continue   # only process each pair once
#             if spins[i] != spins[j]:
#                 continue
#             Jij = data[idx]
#             p_bond = 1.0 - np.exp(-2.0 * beta * Jij)
#             if np.random.rand() < p_bond:
#                 union(i, j)

#     clusters = {}
#     for i in range(N):
#         root = find(i)
#         clusters.setdefault(root, []).append(i)

#     # Flip each cluster with probability 1/2
#     for cluster_sites in clusters.values():
#         if np.random.rand() < 0.5:
#             spins[cluster_sites] *= -1

#     # store back and return
#     ising.lattice = spins
#     return spins


@numba.njit
def find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i

@numba.njit
def union(parent, i, j):
    ri, rj = find(parent, i), find(parent, j)
    if ri != rj:
        parent[rj] = ri

def swendsen_wang(ising):
    spins = ising.lattice
    N     = ising.num_spins
    beta  = 1/ising.T

    # 1) extract edges
    coo   = ising.J.tocoo()
    maskU = coo.col > coo.row
    rows  = coo.row[maskU]; cols = coo.col[maskU]
    Js    = coo.data[maskU]

    # 2) same‐spin mask
    same  = (spins[rows] == spins[cols])
    rows, cols, Js = rows[same], cols[same], Js[same]

    # 3) bond mask
    p_bonds = 1 - np.exp(-2*beta * Js)
    r       = np.random.rand(p_bonds.size)
    active  = np.nonzero(r < p_bonds)[0]

    # 4) union‐find in Numba
    parent = np.arange(N)
    for k in active:
        union(parent, rows[k], cols[k])

    # 5) build & flip clusters via roots array
    roots = np.empty(N, dtype=np.int32)
    for i in range(N):
        roots[i] = find(parent, i)
    order     = np.argsort(roots)
    roots_s   = roots[order]

    # scan runs of equal roots and flip
    start = 0
    for end in range(1, N+1):
        if end==N or roots_s[end] != roots_s[start]:
            cluster_idx = order[start:end]
            if np.random.rand() < 0.5:
                spins[cluster_idx] *= -1
            start = end

    ising.lattice = spins
    return spins