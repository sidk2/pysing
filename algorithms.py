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

def wolff_dynamics(model: lattice.IsingLattice) -> np.ndarray:    
    N = model.num_spins
    i = np.random.randint(N)  # Seed site
    spin_value = model.lattice[i]
    
    checked_pairs = set()

    stack = [i]
    model.lattice[i] *= -1  

    beta = 1.0 / (BOLTZMANN_CONSTANT * model.T)

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

def swendsen_wang(state, J, T):
    """
    Perform the Swendsen-Wang algorithm on the lattice.

    Args:
        state: The original state of the lattice to perform the algorithm on
        J: The coupling constant
        T: The temperature

    Returns:
        np.ndarray: The lattice after performing one step of Swendsen-Wang algorithm.
    """
    
    def calculate_bond_probability(J, T):
        beta = 1 / T
        return 1 - np.exp(-2 * beta * J)
    
    L = state.shape[0]
    p = calculate_bond_probability(J, T)
    visited = np.zeros((L, L), dtype=bool)
    clusters = []

    # Helper function: Find a cluster using breadth-first search (BFS)
    def bfs(start_x, start_y):
        cluster = [(start_x, start_y)]
        queue = [(start_x, start_y)]
        visited[start_x, start_y] = True

        # Directions: up, down, left, right (with periodic boundary conditions)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            current_x, current_y = queue.pop(0)
            for dx, dy in directions:
                neighbor_x = (current_x + dx) % L
                neighbor_y = (current_y + dy) % L

                # If a neighbor has the same spin value and the bond probability criterion is met, it belongs to the same cluster.
                # Check if neighbor is not visited and has the same spin
                if not visited[neighbor_x, neighbor_y] and state[current_x, current_y] == state[neighbor_x, neighbor_y]:
                    # Add neighbor to cluster with probability p
                    if np.random.rand() < p:
                        visited[neighbor_x, neighbor_y] = True
                        cluster.append((neighbor_x, neighbor_y))
                        queue.append((neighbor_x, neighbor_y))
        return cluster

    # Find all clusters
    for x in range(L):
        for y in range(L):
            if not visited[x, y]:
                cluster = bfs(x, y)
                clusters.append(cluster)

    # Now we have all clusters, we can flip them with a certain probability (once per cluster)
    # the flipped cluster do not conbine with other clusters in this step
    for cluster in clusters:
        if np.random.rand() < 0.5:
            for (x, y) in cluster:
                state[x, y] *= -1
