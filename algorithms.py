import numpy as np

import lattice

# An example of how to implement a sampling algorithm. 
# They should take four arguments: state, J, h, and T
# and return the new state of the lattice

def glauber_dynamics(state: np.ndarray, J: np.ndarray, h: np.ndarray, T: float) -> np.ndarray:
    """
    Perform the Glauber dynamics algorithm on the lattice.

    Args:
        spin_glass (lattice.IsingLattice): The lattice to perform Glauber dynamics sampling on.

    Returns:
        lattice.IsingLattice: The lattice after performing Glauber dynamics sampling.
    """
    
    # Select a random spin to trial flip
    i = np.random.randint(0, len(state))
    
    # Calculate the energy change if we flip the spin
    neighbors = np.nonzero(J[i, :])[0]
    energy_change = 2 * state[i] * (np.sum(state[neighbors]) + h[i])
    
    p_flip = 1 / (np.exp(energy_change / T) + 1)
    
    # Flip the spin with probability p_flip
    if np.random.rand() < p_flip:
        state[i] *= -1
        
    return state
    