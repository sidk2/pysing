import matplotlib.pyplot as plt
import numpy as np
import tqdm

import algorithms
import lattice_utils


def sample_autocovariance(x,tmax):
    '''Compute the sample autocorrelation of the time series x 
    for t = 0,1,...,tmax-1.'''
    x_shifted = x - np.mean(x)
    return np.array([np.dot(x_shifted[:len(x)-t],x_shifted[t:])/(len(x)-t) 
                     for t in range(tmax)])

def find_correlation_time(autocov):
    '''Return the index of the first entry that is smaller than autocov[0]/e or the length of autocov if none are smaller.'''
    smaller = np.where(autocov < np.exp(-1)*autocov[0])[0]
    return smaller[0] if len(smaller) > 0 else len(autocov)

# --- Simulation Parameters ---
lattice_dims = [8, 16, 32, 64, 128]
dimension = 1
J_coupling = 1.0  # Ferromagnetic coupling
h_field = 0.0     # No external field for spontaneous magnetization

algorithm = algorithms.blocked_glauber

temperatures = np.linspace(1, 3, 30)
num_equilibration_sweeps = 0  # Sweeps to reach equilibrium; may need tuning
num_measurement_sweeps = 5000    # Sweeps for collecting data

# Data storage dictionaries
results = {L: {'mag': [], 'cv': [], 'chi': [], 'tau': []} for L in lattice_dims}

# Main simulation loop over lattice sizes
for L in lattice_dims:
    N_spins = L**dimension
    print(f"Running simulations for L = {L}")

    for T_current in temperatures:
        print(f"  Simulating T = {T_current:.3f}")

        # Initialize lattice for current temperature
        initial_config = np.ones(N_spins, dtype=int)
        ising_system = lattice_utils.init_nearest_neighbours(
            [L] * dimension, J_coupling, h_field, T_current, initial_config
        )

        # 1. Equilibration
        for _ in tqdm.tqdm(range(num_equilibration_sweeps), desc="Equilibration", leave=False):
            if algorithm == algorithms.glauber_dynamics:
                for _ in range(N_spins):
                    ising_system.step(algorithms.glauber_dynamics)
            elif algorithm == algorithms.blocked_glauber:
                for _ in range(ising_system.num_colors):
                    ising_system.step(algorithms.blocked_glauber)

        # 2. Measurement
        magnetizations = []
        energies = []
        for _ in tqdm.tqdm(range(num_measurement_sweeps), desc="Measurement", leave=False):
            if algorithm == algorithms.glauber_dynamics:
                for _ in range(N_spins):
                    ising_system.step(algorithms.glauber_dynamics)
            elif algorithm == algorithms.blocked_glauber:
                for _ in range(ising_system.num_colors):
                    ising_system.step(algorithms.blocked_glauber)
            
            magnetizations.append(ising_system.magnetization())
            energies.append(ising_system.energy())

        # Compute observables
        M_array = np.array(magnetizations)
        E_array = np.array(energies)
        mean_abs_M = np.mean(np.abs(M_array))
        mean_E = np.mean(E_array)
        mean_E_sq = np.mean(E_array**2)
        mean_M_sq = np.mean(M_array**2)

        # Average absolute magnetization per spin
        mag = mean_abs_M / N_spins
        # Specific heat per spin
        cv = (mean_E_sq - mean_E**2) / (N_spins * T_current**2)
        # Susceptibility per spin
        chi = (mean_M_sq - mean_abs_M**2) / (N_spins * T_current)
        
        tau_M = find_correlation_time(sample_autocovariance(M_array, len(M_array)))
        results[L]['mag'].append(mag)
        results[L]['cv'].append(cv)
        results[L]['chi'].append(chi)
        results[L]['tau'].append(tau_M)

# Plotting all sizes on the same figure
plt.figure(figsize=(18, 5))

# 1. Magnetization
plt.subplot(1, 3, 1)
for L in lattice_dims:
    plt.plot(temperatures, results[L]['mag'], marker='o', label=f"L={L}")
plt.xlabel("Temperature (T)")
plt.ylabel("<|m|> per spin")
plt.title("Magnetization vs. Temperature")
plt.axvline(2.269 * J_coupling, linestyle='--', color='k', label="T_c")
plt.legend()
plt.grid(True)

# 2. Specific Heat
plt.subplot(1, 3, 2)
for L in lattice_dims:
    plt.plot(temperatures, results[L]['cv'], marker='s', label=f"L={L}")
plt.xlabel("Temperature (T)")
plt.ylabel("C_v per spin")
plt.title("Specific Heat vs. Temperature")
plt.axvline(2.269 * J_coupling, linestyle='--', color='k')
plt.legend()
plt.grid(True)

# 3. Susceptibility
plt.subplot(1, 3, 3)
for L in lattice_dims:
    plt.plot(temperatures, results[L]['chi'], marker='^', label=f"L={L}")
plt.xlabel("Temperature (T)")
plt.ylabel("\u03C7 per spin")
plt.title("Susceptibility vs. Temperature")
plt.axvline(2.269 * J_coupling, linestyle='--', color='k')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("pysing/figures/physics.png")
plt.close()

# 2. Autocorrelation Time vs Temperature
plt.figure(figsize=(6,5))
for L in lattice_dims:
    plt.plot(temperatures, results[L]['tau'], marker='o', label=f"L={L}")
plt.xlabel("Temperature (T)")
plt.ylabel(r"Integrated autocorrelation time $\tau_{int}$ (sweeps)")
plt.title("Equilibration Time vs. Temperature")
plt.axvline(2.269 * J_coupling, linestyle='--', color='k')
plt.legend()
plt.grid(True)
plt.savefig("pysing/figures/autocorrelation.png")
plt.close()