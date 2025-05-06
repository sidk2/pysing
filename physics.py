import time
import matplotlib.pyplot as plt
import numpy as np
import tqdm

import lattice
import algorithms
import lattice_utils

# --- Simulation Parameters ---
Lattice_sizes = [16, 32, 64, 128]
J_coupling = 1.0  # Ferromagnetic coupling
h_field = 0.0     # No external field for spontaneous magnetization

algorithm = algorithms.blocked_glauber

temperatures = np.linspace(1.5, 3.5, 30)
num_equilibration_sweeps = 1000  # Sweeps to reach equilibrium; may need tuning
num_measurement_sweeps = 100000    # Sweeps for collecting data

# Data storage dictionaries
results = {L: {'mag': [], 'cv': [], 'chi': []} for L in Lattice_sizes}

# Main simulation loop over lattice sizes
for L in Lattice_sizes:
    N_spins = L * L
    print(f"Running simulations for L = {L}")

    for T_current in temperatures:
        print(f"  Simulating T = {T_current:.3f}")

        # Initialize lattice for current temperature
        initial_config = np.ones(N_spins, dtype=int)
        ising_system = lattice_utils.init_nearest_neighbours(
            [L, L], J_coupling, h_field, T_current, initial_config
        )

        # 1. Equilibration
        for _ in tqdm.tqdm(range(num_equilibration_sweeps), desc="Equilibration", leave=False):
            if algorithm == algorithms.glauber_dynamics:
                for _ in range(N_spins):
                    ising_system.step(algorithms.glauber_dynamics)
            else:
                # blocked_glauber or other stepping function
                ising_system.step(algorithm)

        # 2. Measurement
        magnetizations = []
        energies = []
        for _ in tqdm.tqdm(range(num_measurement_sweeps), desc="Measurement", leave=False):
            if algorithm == algorithms.glauber_dynamics:
                for _ in range(N_spins):
                    ising_system.step(algorithms.glauber_dynamics)
            else:
                ising_system.step(algorithm)
            
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

        results[L]['mag'].append(mag)
        results[L]['cv'].append(cv)
        results[L]['chi'].append(chi)

# Plotting all sizes on the same figure
plt.figure(figsize=(18, 5))

# 1. Magnetization
plt.subplot(1, 3, 1)
for L in Lattice_sizes:
    plt.plot(temperatures, results[L]['mag'], marker='o', label=f"L={L}")
plt.xlabel("Temperature (T)")
plt.ylabel("<|m|> per spin")
plt.title("Magnetization vs. Temperature")
plt.axvline(2.269 * J_coupling, linestyle='--', color='k', label="T_c")
plt.legend()
plt.grid(True)

# 2. Specific Heat
plt.subplot(1, 3, 2)
for L in Lattice_sizes:
    plt.plot(temperatures, results[L]['cv'], marker='s', label=f"L={L}")
plt.xlabel("Temperature (T)")
plt.ylabel("C_v per spin")
plt.title("Specific Heat vs. Temperature")
plt.axvline(2.269 * J_coupling, linestyle='--', color='k')
plt.legend()
plt.grid(True)

# 3. Susceptibility
plt.subplot(1, 3, 3)
for L in Lattice_sizes:
    plt.plot(temperatures, results[L]['chi'], marker='^', label=f"L={L}")
plt.xlabel("Temperature (T)")
plt.ylabel("\u03C7 per spin")
plt.title("Susceptibility vs. Temperature")
plt.axvline(2.269 * J_coupling, linestyle='--', color='k')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()