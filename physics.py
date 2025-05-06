import time

import matplotlib.pyplot as plt
import numpy as np

import lattice
import algorithms
import lattice_utils

# --- Simulation Parameters ---
L = 64  # Lattice dimension (e.g., 16x16)
N_spins = L * L
J_coupling = 1.0  # Ferromagnetic coupling
h_field = 0.0  # No external field for spontaneous magnetization

algorithm = algorithms.glauber_dynamics

temperatures = np.linspace(1.5, 3.5, 40)

num_equilibration_sweeps = (
    50  # Sweeps to reach equilibrium; this may need to be tuned
)
num_measurement_sweeps = 10000  # Sweeps for collecting data

# Data storage
avg_magnitudes = []
specific_heats = []
susceptibilities = []

# --- Main Simulation Loop ---
for T_current in temperatures:
    print(f"Simulating T = {T_current:.3f}")

    # Initialize lattice for current temperature
    initial_config = np.ones(N_spins, dtype=int)
    ising_system = lattice_utils.init_nearest_neighbours(
        [L, L], J_coupling, h_field, T_current, initial_config
    )

    # 1. Equilibration
    for sweep in range(num_equilibration_sweeps):
        if algorithm == algorithms.glauber_dynamics:
            for _ in range(N_spins):  # One sweep
                ising_system.step(algorithms.blocked_glauber)
        elif algorithm == algorithms.blocked_glauber:
            ising_system.step(algorithm)   
        
        if sweep % 500 == 0:
            print(
                f"  Equilibration sweep {sweep}/{num_equilibration_sweeps} at T={T_current:.3f}"
            )

    # 2. Measurement
    magnetizations_at_T = []
    energies_at_T = []

    for sweep in range(num_measurement_sweeps):
        
        if algorithm == algorithms.glauber_dynamics:
            for _ in range(N_spins):  # One sweep
                ising_system.step(algorithms.blocked_glauber)
        elif algorithm == algorithms.blocked_glauber:
            ising_system.step(algorithm) 
              
        current_M_total = ising_system.magnetization()
        current_E_total = ising_system.energy()

        magnetizations_at_T.append(current_M_total)
        energies_at_T.append(current_E_total)
        if sweep % 1000 == 0:
            print(
                f"  Measurement sweep {sweep}/{num_measurement_sweeps} at T={T_current:.3f}"
            )

    # Calculate averages (magnetization is total, energy is total)
    # Absolute magnetization per spin <|m|>
    avg_abs_M_per_spin = np.mean(np.abs(magnetizations_at_T)) / N_spins
    avg_magnitudes.append(avg_abs_M_per_spin)

    # For heat capacity, magnetic susceptibility, we need <E>, <E^2>, <M^2>, <|M|>
    # Note: magnetizations_at_T contains M_total
    # energies_at_T contains E_total

    E_array = np.array(energies_at_T)
    M_array = np.array(magnetizations_at_T)

    mean_E = np.mean(E_array)
    mean_E_sq = np.mean(E_array**2)

    mean_M_sq = np.mean(M_array**2)
    # For susceptibility, <|M_tot|> is needed, which is np.mean(np.abs(M_array))
    mean_abs_M_tot = np.mean(np.abs(M_array))

    # Specific Heat (per spin) Cv = (<E^2> - <E>^2) / (N * T^2) (k_B=1)
    if T_current == 0:  # Avoid division by zero
        specific_heat = 0  # Or handle as appropriate, Cv -> 0 as T -> 0
    else:
        specific_heat = (mean_E_sq - mean_E**2) / (N_spins * T_current**2)
    specific_heats.append(specific_heat)

    # Magnetic Susceptibility (per spin) Chi = (<M_tot^2> - <|M_tot|>^2) / (N * T) (k_B=1)
    if T_current == 0:
        susceptibility = (
            0  # Chi might diverge or be ill-defined at T=0 depending on definition
        )
    else:
        susceptibility = (mean_M_sq - mean_abs_M_tot**2) / (N_spins * T_current)
    susceptibilities.append(susceptibility)

# --- 3. Plotting ---
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.plot(temperatures, avg_magnitudes, "o-", label=f"L={L}")
plt.xlabel("Temperature (T)")
plt.ylabel("Average Absolute Magnetization per Spin <|m|>")
plt.title("Magnetization vs. Temperature")
plt.axvline(
    2.269 * J_coupling,
    color="r",
    linestyle="--",
    label=f"$T_c$ (approx={2.269*J_coupling:.2f})",
)
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(temperatures, specific_heats, "s-", label=f"L={L}")
plt.xlabel("Temperature (T)")
plt.ylabel("Specific Heat ($C_v$) per Spin")
plt.title("Specific Heat vs. Temperature")
plt.axvline(
    2.269 * J_coupling,
    color="r",
    linestyle="--",
    label=f"$T_c$ (approx={2.269*J_coupling:.2f})",
)
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(temperatures, susceptibilities, "^-", label=f"L={L}")
plt.xlabel("Temperature (T)")
plt.ylabel("Magnetic Susceptibility ($\chi$) per Spin")
plt.title("Susceptibility vs. Temperature")
plt.axvline(
    2.269 * J_coupling,
    color="r",
    linestyle="--",
    label=f"$T_c$ (approx={2.269*J_coupling:.2f})",
)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
