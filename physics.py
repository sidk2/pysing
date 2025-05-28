import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy.stats import linregress


import algorithms
import lattice_utils

def compute_mixing_time(M, max_frac=0.5,    # up to frac of N/2
                                        eps=1e-12  , min_len = 20):
    x = M - np.mean(M)
    N = len(x)
    P = np.abs(np.fft.rfft(x))**2
    ac = np.fft.irfft(P, n=N)
    
    ac = np.fft.irfft(P, n=N)[: N//2]
    ac /= ac[0]

    # 2) clip and take up to tau_max
    tau_max = int(max_frac * len(ac))
    ac_clip = np.clip(ac[:tau_max], eps, None)
    t = np.arange(tau_max)
    logac = np.log(ac_clip)

    # 3) scan over window lengths t1 = min_len ... tau_max
    best = {'r2': -np.inf, 'slope': None, 't1': None}
    for t1 in range(min_len, tau_max + 1):
        y = logac[:t1]
        x0 = t[:t1]
        slope, _, r, _, _ = linregress(x0, y)
        if slope < 0 and r*r > best['r2']:
            best.update(r2=r*r, slope=slope, t1=t1)

    if best['slope'] is None:
        raise RuntimeError("No suitable exponential window found")

    tau_rel = -1.0 / best['slope']
    return tau_rel

# --- Simulation Parameters ---
lattice_dims = [16, 32, 64, 128]
dimension = 2
J_coupling = 1.0  # Ferromagnetic coupling
h_field = 0.0  # No external field for spontaneous magnetization

algorithm = algorithms.swendsen_wang

temperatures = np.linspace(1, 3, 30)
num_equilibration_sweeps = 0  # Sweeps to reach equilibrium; may need tuning
num_measurement_sweeps = np.floor(100000*np.exp(-temperatures)).astype(int) # Sweeps for collecting data

# Data storage dictionaries
results = {L: {"mag": [], "cv": [], "chi": [], "tau": []} for L in lattice_dims}

# Main simulation loop over lattice sizes
for L in lattice_dims:
    N_spins = L**dimension
    print(f"Running simulations for L = {L}")

    for T_current, num_sweeps in zip(temperatures, num_measurement_sweeps):
        print(f"  Simulating T = {T_current:.3f}")

        # Initialize lattice for current temperature
        initial_config = np.ones(N_spins, dtype=int)
        ising_system = lattice_utils.init_nearest_neighbours(
            [L] * dimension, J_coupling, h_field, T_current, initial_config
        )

        # 1. Equilibration
        for _ in tqdm.tqdm(
            range(num_equilibration_sweeps), desc="Equilibration", leave=False
        ):
            if algorithm == algorithms.glauber_dynamics:
                for _ in range(N_spins):
                    ising_system.step(algorithms.glauber_dynamics)
            elif algorithm == algorithms.blocked_glauber:
                for _ in range(ising_system.num_colors):
                    ising_system.step(algorithms.blocked_glauber)
            elif algorithm == algorithms.swendsen_wang:
                ising_system.step(algorithms.swendsen_wang)
            elif algorithm == algorithms.wolff_dynamics:
                ising_system.step(algorithms.wolff_dynamics)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

        # 2. Measurement
        magnetizations = []
            
        energies = []
        for sweep in tqdm.tqdm(
            range(num_sweeps), desc="Measurement", leave=False
        ):
            if algorithm == algorithms.glauber_dynamics:
                for _ in range(N_spins):
                    ising_system.step(algorithms.glauber_dynamics)
            elif algorithm == algorithms.blocked_glauber:
                for _ in range(ising_system.num_colors):
                    ising_system.step(algorithms.blocked_glauber)
            elif algorithm == algorithms.swendsen_wang:
                ising_system.step(algorithms.swendsen_wang)
            elif algorithm == algorithms.wolff_dynamics:
                ising_system.step(algorithms.wolff_dynamics)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

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

        tau_M = compute_mixing_time(M_array / N_spins)
        
        print(f"Mixing time at T = {T_current} is {tau_M:.3f}")
        results[L]["mag"].append(mag)
        results[L]["cv"].append(cv)
        results[L]["chi"].append(chi)
        results[L]["tau"].append(tau_M)

    plt.show()
# Plotting all sizes on the same figure
plt.figure(figsize=(18, 5))

# 1. Magnetization
plt.subplot(1, 3, 1)
for L in lattice_dims:
    plt.plot(temperatures, results[L]["mag"], marker="o", label=f"L={L}")
plt.xlabel("Temperature (T)")
plt.ylabel("<|m|> per spin")
plt.title("Magnetization vs. Temperature")
plt.axvline(2.269 * J_coupling, linestyle="--", color="k", label="T_c")
plt.legend()
plt.grid(True)

# 2. Specific Heat
plt.subplot(1, 3, 2)
for L in lattice_dims:
    plt.plot(temperatures, results[L]["cv"], marker="s", label=f"L={L}")
plt.xlabel("Temperature (T)")
plt.ylabel("C_v per spin")
plt.title("Specific Heat vs. Temperature")
plt.axvline(2.269 * J_coupling, linestyle="--", color="k")
plt.legend()
plt.grid(True)

# 3. Susceptibility
plt.subplot(1, 3, 3)
for L in lattice_dims:
    plt.plot(temperatures, results[L]["chi"], marker="^", label=f"L={L}")
plt.xlabel("Temperature (T)")
plt.ylabel("\u03c7 per spin")
plt.title("Susceptibility vs. Temperature")
plt.axvline(2.269 * J_coupling, linestyle="--", color="k")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("pysing/figures/physics.png")
plt.close()

# 2. Autocorrelation Time vs Temperature
plt.figure(figsize=(6, 5))
for L in lattice_dims:
    plt.plot(temperatures, results[L]["tau"], marker="o", label=f"L={L}")
plt.xlabel("Temperature (T)")
plt.ylabel("Relaxation time")
plt.title("Equilibration Time vs. Temperature")
plt.axvline(2.269 * J_coupling, linestyle="--", color="k")
plt.legend()
plt.grid(True)
plt.savefig("pysing/figures/autocorrelation.png")
plt.close()
