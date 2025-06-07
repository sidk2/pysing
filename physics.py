import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy.stats import linregress

import algorithms
import lattice_utils

def compute_mixing_time(M, max_frac=0.5, eps=1e-12, min_len=20):
    """
    Given a time series M, subtract its mean, compute the autocorrelation via FFT,
    then find the best exponential decay window to extract the relaxation time tau_rel.
    """
    x = M - np.mean(M)
    N = len(x)
    P = np.abs(np.fft.rfft(x))**2
    ac = np.fft.irfft(P, n=N)[: N // 2]
    ac /= ac[0]

    tau_max = int(max_frac * len(ac))
    ac_clip = np.clip(ac[:tau_max], eps, None)
    t = np.arange(tau_max)
    logac = np.log(ac_clip)

    best = {'r2': -np.inf, 'slope': None, 't1': None}
    for t1 in range(min_len, tau_max + 1):
        y = logac[:t1]
        x0 = t[:t1]
        slope, _, r, _, _ = linregress(x0, y)
        if (slope < 0) and (r * r > best['r2']):
            best.update(r2=r * r, slope=slope, t1=t1)

    if best['slope'] is None:
        raise RuntimeError("No suitable exponential window found for autocorrelation decay.")

    tau_rel = -1.0 / best['slope']
    return tau_rel

# --- Simulation Parameters ---
lattice_dims = [64, 32, 16]
dimension = 2
J_coupling = 1.0  # Ferromagnetic coupling
h_field = 0.0     # No external field for spontaneous magnetization

algorithm = algorithms.blocked_glauber

temperatures = np.linspace(1.5, 3, 20)
num_equilibration_sweeps = 5000  # Might need to increase if you want true equilibration
num_measurement_sweeps = [50000] * len(temperatures)

num_trials = 1  # <<-- Number of independent trials per (L, T)

# Prepare data structures:
# For each L, we'll store a dict of arrays: 
#   - each observable gets Q1, median, and Q3, all as length-30 arrays.
results = {}
for L in lattice_dims:
    results[L] = {
        "mag_q1": [],   "mag_med": [],   "mag_q3": [],
        "cv_q1": [],    "cv_med": [],    "cv_q3": [],
        "chi_q1": [],   "chi_med": [],   "chi_q3": [],
        "tau_q1": [],   "tau_med": [],   "tau_q3": []
    }

# -----------------------------------
# Main simulation loop over lattice sizes
# -----------------------------------
for L in lattice_dims:
    print(f"\n=== Running simulations for L = {L} ===")
    N_spins = L**dimension

    # For each L, preallocate arrays of shape (num_trials, num_temperatures)
    mag_trials = np.zeros((num_trials, len(temperatures)))
    cv_trials  = np.zeros((num_trials, len(temperatures)))
    chi_trials = np.zeros((num_trials, len(temperatures)))
    tau_trials = np.zeros((num_trials, len(temperatures)))

    for t_idx, (T_current, num_sweeps) in enumerate(zip(temperatures, num_measurement_sweeps)):
        print(f"  Temperature T = {T_current:.3f}")

        # Run `num_trials` independent simulations at this (L, T)
        for trial in range(num_trials):
            # 1) Initialize a fresh Ising system
            ising_system = lattice_utils.init_nearest_neighbours(
                [L] * dimension, J_coupling, h_field, T_current
            )

            # 2) Equilibration sweeps
            for _ in range(num_equilibration_sweeps):
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

            # 3) Measurement sweeps
            magnetizations = []
            energies = []
            for _ in range(num_sweeps):
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

            # Convert to numpy arrays
            M_array = np.array(magnetizations)
            E_array = np.array(energies)

            # Compute trial-specific observables
            mean_abs_M = np.mean(np.abs(M_array))
            mean_E = np.mean(E_array)
            mean_E_sq = np.mean(E_array**2)
            mean_M_sq = np.mean(M_array**2)

            mag_per_spin = mean_abs_M / N_spins
            cv_per_spin = (mean_E_sq - mean_E**2) / (N_spins * T_current**2)
            chi_per_spin = (mean_M_sq - mean_abs_M**2) / (N_spins * T_current)

            # Compute relaxation time Tau from autocorrelation of M_array/N_spins
            tau_M = compute_mixing_time(M_array / N_spins)

            # Store into the trial arrays
            mag_trials[trial, t_idx] = mag_per_spin
            cv_trials[trial, t_idx]  = cv_per_spin
            chi_trials[trial, t_idx] = chi_per_spin
            tau_trials[trial, t_idx] = tau_M

            print(f"    Trial {trial+1}/{num_trials}:  |m| = {mag_per_spin:.4f},  C_v = {cv_per_spin:.4f},  χ = {chi_per_spin:.4f},  τ = {tau_M:.3f}")

    # ----------------------------------------------------------------------------
    # After all trials for every T, compute Q1, median, and Q3 across the “trial” axis
    # ----------------------------------------------------------------------------

    # 1) Magnetization percentiles
    mag_q1 = np.percentile(mag_trials, 25, axis=0)
    mag_med = np.median(mag_trials, axis=0)
    mag_q3 = np.percentile(mag_trials, 75, axis=0)

    # 2) Specific heat percentiles
    cv_q1 = np.percentile(cv_trials, 25, axis=0)
    cv_med = np.median(cv_trials, axis=0)
    cv_q3 = np.percentile(cv_trials, 75, axis=0)

    # 3) Susceptibility percentiles
    chi_q1 = np.percentile(chi_trials, 25, axis=0)
    chi_med = np.median(chi_trials, axis=0)
    chi_q3 = np.percentile(chi_trials, 75, axis=0)

    # 4) Relaxation time percentiles
    tau_q1 = np.percentile(tau_trials, 25, axis=0)
    tau_med = np.median(tau_trials, axis=0)
    tau_q3 = np.percentile(tau_trials, 75, axis=0)

    # Store them in the results dict
    results[L]["mag_q1"].append(mag_q1)
    results[L]["mag_med"].append(mag_med)
    results[L]["mag_q3"].append(mag_q3)

    results[L]["cv_q1"].append(cv_q1)
    results[L]["cv_med"].append(cv_med)
    results[L]["cv_q3"].append(cv_q3)

    results[L]["chi_q1"].append(chi_q1)
    results[L]["chi_med"].append(chi_med)
    results[L]["chi_q3"].append(chi_q3)

    results[L]["tau_q1"].append(tau_q1)
    results[L]["tau_med"].append(tau_med)
    results[L]["tau_q3"].append(tau_q3)

# =============================================================================
# Plotting: Median with IQR bands for each observable
# =============================================================================

# 1) Magnetization vs Temperature (Median ± IQR)
plt.figure(figsize=(6, 5))
for L in lattice_dims:
    mag_q1 = np.array(results[L]["mag_q1"]).squeeze()   # shape: (30,)
    mag_med = np.array(results[L]["mag_med"]).squeeze() # shape: (30,)
    mag_q3 = np.array(results[L]["mag_q3"]).squeeze()   # shape: (30,)

    plt.plot(temperatures, mag_med, marker="o", label=f"L={L}")
    plt.fill_between(
        temperatures,
        mag_q1,
        mag_q3,
        alpha=0.3
    )

plt.xlabel("Temperature (T)")
plt.ylabel("Median ⟨|m|⟩ per spin (with IQR)")
plt.title("Magnetization vs. Temperature (Median ± IQR)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/magnetization_vs_T_median_IQR.png")
plt.close()

# 2) Specific Heat vs Temperature (Median ± IQR)
plt.figure(figsize=(6, 5))
for L in lattice_dims:
    cv_q1 = np.array(results[L]["cv_q1"]).squeeze()
    cv_med = np.array(results[L]["cv_med"]).squeeze()
    cv_q3 = np.array(results[L]["cv_q3"]).squeeze()

    plt.plot(temperatures, cv_med, marker="s", label=f"L={L}")
    plt.fill_between(
        temperatures,
        cv_q1,
        cv_q3,
        alpha=0.3
    )

plt.xlabel("Temperature (T)")
plt.ylabel("Median C_v per spin (with IQR)")
plt.title("Specific Heat vs. Temperature (Median ± IQR)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/specific_heat_vs_T_median_IQR.png")
plt.close()

# 3) Susceptibility vs Temperature (Median ± IQR)
plt.figure(figsize=(6, 5))
for L in lattice_dims:
    chi_q1 = np.array(results[L]["chi_q1"]).squeeze()
    chi_med = np.array(results[L]["chi_med"]).squeeze()
    chi_q3 = np.array(results[L]["chi_q3"]).squeeze()

    plt.plot(temperatures, chi_med, marker="^", label=f"L={L}")
    plt.fill_between(
        temperatures,
        chi_q1,
        chi_q3,
        alpha=0.3
    )

plt.xlabel("Temperature (T)")
plt.ylabel("Median χ per spin (with IQR)")
plt.title("Susceptibility vs. Temperature (Median ± IQR)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/susceptibility_vs_T_median_IQR.png")
plt.close()

# 4) Relaxation Time vs Temperature (Median ± IQR)
plt.figure(figsize=(6, 5))
for L in lattice_dims:
    tau_q1 = np.array(results[L]["tau_q1"]).squeeze()
    tau_med = np.array(results[L]["tau_med"]).squeeze()
    tau_q3 = np.array(results[L]["tau_q3"]).squeeze()

    plt.plot(temperatures, tau_med, marker="o", label=f"L={L}")
    plt.fill_between(
        temperatures,
        tau_q1,
        tau_q3,
        alpha=0.3
    )

plt.xlabel("Temperature (T)")
plt.ylabel("Median Relaxation time (with IQR)")
plt.title("Equilibration Time vs. Temperature (Median ± IQR)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/autocorrelation_vs_T_median_IQR.png")
plt.close()
