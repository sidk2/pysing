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
lattice_dims = [16, 32, 64]
dimension = 2
J_coupling = 1.0  # Ferromagnetic coupling
h_field = 0.0     # No external field for spontaneous magnetization

algorithm = algorithms.swendsen_wang

temperatures = np.linspace(1.0, 3.0, 30)
num_equilibration_sweeps = 0  # Might need to increase if you want true equilibration
num_measurement_sweeps = [5000] * len(temperatures)

num_trials = 5  # <<-- Number of independent trials per (L, T)

# Prepare data structures:
# For each L, we'll store a dict of lists. Inside each list, we'll append one entry per T,
# but each entry will itself be arrays (length = num_trials).
results = {}
for L in lattice_dims:
    results[L] = {
        "mag_mean": [],
        "mag_std": [],
        "cv_mean": [],
        "cv_std": [],
        "chi_mean": [],
        "chi_std": [],
        "tau_mean": [],
        "tau_std": []
    }

# -----------------------------------
# Main simulation loop over lattice sizes
# -----------------------------------
for L in lattice_dims:
    print(f"\n=== Running simulations for L = {L} ===")
    N_spins = L**dimension

    # For each L, preallocate arrays of shape (num_trials, num_temperatures)
    mag_trials = np.zeros((num_trials, len(temperatures)))
    cv_trials = np.zeros((num_trials, len(temperatures)))
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

            # 2) Equilibration (if any)
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

            # Compute observables for this trial
            mean_abs_M = np.mean(np.abs(M_array))
            mean_E = np.mean(E_array)
            mean_E_sq = np.mean(E_array**2)
            mean_M_sq = np.mean(M_array**2)

            # Observables per spin
            mag_per_spin = (mean_abs_M / N_spins)
            cv_per_spin = (mean_E_sq - mean_E**2) / (N_spins * T_current**2)
            chi_per_spin = (mean_M_sq - mean_abs_M**2) / (N_spins * T_current)

            # Compute relaxation time using the raw M_array / N_spins
            tau_M = compute_mixing_time(M_array / N_spins)

            # Store into the trial array
            mag_trials[trial, t_idx] = mag_per_spin
            cv_trials[trial, t_idx] = cv_per_spin
            chi_trials[trial, t_idx] = chi_per_spin
            tau_trials[trial, t_idx] = tau_M

            print(f"    Trial {trial+1}/{num_trials}:  |m| = {mag_per_spin:.4f},  C_v = {cv_per_spin:.4f},  χ = {chi_per_spin:.4f},  τ = {tau_M:.3f}")

    # After all trials for every T, compute mean and std over the “trial” axis
    mag_mean = np.mean(mag_trials, axis=0)
    mag_std  = np.std(mag_trials, axis=0)
    cv_mean  = np.mean(cv_trials, axis=0)
    cv_std   = np.std(cv_trials, axis=0)
    chi_mean = np.mean(chi_trials, axis=0)
    chi_std  = np.std(chi_trials, axis=0)
    tau_mean = np.mean(tau_trials, axis=0)
    tau_std  = np.std(tau_trials, axis=0)

    # Store into results dict
    results[L]["mag_mean"].append(mag_mean)
    results[L]["mag_std"].append(mag_std)
    results[L]["cv_mean"].append(cv_mean)
    results[L]["cv_std"].append(cv_std)
    results[L]["chi_mean"].append(chi_mean)
    results[L]["chi_std"].append(chi_std)
    results[L]["tau_mean"].append(tau_mean)
    results[L]["tau_std"].append(tau_std)

# =============================================================================
# Plotting: Mean ± 1σ bands for each observable
# =============================================================================

# 1) Magnetization vs Temperature
plt.figure(figsize=(6, 5))
for L in lattice_dims:
    mag_mean = np.array(results[L]["mag_mean"]).squeeze()  # shape: (30,)
    mag_std  = np.array(results[L]["mag_std"]).squeeze()   # shape: (30,)
    plt.plot(temperatures, mag_mean, marker="o", label=f"L={L}")
    plt.fill_between(
        temperatures,
        mag_mean - mag_std,
        mag_mean + mag_std,
        alpha=0.3
    )
plt.xlabel("Temperature (T)")
plt.ylabel("<|m|> per spin (mean ± 1σ)")
plt.title("Magnetization vs. Temperature")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/magnetization_vs_T.png")
plt.close()

# 2) Specific Heat vs Temperature
plt.figure(figsize=(6, 5))
for L in lattice_dims:
    cv_mean = np.array(results[L]["cv_mean"]).squeeze()
    cv_std  = np.array(results[L]["cv_std"]).squeeze()
    plt.plot(temperatures, cv_mean, marker="s", label=f"L={L}")
    plt.fill_between(
        temperatures,
        cv_mean - cv_std,
        cv_mean + cv_std,
        alpha=0.3
    )
plt.xlabel("Temperature (T)")
plt.ylabel("C_v per spin (mean ± 1σ)")
plt.title("Specific Heat vs. Temperature")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/specific_heat_vs_T.png")
plt.close()

# 3) Susceptibility vs Temperature
plt.figure(figsize=(6, 5))
for L in lattice_dims:
    chi_mean = np.array(results[L]["chi_mean"]).squeeze()
    chi_std  = np.array(results[L]["chi_std"]).squeeze()
    plt.plot(temperatures, chi_mean, marker="^", label=f"L={L}")
    plt.fill_between(
        temperatures,
        chi_mean - chi_std,
        chi_mean + chi_std,
        alpha=0.3
    )
plt.xlabel("Temperature (T)")
plt.ylabel("χ per spin (mean ± 1σ)")
plt.title("Susceptibility vs. Temperature")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/susceptibility_vs_T.png")
plt.close()

# 4) Relaxation Time vs Temperature
plt.figure(figsize=(6, 5))
for L in lattice_dims:
    tau_mean = np.array(results[L]["tau_mean"]).squeeze()
    tau_std  = np.array(results[L]["tau_std"]).squeeze()
    plt.plot(temperatures, tau_mean, marker="o", label=f"L={L}")
    plt.fill_between(
        temperatures,
        tau_mean - tau_std,
        tau_mean + tau_std,
        alpha=0.3
    )
plt.xlabel("Temperature (T)")
plt.ylabel("Relaxation time (mean ± 1σ)")
plt.title("Equilibration Time vs. Temperature")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/autocorrelation_vs_T.png")
plt.close()
