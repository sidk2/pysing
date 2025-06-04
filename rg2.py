import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import tqdm

import algorithms
import lattice_utils

# -------------------------------
# 1) Coarse‐Graining Utility
# -------------------------------
def coarse_grain_3x3(spin_config, N):
    """
    Given an N×N numpy array of spins (+1 or -1), with N divisible by 3,
    return the coarse‐grained (N/3)×(N/3) array by 3×3 majority voting.
    In case of tie (4 up, 5 down or vice versa), majority is clear. 
    If exactly 4 up, 4 down, 1 either way will break ties, but in an
    odd‐sized block 3×3, ties cannot occur because 9 is odd.
    """
    assert (N % 3) == 0, "N must be divisible by 3"
    new_N = N // 3
    cg = np.zeros((new_N, new_N), dtype=int)
    for i in range(new_N):
        for j in range(new_N):
            block = spin_config[i*3:(i+1)*3, j*3:(j+1)*3]
            # Sum of spins in 3x3 block. If positive → +1, if negative → -1.
            s = np.sum(block)
            if s > 0:
                cg[i, j] = +1
            else:
                cg[i, j] = -1
    return cg

# -------------------------------
# 2) Single Simulation Function
# -------------------------------
def run_ising_and_collect_M2(
    L,               # lattice linear size
    beta,            # β = 1/T
    n_equil_sweeps,  # number of Wolff/Swendsen‐Wang steps for equilibration
    n_meas_sweeps,   # number of measurement steps
    algorithm=algorithms.swendsen_wang
):
    """
    Run an L×L Ising simulation at inverse‐temperature beta (J=1),
    return two lists (length = n_meas_sweeps):
      – M2_native:  <M_total^2> per spin of the native L×L configuration
      – M2_cg:      <M_total^2> per spin AFTER one 3×3 coarse‐graining (if L%3==0)
    
    If beta == 0, we sample random spins (infinite T), bypassing Monte Carlo.
    """
    N_spins = L * L
    # If beta = 0, infinite T → each spin is ±1 with probability ½.
    if beta == 0.0:
        # For each measurement sweep, generate a random config, compute M2_native,
        # then coarse‐grain once to size L/3 if divisible.
        M2_native = []
        M2_cg = []
        for _ in range(n_meas_sweeps):
            flat = np.random.choice([+1, -1], size=N_spins)
            config = flat.reshape((L, L))
            M_tot = np.sum(config)
            M2_native.append((M_tot * M_tot) / (N_spins**2))
            if (L % 3) == 0:
                cg_config = coarse_grain_3x3(config, L)
                M_cg_tot = np.sum(cg_config)
                N_cg_spins = (L//3) * (L//3)
                M2_cg.append((M_cg_tot * M_cg_tot) / (N_cg_spins**2))
            else:
                M2_cg.append(None)
        return np.array(M2_native), np.array(M2_cg)
    
    # Otherwise beta > 0: do a Monte Carlo run with given algorithm
    T = 1.0 / beta
    # Start from all‐up initial condition
    init_config = np.ones(N_spins, dtype=int)
    ising = lattice_utils.init_nearest_neighbours(
        [L, L],         # 2D lattice of size L×L
        coupling_strength=1.0,          # coupling J=1
        external_field=0.0,          # no external field
        T=T,
        starting_config=init_config
    )
    # 1) Equilibration:
    for _ in range(n_equil_sweeps):
        if algorithm == algorithms.glauber_dynamics:
            for __ in range(N_spins):
                ising.step(algorithms.glauber_dynamics)
        elif algorithm == algorithms.blocked_glauber:
            for __ in range(ising.num_colors):
                ising.step(algorithms.blocked_glauber)
        elif algorithm == algorithms.swendsen_wang:
            ising.step(algorithms.swendsen_wang)
        elif algorithm == algorithms.wolff_dynamics:
            ising.step(algorithms.wolff_dynamics)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    # 2) Measurement:
    M2_native = []
    M2_cg = []
    for _ in range(n_meas_sweeps):
        if algorithm == algorithms.glauber_dynamics:
            for __ in range(N_spins):
                ising.step(algorithms.glauber_dynamics)
        elif algorithm == algorithms.blocked_glauber:
            for __ in range(ising.num_colors):
                ising.step(algorithms.blocked_glauber)
        elif algorithm == algorithms.swendsen_wang:
            ising.step(algorithms.swendsen_wang)
        elif algorithm == algorithms.wolff_dynamics:
            ising.step(algorithms.wolff_dynamics)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Get the current spin configuration as a 1D array of length L^2
        flat = ising.lattice.copy()  # or ising.state, depending on your API
        config = flat.reshape((L, L))
        M_tot = np.sum(config)
        M2_native.append((M_tot * M_tot) / (N_spins**2))

        if (L % 3) == 0:
            cg_config = coarse_grain_3x3(config, L)
            M_cg_tot = np.sum(cg_config)
            N_cg_spins = (L//3) * (L//3)
            M2_cg.append((M_cg_tot * M_cg_tot) / (N_cg_spins**2))
        else:
            M2_cg.append(None)

    return np.array(M2_native), np.array(M2_cg)

# -------------------------------
# 3) Collect ⟨M²⟩ Curves
# -------------------------------
# We will build:
#   – native_M2_27[β]   for L=27
#   – cg_M2_from_81[β]  coarse‐grain 81→27
#   – (Optional: cg2_M2_from_27[β] coarse‐grain 27→9, for snapshots)
# Choose βJ values:
betas = np.linspace(0, 1, 30)
# If J=1, these β’s correspond directly to βJ in the problem statement.

# Simulation parameters:
n_equil_sweeps = 100       # e.g. 100 cluster steps to equilibrate
n_meas_sweeps = 2000       # e.g. 2000 measurement steps
algorithm = algorithms.swendsen_wang

# Storage dictionaries:
native_M2_27 = {}
cg_M2_from_81 = {}

print("===== Collecting ⟨M²⟩ for 27×27 (native) and coarse‐grained from 81×81 =====")
for beta in betas:
    # 1) Run native 27×27
    M2_native_27, _ = run_ising_and_collect_M2(
        L=27,
        beta=beta,
        n_equil_sweeps=n_equil_sweeps,
        n_meas_sweeps=n_meas_sweeps,
        algorithm=algorithm
    )
    native_M2_27[beta] = np.mean(M2_native_27)

    # 2) Run 81×81 and collect coarse‐grained 27×27 M²
    M2_native_81, M2_cg_81_to_27 = run_ising_and_collect_M2(
        L=81,
        beta=beta,
        n_equil_sweeps=n_equil_sweeps,
        n_meas_sweeps=n_meas_sweeps,
        algorithm=algorithm
    )
    cg_M2_from_81[beta] = np.mean(M2_cg_81_to_27)

    print(f"β={beta:.2f} | ⟨M²⟩_native27 = {native_M2_27[beta]:.4e} | ⟨M²⟩_cg(81→27) = {cg_M2_from_81[beta]:.4e}")

# -------------------------------
# 4) Plot ⟨M²⟩ vs β for Native 27×27 and CG(81→27)
# -------------------------------
plt.figure(figsize=(6, 5))
beta_list = list(betas)

native_vals = [native_M2_27[b] for b in beta_list]
cg_vals     = [cg_M2_from_81[b] for b in beta_list]

plt.plot(beta_list, native_vals, 'o-', label="Native 27×27 ⟨M²⟩")
plt.plot(beta_list, cg_vals,    's--', label="Coarse‐grained 81→27 ⟨M²⟩")
# Mark β=0 analytically (random spins) if desired:
#   At β=0, for 27×27, spins iid → ⟨M²⟩ = 1/(27^2). Can compute exactly:
beta0_val = native_M2_27[0.0]
plt.scatter([0.0], [beta0_val], color='red', zorder=5)
plt.text(0.0, beta0_val * 1.05, f"{beta0_val:.2e}", color='red', ha='center')

plt.xlabel(r"$\beta J$")
plt.ylabel(r"$\langle M^2\rangle$")
plt.title(r"Native $27\times27$ vs CG$(81\to27)$ $\langle M^2\rangle$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rg_M2_curves.png")
plt.close()

# -------------------------------
# 5) Compute R(β): β′ vs β
# -------------------------------
# We invert native_M2_27 curve: given ⟨M²⟩_target, find β that yields that ⟨M²⟩ in native 27.
# Use linear interpolation. Outside of range, interp1d will NaN or error, so restrict domain.
native_interp = interp1d(
    native_vals,               # x = ⟨M²⟩
    beta_list,                 # y = β that gives that ⟨M²⟩
    kind='linear',
    bounds_error=False,
    fill_value="extrapolate"
)

R_beta = {}
for beta in beta_list:
    m2_cg = cg_M2_from_81[beta]
    beta_prime = native_interp(m2_cg)
    R_beta[beta] = float(beta_prime)
    print(f"β → β′ : {beta:.3f} → {beta_prime:.3f}")

# Plot R(β) vs β and y=x line
plt.figure(figsize=(6, 5))
beta_vals = np.array(beta_list)
beta_primes = np.array([R_beta[b] for b in beta_vals])

plt.plot(beta_vals, beta_primes, 'o-', label=r"$\beta' = R(\beta)$")
plt.plot(beta_vals, beta_vals, 'k--', label=r"$\beta' = \beta$")
plt.xlabel(r"$\beta$")
plt.ylabel(r"$R(\beta)$")
plt.title(r"Renormalization‐Group Flow: $R(\beta)$ vs. $\beta$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("R_beta_curve.png")
plt.close()

# -------------------------------
# 6) (Optional) Plot RG Flow Arrows
# -------------------------------
# Pick two starting β values, e.g. 0.35, 0.5, and draw arrows: (β, R(β)) → (R(β), R(R(β)))
def draw_rg_arrows(beta_start, n_steps=3, color='blue'):
    b = beta_start
    for _ in range(n_steps):
        b_next = np.interp(b, beta_vals, beta_primes)
        # Arrow from (b, b_next) to (b_next, R(b_next))
        R_b_next = np.interp(b_next, beta_vals, beta_primes)
        plt.arrow(
            b, b_next,
            b_next - b,  # dx
            R_b_next - b_next,  # dy
            length_includes_head=True,
            head_width=0.015, head_length=0.015,
            color=color, alpha=0.7
        )
        b = b_next

plt.figure(figsize=(6, 5))
plt.plot(beta_vals, beta_primes, 'o-', label=r"$R(\beta)$")
plt.plot(beta_vals, beta_vals, 'k--', alpha=0.6)

draw_rg_arrows(0.35, n_steps=3, color='red')
draw_rg_arrows(0.50, n_steps=3, color='green')

plt.xlabel(r"$\beta$")
plt.ylabel(r"$R(\beta)$")
plt.title("RG Flow Arrows")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("RG_flow_arrows.png")
plt.close()

print("All RG plots generated and saved:")
print("  • rg_M2_curves.png")
print("  • R_beta_curve.png")
print("  • RG_flow_arrows.png")
