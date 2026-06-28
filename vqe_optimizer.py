"""
vqe_optimizer.py
================
VQE-Cryo-Optimizer: Variational Quantum Eigensolver for Stratospheric
Cryogenic Power Distribution in High-Altitude Platform Quantum Data Centers.

Authors : Research Intern, Green Quantum Computing in the Sky Project
License : MIT
Python  : 3.10+

Physical Context
----------------
Superconducting qubits operate near 15 mK, requiring active cryogenic cooling.
When a Quantum Data Center (QDC) payload is deployed on a High-Altitude Platform
(HAP) in the stratosphere (~20 km), the ambient temperature T_o ≈ 215 K, vs. the
terrestrial 300 K. This significantly reduces the cooling overhead. However, under
strict Size, Weight, and Power (SWaP) constraints, we must optimally partition a
fixed total power budget P_T between:
  - P_comp : power allocated to the quantum processor (computation)
  - P_cool : power allocated to the active cryogenic cooler

The Variational Quantum Eigensolver (VQE) is used here as a hybrid
quantum-classical optimizer: the parameterized ansatz circuit encodes a
candidate power-split (via expectation values), and the classical optimizer
(L-BFGS-B via SciPy) minimises a thermodynamically-grounded cost function
that captures the Effective Error Rate (epsilon_eff) under the SWaP budget.

Thermodynamic Model
-------------------
Cooling Power Overhead (stratospheric):

    P_cool_overhead = (1 / eta_c) * ((T_o - T_c) / T_c) * P_deposit

where:
  eta_c     : Carnot efficiency of the cryocooler (dimensionless, 0 < eta_c <= 1)
  T_o       : Ambient (stratospheric) temperature [K]  → 215 K (HAP) vs 300 K (ground)
  T_c       : Cryogenic target temperature [K]         → ~0.015 K (15 mK)
  P_deposit : Thermal heat deposited into the cold stage [W]

The cost function minimised by VQE is the Effective Error Rate:

    epsilon_eff(P_comp, P_cool) = epsilon_base * exp(alpha * P_comp_deficit)
                                  + beta * P_cool_deficit^2

subject to:
    P_comp + P_cool_overhead(P_cool) <= P_T   [SWaP power budget constraint]
    P_comp >= 0, P_cool >= 0

where:
  epsilon_base   : baseline qubit error rate (hardware-determined)
  alpha          : error amplification coefficient (fault-tolerance metric)
  beta           : cooling-failure penalty coefficient
  P_comp_deficit : max(0, P_comp_min - P_comp)   — processor underpowered penalty
  P_cool_deficit : max(0, P_cool_min - P_cool)   — cooler underpowered penalty
"""

# ─────────────────────────────────────────────────────────────────────────────
# Standard library
# ─────────────────────────────────────────────────────────────────────────────
import warnings
from dataclasses import dataclass, field
from typing import Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Third-party
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pennylane as qml
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore", category=UserWarning)


# ═════════════════════════════════════════════════════════════════════════════
# 1.  PHYSICAL & MISSION PARAMETERS  (all SI units unless noted)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class StratosphericMissionParams:
    """
    Encapsulates all physical constants and mission-specific parameters
    for a HAP-deployed Quantum Data Center.

    Attributes
    ----------
    T_o_hap   : Ambient stratospheric temperature [K]  (~215 K at ~20 km alt.)
    T_o_gnd   : Terrestrial baseline temperature  [K]  (~300 K sea level)
    T_c       : Cryogenic cold-stage temperature  [K]  (15 mK superconducting)
    eta_c     : Real Carnot efficiency fraction         (0 < eta_c <= 1)
    P_T       : Total SWaP power budget           [W]
    P_deposit : Baseline thermal load on cold stage[W] (from qubit dissipation)
    P_comp_min: Minimum processor power for operation [W]
    P_cool_min: Minimum cooler drive power        [W]
    epsilon_base : Baseline qubit error rate (hardware floor)
    alpha     : Error amplification coefficient   [1/W]
    beta      : Cooling-failure penalty coefficient [1/W^2]
    """
    # ── Thermal environment ──────────────────────────────────────────────────
    T_o_hap    : float = 215.0      # K – stratospheric HAP ambient
    T_o_gnd    : float = 300.0      # K – terrestrial ground baseline
    T_c        : float = 0.015      # K – target superconducting cold stage (15 mK)

    # ── Cryocooler efficiency ────────────────────────────────────────────────
    eta_c      : float = 0.10       # 10 % of ideal Carnot (realistic pulse-tube)

    # ── SWaP budget (HAP payload-class) ─────────────────────────────────────
    P_T        : float = 2000.0     # W – total available power on HAP
    P_deposit  : float = 5.0        # W – heat leak into cold stage (cabling + radiation)

    # ── Operational floor constraints ────────────────────────────────────────
    P_comp_min : float = 50.0       # W – minimum for stable qubit operation
    P_cool_min : float = 500.0      # W – minimum cooler drive (compressor threshold)

    # ── Error model ─────────────────────────────────────────────────────────
    epsilon_base : float = 1e-3     # baseline gate error rate (~10^-3 NISQ era)
    alpha        : float = 0.05     # error amplification per W of computation deficit
    beta         : float = 2e-6     # cooling-failure quadratic penalty coefficient


# ═════════════════════════════════════════════════════════════════════════════
# 2.  THERMODYNAMIC COST FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def cooling_overhead(
    P_cool_drive: float,
    T_o: float,
    T_c: float,
    eta_c: float,
    P_deposit: float
) -> float:
    """
    Compute the total cryocooler wall-plug power for a given cold-stage heat
    extraction under the stratospheric thermal environment.

    The idealised Carnot coefficient of performance (COP) for a refrigerator is:
        COP_Carnot = T_c / (T_o - T_c)

    A real cryocooler operates at fraction eta_c of this ideal, giving:
        COP_real = eta_c * COP_Carnot = eta_c * T_c / (T_o - T_c)

    The wall-plug overhead to extract P_deposit from the cold stage is:
        P_cool_overhead = P_deposit / COP_real
                        = (1 / eta_c) * ((T_o - T_c) / T_c) * P_deposit

    This is the primary thermodynamic equation driving the optimisation.

    Parameters
    ----------
    P_cool_drive : float  – cooler drive power budget allocated  [W] (unused here;
                             kept for API symmetry; the overhead is set by P_deposit)
    T_o          : float  – ambient temperature                  [K]
    T_c          : float  – cold-stage temperature               [K]
    eta_c        : float  – real Carnot efficiency fraction
    P_deposit    : float  – heat deposited into cold stage       [W]

    Returns
    -------
    float – required wall-plug power overhead                    [W]
    """
    cop_ideal  = T_c / (T_o - T_c)          # Carnot COP (dimensionless)
    cop_real   = eta_c * cop_ideal           # real COP accounting for irreversibilities
    overhead   = P_deposit / cop_real        # wall-plug power required
    return overhead


def effective_error_rate(
    P_comp: float,
    P_cool: float,
    params: StratosphericMissionParams,
    T_o: float
) -> float:
    """
    Compute the Effective Error Rate (epsilon_eff) for a given power allocation.

    The cost is composed of two physically motivated terms:

    1. Processor under-power penalty:
       If P_comp < P_comp_min, gate fidelity degrades exponentially because
       control electronics (AWGs, microwave sources) are starved of power.
           epsilon_comp = epsilon_base * exp(alpha * max(0, P_comp_min - P_comp))

    2. Cooling-failure penalty:
       If P_cool < P_cool_overhead, the cryocooler cannot maintain T_c,
       raising the qubit temperature and causing decoherence. This is modelled
       as a quadratic penalty:
           epsilon_cool = beta * max(0, P_cool_overhead - P_cool)^2

    Total:
        epsilon_eff = epsilon_comp + epsilon_cool

    Parameters
    ----------
    P_comp  : float                       – power to processor          [W]
    P_cool  : float                       – power to cryocooler drive   [W]
    params  : StratosphericMissionParams  – mission physical parameters
    T_o     : float                       – ambient temperature in use   [K]

    Returns
    -------
    float – effective error rate (dimensionless; lower is better)
    """
    # ── Required cooling overhead given the ambient environment ──────────────
    P_required_cool = cooling_overhead(
        P_cool, T_o, params.T_c, params.eta_c, params.P_deposit
    )

    # ── Processor under-power deficit ────────────────────────────────────────
    P_comp_deficit = max(0.0, params.P_comp_min - P_comp)
    epsilon_comp   = params.epsilon_base * np.exp(params.alpha * P_comp_deficit)

    # ── Cooling deficit penalty ───────────────────────────────────────────────
    P_cool_deficit = max(0.0, P_required_cool - P_cool)
    epsilon_cool   = params.beta * P_cool_deficit ** 2

    return float(epsilon_comp + epsilon_cool)


# ═════════════════════════════════════════════════════════════════════════════
# 3.  VQE ANSATZ & QUANTUM DEVICE
# ═════════════════════════════════════════════════════════════════════════════

N_QUBITS = 4   # Encoding dimensionality for the power-split parameter space.
               # With 4 qubits we obtain 4 expectation values whose weighted
               # combination spans a rich, entangled parameter landscape.

# Use PennyLane's default.qubit simulator (statevector, exact arithmetic).
dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev)
def vqe_ansatz(theta: np.ndarray) -> list:
    """
    Parameterised quantum ansatz circuit.

    Architecture: Hardware-Efficient Ansatz (HEA) with 2 layers of
    strongly-entangling Rot gates followed by CNOT entanglers.

    The expectation values <Z_0>, <Z_1>, <Z_2>, <Z_3> are used as
    normalised coordinates in [-1, +1]^4, which are then decoded
    (see decode_expectation_values) into physical power allocations.

    Parameters
    ----------
    theta : np.ndarray, shape (N_QUBITS * 3 * n_layers,)
        Variational parameters: [phi, theta, omega] Euler angles per qubit
        per layer.

    Returns
    -------
    list of float – Pauli-Z expectation values for each qubit.
    """
    n_layers = 2   # depth of the HEA ansatz

    # ── Layer 1 : Ry rotations + entangling CNOT ring ────────────────────────
    for layer in range(n_layers):
        offset = layer * N_QUBITS * 3
        # General SU(2) rotation on each qubit
        for q in range(N_QUBITS):
            idx = offset + q * 3
            qml.Rot(theta[idx], theta[idx + 1], theta[idx + 2], wires=q)
        # Entangling layer: CNOT ring topology
        for q in range(N_QUBITS):
            qml.CNOT(wires=[q, (q + 1) % N_QUBITS])

    # ── Measure Pauli-Z on each qubit ────────────────────────────────────────
    return [qml.expval(qml.PauliZ(q)) for q in range(N_QUBITS)]


def decode_expectation_values(
    exp_vals: np.ndarray,
    P_T: float,
    P_comp_min: float,
    P_cool_min: float
) -> Tuple[float, float]:
    """
    Map quantum circuit expectation values ∈ [-1, +1]^4 to physical
    power allocations (P_comp, P_cool) satisfying P_comp + P_cool ≤ P_T.

    Strategy:
    - Use the mean of the first two qubits to derive a split fraction f ∈ [0, 1].
    - f encodes what fraction of the available surplus budget goes to computation.
    - Hard lower bounds (P_comp_min, P_cool_min) are enforced via offsetting.

    Parameters
    ----------
    exp_vals  : np.ndarray – 4 Pauli-Z expectation values from the circuit
    P_T       : float      – total power budget [W]
    P_comp_min: float      – minimum processor power [W]
    P_cool_min: float      – minimum cooler power [W]

    Returns
    -------
    (P_comp, P_cool) : Tuple[float, float] – physical power allocations [W]
    """
    # Map mean of first two qubits from [-1, +1] → [0, 1]
    f = (np.mean(exp_vals[:2]) + 1.0) / 2.0   # split fraction

    # Reserved power for mandatory minimums
    P_reserved = P_comp_min + P_cool_min
    P_surplus   = max(0.0, P_T - P_reserved)

    # Allocate surplus according to quantum-derived split fraction
    P_comp = P_comp_min + f * P_surplus
    P_cool = P_cool_min + (1.0 - f) * P_surplus

    return P_comp, P_cool


# ═════════════════════════════════════════════════════════════════════════════
# 4.  HYBRID VQE COST FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def vqe_cost(
    theta: np.ndarray,
    params: StratosphericMissionParams,
    T_o: float
) -> float:
    """
    Hybrid quantum-classical cost function for the VQE optimisation loop.

    Workflow per classical optimiser call:
      1. Execute quantum ansatz circuit → expectation values
      2. Decode expectation values → (P_comp, P_cool)
      3. Enforce SWaP constraint (soft penalty if violated)
      4. Evaluate thermodynamic cost → epsilon_eff

    Parameters
    ----------
    theta  : np.ndarray                  – variational circuit parameters
    params : StratosphericMissionParams  – mission physical parameters
    T_o    : float                       – ambient temperature           [K]

    Returns
    -------
    float – scalar cost (effective error rate + SWaP violation penalty)
    """
    # ── Quantum forward pass ─────────────────────────────────────────────────
    exp_vals = np.array(vqe_ansatz(theta))

    # ── Decode to physical powers ─────────────────────────────────────────────
    P_comp, P_cool = decode_expectation_values(
        exp_vals, params.P_T, params.P_comp_min, params.P_cool_min
    )

    # ── SWaP hard-constraint enforcement (quadratic barrier) ─────────────────
    # P_comp + P_cool must not exceed P_T.
    P_total_used   = P_comp + P_cool
    swap_violation = max(0.0, P_total_used - params.P_T)
    swap_penalty   = 1e4 * swap_violation ** 2   # large coefficient → hard barrier

    # ── Thermodynamic cost ───────────────────────────────────────────────────
    cost = effective_error_rate(P_comp, P_cool, params, T_o) + swap_penalty

    return cost


# ═════════════════════════════════════════════════════════════════════════════
# 5.  VQE OPTIMISATION RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_vqe(
    params: StratosphericMissionParams,
    T_o: float,
    n_restarts: int = 5,
    seed: int = 42
) -> dict:
    """
    Execute the VQE optimisation loop with multi-restart L-BFGS-B.

    Multi-restart is critical for VQE because the loss landscape of parameterised
    quantum circuits is non-convex and susceptible to barren plateaus. By sampling
    multiple random initial parameter vectors and keeping the best result, we
    increase the probability of finding the global optimum.

    Parameters
    ----------
    params     : StratosphericMissionParams – mission physical parameters
    T_o        : float – ambient temperature to use                      [K]
    n_restarts : int   – number of random restarts                       (default 5)
    seed       : int   – master random seed for reproducibility

    Returns
    -------
    dict with keys:
        'theta_opt'   – optimal variational parameters (np.ndarray)
        'P_comp_opt'  – optimal computation power      [W]
        'P_cool_opt'  – optimal cooling power          [W]
        'epsilon_opt' – optimal effective error rate
        'cost_history'– list of costs at each optimiser evaluation
        'env_label'   – string label for the thermal environment
    """
    rng = np.random.default_rng(seed)
    n_params = N_QUBITS * 3 * 2   # 4 qubits × 3 Euler angles × 2 layers

    best_result = None
    cost_history = []

    for restart in range(n_restarts):
        # Random initialisation in [−π, π)
        theta_init = rng.uniform(-np.pi, np.pi, size=n_params)

        result = minimize(
            fun     = vqe_cost,
            x0      = theta_init,
            args    = (params, T_o),
            method  = "L-BFGS-B",
            options = {"maxiter": 300, "ftol": 1e-12, "gtol": 1e-8},
            callback= lambda xk: cost_history.append(vqe_cost(xk, params, T_o))
        )

        if best_result is None or result.fun < best_result.fun:
            best_result = result

    # ── Decode optimal parameters ────────────────────────────────────────────
    exp_vals_opt = np.array(vqe_ansatz(best_result.x))
    P_comp_opt, P_cool_opt = decode_expectation_values(
        exp_vals_opt, params.P_T, params.P_comp_min, params.P_cool_min
    )

    epsilon_opt = effective_error_rate(P_comp_opt, P_cool_opt, params, T_o)

    env_label = f"HAP Stratospheric (T_o = {T_o:.0f} K)" \
                if T_o < 260 else f"Terrestrial Ground (T_o = {T_o:.0f} K)"

    return {
        "theta_opt"    : best_result.x,
        "P_comp_opt"   : P_comp_opt,
        "P_cool_opt"   : P_cool_opt,
        "epsilon_opt"  : epsilon_opt,
        "cost_history" : cost_history,
        "env_label"    : env_label,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 6.  ANALYTICAL BASELINE (Classical Brute-Force Grid Search)
# ═════════════════════════════════════════════════════════════════════════════

def classical_grid_search(
    params: StratosphericMissionParams,
    T_o: float,
    n_points: int = 500
) -> Tuple[float, float, float]:
    """
    Brute-force 1D grid search over the power-split fraction f ∈ [0, 1].

    This provides a classical baseline for comparison with the VQE result.
    The search space is 1D because P_comp + P_cool = P_T (fully utilise budget).

    Parameters
    ----------
    params   : StratosphericMissionParams
    T_o      : float – ambient temperature [K]
    n_points : int   – grid resolution

    Returns
    -------
    (P_comp_best, P_cool_best, epsilon_best) : Tuple[float, float, float]
    """
    P_reserved = params.P_comp_min + params.P_cool_min
    P_surplus  = max(0.0, params.P_T - P_reserved)
    f_grid     = np.linspace(0, 1, n_points)

    best_eps  = np.inf
    best_comp = params.P_comp_min
    best_cool = params.P_cool_min

    for f in f_grid:
        P_comp = params.P_comp_min + f * P_surplus
        P_cool = params.P_cool_min + (1 - f) * P_surplus
        eps    = effective_error_rate(P_comp, P_cool, params, T_o)
        if eps < best_eps:
            best_eps  = eps
            best_comp = P_comp
            best_cool = P_cool

    return best_comp, best_cool, best_eps


# ═════════════════════════════════════════════════════════════════════════════
# 7.  VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════

def plot_results(
    result_hap: dict,
    result_gnd: dict,
    params: StratosphericMissionParams
) -> None:
    """
    Generate a four-panel diagnostic figure:
      (A) Cooling overhead vs ambient temperature comparison
      (B) Effective error rate landscape over power-split fractions
      (C) VQE optimisation convergence curves (HAP vs Ground)
      (D) Optimal power allocation bar chart comparison

    Parameters
    ----------
    result_hap : dict – VQE results for HAP stratospheric scenario
    result_gnd : dict – VQE results for terrestrial ground scenario
    params     : StratosphericMissionParams – mission parameters
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "VQE-Cryo-Optimizer: HAP Quantum Data Center Power Allocation\n"
        "Stratospheric vs. Terrestrial Thermal Environment",
        fontsize=14, fontweight="bold", y=0.98
    )
    colours = {"HAP": "#0077BB", "GND": "#EE7733"}

    # ── Panel A: Cooling overhead vs ambient temperature ─────────────────────
    ax = axes[0, 0]
    T_range  = np.linspace(10, 320, 400)
    overhead = [cooling_overhead(0, T, params.T_c, params.eta_c, params.P_deposit)
                for T in T_range]
    ax.plot(T_range, overhead, color="#AA3377", lw=2.0)
    ax.axvline(params.T_o_hap, color=colours["HAP"], ls="--", label=f"HAP  {params.T_o_hap:.0f} K")
    ax.axvline(params.T_o_gnd, color=colours["GND"], ls="--", label=f"GND  {params.T_o_gnd:.0f} K")
    oh_hap = cooling_overhead(0, params.T_o_hap, params.T_c, params.eta_c, params.P_deposit)
    oh_gnd = cooling_overhead(0, params.T_o_gnd, params.T_c, params.eta_c, params.P_deposit)
    ax.axhline(oh_hap, color=colours["HAP"], ls=":", alpha=0.6)
    ax.axhline(oh_gnd, color=colours["GND"], ls=":", alpha=0.6)
    ax.set_xlabel("Ambient Temperature $T_o$ [K]", fontsize=10)
    ax.set_ylabel("Cooling Overhead [W]", fontsize=10)
    ax.set_title("(A)  Cryocooler Wall-Plug Overhead vs. $T_o$", fontsize=10)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.grid(True, alpha=0.3)

    # ── Panel B: Error rate landscape ────────────────────────────────────────
    ax = axes[0, 1]
    P_reserved = params.P_comp_min + params.P_cool_min
    P_surplus  = max(0.0, params.P_T - P_reserved)
    f_vals     = np.linspace(0, 1, 300)
    for T_o, label, col in [
        (params.T_o_hap, "HAP 215 K", colours["HAP"]),
        (params.T_o_gnd, "GND 300 K", colours["GND"])
    ]:
        eps_vals = []
        for f in f_vals:
            P_c = params.P_comp_min + f * P_surplus
            P_k = params.P_cool_min + (1 - f) * P_surplus
            eps_vals.append(effective_error_rate(P_c, P_k, params, T_o))
        ax.plot(f_vals, eps_vals, color=col, lw=2.0, label=label)

    ax.set_xlabel("Power-Split Fraction $f$ (→ computation)", fontsize=10)
    ax.set_ylabel("Effective Error Rate $\\epsilon_{eff}$", fontsize=10)
    ax.set_title("(B)  Error Rate Landscape vs. Power Split", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)

    # ── Panel C: VQE convergence ──────────────────────────────────────────────
    ax = axes[1, 0]
    for res, col in [(result_hap, colours["HAP"]), (result_gnd, colours["GND"])]:
        history = res["cost_history"]
        if history:
            ax.plot(range(len(history)), history, color=col, lw=1.5,
                    alpha=0.85, label=res["env_label"])
    ax.set_xlabel("Optimiser Iteration", fontsize=10)
    ax.set_ylabel("VQE Cost (Effective Error Rate)", fontsize=10)
    ax.set_title("(C)  VQE Convergence History", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)

    # ── Panel D: Optimal power allocation bar chart ───────────────────────────
    ax = axes[1, 1]
    categories = ["P_comp (opt)", "P_cool (opt)", "Remaining"]
    hap_vals = [
        result_hap["P_comp_opt"],
        result_hap["P_cool_opt"],
        max(0, params.P_T - result_hap["P_comp_opt"] - result_hap["P_cool_opt"])
    ]
    gnd_vals = [
        result_gnd["P_comp_opt"],
        result_gnd["P_cool_opt"],
        max(0, params.P_T - result_gnd["P_comp_opt"] - result_gnd["P_cool_opt"])
    ]
    x     = np.arange(len(categories))
    width = 0.35
    bars_hap = ax.bar(x - width/2, hap_vals, width, label="HAP 215 K",
                      color=colours["HAP"], alpha=0.85)
    bars_gnd = ax.bar(x + width/2, gnd_vals, width, label="GND 300 K",
                      color=colours["GND"], alpha=0.85)
    ax.bar_label(bars_hap, fmt="%.0f W", fontsize=8, padding=3)
    ax.bar_label(bars_gnd, fmt="%.0f W", fontsize=8, padding=3)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel("Power [W]", fontsize=10)
    ax.set_title("(D)  Optimal Power Allocation (VQE Result)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("vqe_cryo_optimizer_results.png", dpi=150, bbox_inches="tight")
    print("\n[INFO] Figure saved → vqe_cryo_optimizer_results.png")
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# 8.  MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Main driver: compare VQE-optimised power allocation for
    (a) stratospheric HAP deployment  (T_o = 215 K)
    (b) terrestrial ground baseline   (T_o = 300 K)
    and print a structured mission report to stdout.
    """
    print("=" * 70)
    print("  VQE-Cryo-Optimizer  |  Green Quantum Computing in the Sky")
    print("  High-Altitude Platform Quantum Data Center  —  SWaP Analysis")
    print("=" * 70)

    # Initialise mission parameters
    params = StratosphericMissionParams()

    print(f"\n[PHYSICS] Cooling overhead model:")
    print(f"  P_cool_overhead = (1/eta_c) * ((T_o - T_c) / T_c) * P_deposit")
    print(f"  eta_c     = {params.eta_c:.2f}  (real Carnot efficiency)")
    print(f"  T_c       = {params.T_c:.3f} K  (superconducting cold stage)")
    print(f"  P_deposit = {params.P_deposit:.1f} W  (thermal load on cold stage)")
    print(f"  P_T       = {params.P_T:.0f} W  (total SWaP power budget)")

    oh_hap = cooling_overhead(0, params.T_o_hap, params.T_c, params.eta_c, params.P_deposit)
    oh_gnd = cooling_overhead(0, params.T_o_gnd, params.T_c, params.eta_c, params.P_deposit)

    print(f"\n[THERMAL ENVIRONMENT COMPARISON]")
    print(f"  HAP stratospheric  : T_o = {params.T_o_hap:.0f} K  → overhead = {oh_hap:,.1f} W")
    print(f"  Ground terrestrial : T_o = {params.T_o_gnd:.0f} K  → overhead = {oh_gnd:,.1f} W")
    print(f"  Overhead reduction factor (HAP/GND): {oh_hap/oh_gnd:.4f}  ({(1-oh_hap/oh_gnd)*100:.1f}% saving)")

    # ── Classical baseline ────────────────────────────────────────────────────
    print("\n[CLASSICAL BASELINE — Grid Search]")
    for T_o, label in [(params.T_o_hap, "HAP"), (params.T_o_gnd, "GND")]:
        Pc, Pk, eps = classical_grid_search(params, T_o)
        print(f"  {label}: P_comp={Pc:.1f} W  P_cool={Pk:.1f} W  ε_eff={eps:.4e}")

    # ── VQE Optimisation ─────────────────────────────────────────────────────
    print("\n[VQE OPTIMISATION — Hybrid Quantum-Classical]")
    print(f"  Ansatz   : Hardware-Efficient (HEA), {N_QUBITS} qubits, 2 layers")
    print(f"  Backend  : PennyLane default.qubit (statevector simulation)")
    print(f"  Optimizer: L-BFGS-B (5 random restarts)\n")

    print("  Running HAP scenario ...", end=" ", flush=True)
    result_hap = run_vqe(params, params.T_o_hap, n_restarts=5, seed=42)
    print("done.")

    print("  Running GND scenario ...", end=" ", flush=True)
    result_gnd = run_vqe(params, params.T_o_gnd, n_restarts=5, seed=42)
    print("done.")

    # ── Print mission report ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  MISSION OPTIMISATION REPORT")
    print("=" * 70)

    for res, tag in [(result_hap, "HAP STRATOSPHERIC"), (result_gnd, "TERRESTRIAL GROUND")]:
        print(f"\n  ── {tag} ──")
        print(f"     Environment   : {res['env_label']}")
        print(f"     P_comp (opt)  : {res['P_comp_opt']:.2f} W")
        print(f"     P_cool (opt)  : {res['P_cool_opt']:.2f} W")
        print(f"     P_total used  : {res['P_comp_opt'] + res['P_cool_opt']:.2f} W "
              f"(budget: {params.P_T:.0f} W)")
        print(f"     ε_eff (opt)   : {res['epsilon_opt']:.6e}")

    delta_eps = result_gnd["epsilon_opt"] - result_hap["epsilon_opt"]
    print(f"\n  ── DELTA (HAP vs. GND) ──")
    print(f"     Δε_eff          : {delta_eps:.6e}  (HAP improvement)")
    print(f"     Relative gain   : {delta_eps / result_gnd['epsilon_opt'] * 100:.2f}%")

    # ── Visualisation ─────────────────────────────────────────────────────────
    print("\n[VISUALISATION] Generating diagnostic plots ...")
    plot_results(result_hap, result_gnd, params)

    print("\n[DONE] VQE-Cryo-Optimizer completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
