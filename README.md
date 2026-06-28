# QC-HAP-Decoherence-Simulator

## Stratospheric Noise Modeling and Decoherence Simulation for Aerial Quantum Data Centers

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Google Cirq](https://img.shields.io/badge/Cirq-latest-orange.svg)](https://quantumai.google/cirq)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Research](https://img.shields.io/badge/domain-Quantum%20Aerospace-darkblue)](.)

---

### Abstract
[cite_start]This repository implements a custom quantum noise simulation pipeline for the Green Quantum Computing in the Sky research programme[cite: 98]. [cite_start]The core objective is to quantify the quantum fidelity advantage of deploying superconducting Quantum Data Centers (QDCs) aboard High-Altitude Platforms (HAPs) in the stratosphere[cite: 99]. [cite_start]By extending Google Cirq's native noise frameworks, this project maps ambient atmospheric temperature directly to qubit relaxation ($T_1$) and dephasing ($T_2$) times[cite: 100]. [cite_start]The simulator executes fragile quantum circuits under a custom density matrix environment, analytically proving that the 215 K stratospheric ambient temperature significantly suppresses amplitude and phase damping compared to terrestrial baselines[cite: 101].

---

### Table of Contents
1. [Scientific Motivation](#1-scientific-motivation)
2. [Thermodynamic Noise Model](#2-thermodynamic-noise-model)
3. [Cirq Implementation Architecture](#3-cirq-implementation-architecture)
4. [Repository Structure](#4-repository-structure)
5. [Installation](#5-installation)
6. [Running the Simulation](#6-running-the-simulation)
7. [Interpreting the Output](#7-interpreting-the-output)
8. [cite_start][References](#8-references) [cite: 102]

---

### 1. Scientific Motivation

**1.1 The Threat of Thermal Decoherence**
[cite_start]In superconducting quantum hardware, ambient heat increases the thermal population of the environment, accelerating energy loss (Amplitude Damping) and phase loss (Phase Damping)[cite: 102]. [cite_start]Heat essentially creates atomic kinetic energy, causing physical vibrations (phonons) that collide with qubits[cite: 103]. [cite_start]This interaction acts as an "observation" from the environment, collapsing the delicate quantum state back into a classical state—a process known as decoherence[cite: 104].

**1.2 The Stratospheric Advantage**
[cite_start]Operating at an altitude of approximately 20 km, HAPs are exposed to an ambient temperature of approximately 215 K (compared to 300 K at sea level)[cite: 105, 106]. [cite_start]This naturally sub-zero environment imposes a significantly smaller "thermal penalty" on the cryogenic payload[cite: 106]. [cite_start]By reducing the thermal gap the cryocooler must bridge, the system achieves longer effective $T_1$ and $T_2$ times, providing a inherently quieter environment for quantum logic gates to operate[cite: 107].

---

### 2. Thermodynamic Noise Model

**2.1 Phenomenological Thermal Mapping**
[cite_start]To bridge aerospace thermodynamics and quantum mechanics, this simulator establishes a thermal penalty ratio mapping the ambient temperature ($T_o$) to baseline coherence metrics[cite: 108]. The effective coherence times are derived as:

$$T_{1,\text{eff}} = \frac{T_{1,\text{base}}}{(T_o / 300)}$$
[cite_start]$$T_{2,\text{eff}} = \frac{T_{2,\text{base}}}{(T_o / 300)}$$ [cite: 109]

**2.2 Quantum Channel Probabilities**
The probability of a qubit spontaneously losing its excited state (Amplitude Damping, $\gamma$) over a specific gate time ($t_g$) is defined by:

[cite_start]$$\gamma = 1 - e^{-t_g / T_{1,\text{eff}}}$$ [cite: 110]

The probability of a qubit losing its phase relationship (Phase Damping, $\lambda$) is defined by:

[cite_start]$$\lambda = 1 - e^{-t_g / T_{2,\text{eff}}}$$ [cite: 110]

---

### 3. Cirq Implementation Architecture

**3.1 Custom `cirq.NoiseModel`**
[cite_start]Rather than applying generic depolarization, this project subclasses `cirq.NoiseModel` to build the `StratosphericNoiseModel`[cite: 110]. [cite_start]This class dynamically calculates $\gamma$ and $\lambda$ based on the input atmospheric temperature and injects `cirq.amplitude_damp` and `cirq.phase_damp` channels into every operation within the circuit[cite: 111].

**3.2 Density Matrix Simulation**
[cite_start]Because amplitude and phase damping are non-unitary operations that convert pure quantum states into mixed states, standard state-vector simulators cannot mathematically resolve the circuit[cite: 112]. [cite_start]This project utilizes `cirq.DensityMatrixSimulator` to accurately model the thermodynamic decay of the quantum states over 1,000+ simulation repetitions[cite: 113].

---

### 4. Repository Structure
```text
QC-HAP-Decoherence-Simulator/
├── stratospheric_noise.py     # Core custom NoiseModel and thermal physics classes
├── simulate_hap.py            # Main execution script for Bell State fidelity tests
├── requirements.txt           # Python dependency specification
├── README.md                  # This document
└── coherence_comparison.png   # Generated fidelity graphs (HAP vs Terrestrial)

http://googleusercontent.com/immersive_entry_chip/0
http://googleusercontent.com/immersive_entry_chip/1
http://googleusercontent.com/immersive_entry_chip/2

---

### 7. Interpreting the Output

The simulator outputs a diagnostic histogram comparing the measurement probabilities[cite: 115].
* **The Ideal State:** Without noise, a Bell State measurement should yield a perfect 50/50 split between $|00\rangle$ and $|11\rangle$, with zero counts in $|01\rangle$ and $|10\rangle$[cite: 116].
* **Terrestrial (300 K):** High amplitude damping pushes the distribution heavily toward the $|00\rangle$ ground state, breaking the entanglement[cite: 117].
* **Stratospheric (215 K):** The reduced thermal penalty suppresses the damping channels, resulting in a histogram that remains significantly closer to the ideal 50/50 split[cite: 118].

---

### 8. References
1. Krantz, P. et al. (2019). A quantum engineer's guide to superconducting qubits. Applied Physics Reviews, 6, 021318. [cite: 119, 120]
2. Google Quantum AI. (2023). Cirq Documentation: Noise and Density Matrices. [cite: 120]
3. Preskill, J. (2018). Quantum Computing in the NISQ Era and Beyond. Quantum, 2, 79. [cite: 121, 122]

---

### License
This project is licensed under the MIT License. See `LICENSE` for details[cite: 122].
