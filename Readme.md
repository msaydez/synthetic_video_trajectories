# Synthetic Skier Trajectory Simulation

This repository contains the code used to generate **synthetic skier trajectories** using a cellular automaton model and to evaluate them against real data using the **Wasserstein distance**.

The project was developed for generating synthetic trajectory data for ski slope environments and evaluating its similarity to real-world trajectories.

---

# Files

### Simulation

This script runs the **cellular automaton simulation** and generates synthetic skier trajectories.

The simulation models skier motion on a slope using:

- slope dynamics
- skier ability levels
- behavioral parameters
- multi-skier interactions

Running this script produces synthetic trajectories that can later be evaluated.

---

### Evaluation

This generates synthetic skier trajectories.

---

### 2. Compute Wasserstein distance


This evaluates the similarity between the synthetic trajectories and the real dataset.

---

# Purpose

The goal of this project is to explore **synthetic trajectory generation** as a way to reduce the need for manually annotated trajectory data in ski slope environments.

---

# Notes

- The repository focuses on **trajectory generation and evaluation** rather than full visual rendering.
- The generated trajectories are intended for **trajectory analysis and tracking research**.