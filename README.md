# Contextual Bandit Learning: Exploration and Exploitation with Context

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NumPy](https://img.shields.io/badge/Library-NumPy-013243)
![Pandas](https://img.shields.io/badge/Library-Pandas-150458)
![Status](https://img.shields.io/badge/Status-Research_POC-orange)

> **Master of Engineering in Information Technology**
> **Frankfurt University of Applied Sciences**
> **Author:** Manoj Hanumanthu (RegNo. 1566325)

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [The Contextual Bandit Problem](#-the-contextual-bandit-problem)
3. [Algorithms Implemented](#-algorithms-implemented)
4. [Simulation & Methodology](#-simulation--methodology)
5. [Performance Analysis](#-performance-analysis)
6. [Installation & Usage](#-installation--usage)
7. [Visualizations](#-visualizations)
8. [References](#-references)

---

## 📄 Project Overview

In the era of big data, decision-making systems must intelligently adapt to specific situational cues. [cite_start]Whether choosing a video on a streaming platform or selecting a treatment in a medical system, decisions must be made with **context** in mind[cite: 33, 34].

[cite_start]This project implements and compares three popular algorithms—**ε-Greedy**, **Linear UCB**, and **Thompson Sampling**—to solve the "Exploration-Exploitation Dilemma" under context-dependent rewards[cite: 26]. [cite_start]Using a simulated environment with binary contextual features, we analyze how each algorithm balances exploring new actions versus exploiting known rewards to minimize cumulative regret[cite: 27].

**Key Findings:**
* [cite_start]**ε-Greedy:** A simple baseline that struggles with dynamic learning due to static exploration[cite: 28].
* [cite_start]**Linear UCB:** Provides the most consistent learning efficiency and lowest regret[cite: 28].
* [cite_start]**Thompson Sampling:** Allows for adaptive exploration via Bayesian methods but is sensitive to stochastic variability[cite: 28].

---

## 🎯 The Contextual Bandit Problem


[cite_start]Unlike traditional Multi-Armed Bandits (MAB) which assume a stationary environment, Contextual Bandits receive a **context vector** ($x_t$) before selecting an action[cite: 37, 39].

### Mathematical Formulation
The goal is to maximize total reward (or minimize regret) over time.
* **Context ($x_t$):** Observed features (e.g., user attributes).
* **Action ($a_t$):** The arm chosen by the agent.
* **Reward ($r_t$):** The feedback received, which depends on both context and action.

The objective is to minimize **Cumulative Regret** ($R_T$):
$$R_T = \sum_{t=1}^{T} (E[r_t^{a^*}|x_t] - E[r_t^a|x_t])$$
[cite_start]Where $a^*$ is the optimal action for context $x_t$[cite: 139].

---

## 🧠 Algorithms Implemented

### 1. ε-Greedy (Epsilon-Greedy)
[cite_start]An intuitive method that mostly exploits the best-known action but explores randomly with a small probability $\epsilon$[cite: 50].
* **Mechanism:** Estimates rewards using linear models. [cite_start]With probability $1-\epsilon$, it chooses the best action; with probability $\epsilon$, it chooses a random action[cite: 168, 169].
* [cite_start]**Limitation:** It lacks a mechanism to estimate confidence, leading to inefficient exploration in complex contexts[cite: 179].

### 2. Linear Upper Confidence Bound (Linear UCB)
A principled approach based on "optimism in the face of uncertainty." [cite_start]It maintains confidence intervals for rewards and selects actions based on the upper end of this interval[cite: 52].
* **Selection Rule:**
    $$a_t = \text{argmax}_a (x_t^\top \hat{\theta}_a + \alpha \sqrt{x_t^\top A_a^{-1} x_t})$$
* [cite_start]**Strength:** Naturally shifts from exploration to exploitation as uncertainty decreases[cite: 203].

### 3. Thompson Sampling
[cite_start]A Bayesian approach that maintains a posterior distribution of reward parameters for each arm[cite: 54].
* [cite_start]**Mechanism:** It samples parameters from the posterior and chooses the arm with the highest expected reward based on that sample[cite: 223].
* [cite_start]**Strength:** Exploration is guided probabilistically by uncertainty[cite: 226].

---

## ⚙️ Simulation & Methodology

[cite_start]We developed a **Toy Dataset** to emulate real-world decision-making (e.g., ads, clinical diagnostics)[cite: 240].

### Dataset Setup
* [cite_start]**Contexts:** Sampled from a fixed-dimensional space (5-10 dimensions)[cite: 244].
* **Reward Model:** Linear function with Gaussian noise.
    $$r_t^a = x_t^\top \theta_a + \eta_t$$
    [cite_start]Where $\eta_t \sim N(0, \sigma^2)$ represents noise[cite: 255].
* [cite_start]**Arm-Reward Matrix:** Predefined weight vectors $\theta_a$ serve as the "ground truth" to calculate optimal regret[cite: 260].

### Exploration Parameters
| Algorithm | Parameter | Role | Impact |
| :--- | :--- | :--- | :--- |
| **ε-Greedy** | $\epsilon$ (Epsilon) | Random exploration rate | [cite_start]High $\epsilon$ delays convergence; low $\epsilon$ exploits too early[cite: 391]. |
| **Linear UCB** | $\alpha$ (Alpha) | Confidence scaling | [cite_start]Tuned to $\alpha=1.0$ for balanced convergence[cite: 372]. |
| **Thompson Sampling** | Prior, $\sigma^2$ | Bayesian mechanism | [cite_start]Sensitive to noise estimation[cite: 391]. |

---

## 📊 Performance Analysis

We evaluated the algorithms over thousands of rounds. [cite_start]The results highlight the "Performance-Simplicity Trade-off"[cite: 30].

### 1. Cumulative Regret Comparison
Linear UCB achieved the lowest cumulative regret, indicating it learned the optimal strategy fastest.

| Algorithm | Final Cumulative Regret | Assessment |
| :--- | :--- | :--- |
| **ε-Greedy** | ~2.1 | [cite_start]Suboptimal due to fixed exploration[cite: 440]. |
| **Linear UCB** | **~1.5** | [cite_start]**Best Performance.** Efficient learning via confidence bounds[cite: 440]. |
| **Thompson Sampling** | ~2.6 | [cite_start]Good adaptive behavior but sensitive to priors[cite: 440]. |

### 2. Total Reward Accumulated
Consistent with regret metrics, Linear UCB maximized the total reward collected.

| Algorithm | Total Reward |
| :--- | :--- |
| ε-Greedy | [cite_start]4.1 [cite: 521] |
| **Linear UCB** | [cite_start]**4.8** [cite: 521] |
| Thompson Sampling | [cite_start]3.8 [cite: 521] |

### 3. Feature Importance (Interpretability)
Using linear regression approximation, we analyzed which context features drove decisions.
* [cite_start]**Feature 1:** Weight ~0.57 (Primary driver)[cite: 545].
* [cite_start]**Feature 2:** Weight ~0.43 (Secondary driver)[cite: 547].
[cite_start]This transparency is crucial for applications in healthcare and finance[cite: 514].

---

## 💻 Installation & Usage

### Prerequisites
* Python 3.x
* Libraries: `numpy`, `pandas`, `matplotlib`

### Running the Code
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Hanumanthumanoj01/Contextual-Bandit-Learning.git](https://github.com/Hanumanthumanoj01/Contextual-Bandit-Learning.git)
    cd Contextual-Bandit-Learning/POC
    ```

2.  **Install Dependencies:**
    ```bash
    pip install numpy pandas matplotlib
    ```

3.  **Run the Simulation:**
    *(Note: Replace `main.py` with the actual entry point of your script if different)*
    ```bash
    python main.py
    ```

---

## 📈 Visualizations

### Cumulative Regret
*Comparison of how algorithms minimize regret over time.*
> *[Insert Fig 7 from report: Cumulative Regret Comparison on Toy Dataset]*
> *Linear UCB (Orange) flattens out fastest, indicating convergence.*

### Arm Selection Behavior
*How algorithms shift from exploration to exploitation.*
> *[Insert Fig 8 from report: Arm Selection Behavior]*
> *Linear UCB rapidly converges to the optimal arm (Arm 0), while ε-Greedy continues to have noise.*

### Feature Importance
*The weight of different context features in decision making.*
> *[Insert Fig 9 from report: Context Feature Importance]*

---

## 📚 References

1.  [cite_start]J. Zheng et al., “Neural Contextual Combinatorial Bandit under Non-stationary Environment,” IEEE ICDM, 2023. [cite: 635]
2.  [cite_start]Z. Qu et al., “Context-Aware Online Client Selection,” IEEE TPDS, 2022. [cite: 636]
3.  [cite_start]C. Wang et al., “Adaptive Noise Exploration for Neural Contextual Multi-Armed Bandits,” Algorithms, 2025. [cite: 638]
4.  [cite_start]J. Y. Wang et al., “Identifying general reaction conditions by bandit optimization,” Nature, 2024. [cite: 654]
5.  [cite_start]A. Sekhari et al., “Contextual Bandits and Imitation Learning with Preference-Based Active Queries”. [cite: 669]

---
*Disclaimer: This POC uses a synthetic toy dataset to demonstrate algorithmic behaviors. Real-world performance may vary based on data sparsity and noise levels.*
