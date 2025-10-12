"""
Contextual Bandit Learning: Exploration vs Exploitation
Implementation of ε-Greedy, Linear UCB, and Thompson Sampling algorithms
with comparative analysis and visualizations
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from sklearn.linear_model import LinearRegression
import matplotlib.backends.backend_pdf

# Set random seed for reproducibility
np.random.seed(42)


# =====================================================================
# 1. CONTEXTUAL BANDIT ENVIRONMENT SETUP
# =====================================================================
def create_environment(n_rounds=500, n_features=2, n_arms=3, noise=0.1):
    """
    Create simulated contextual bandit environment
    - contexts: Binary feature vectors (user attributes/environment)
    - weights: True reward weights for each arm (hidden from algorithms)
    - true_rewards: Actual rewards for each context-arm pair
    """
    # Generate random binary contexts
    contexts = np.random.randint(0, 2, (n_rounds, n_features))

    # Define true reward weights (hidden from algorithms)
    weights = np.array([
        [0.7, 0.3],  # Arm 0 weights
        [0.4, 0.6],  # Arm 1 weights
        [0.2, 0.8]  # Arm 2 weights
    ])

    # Calculate true rewards (linear combination + noise)
    true_rewards = np.dot(contexts, weights.T) + np.random.normal(0, noise, (n_rounds, n_arms))

    return contexts, true_rewards


# =====================================================================
# 2. ε-GREEDY ALGORITHM IMPLEMENTATION
# =====================================================================
def epsilon_greedy(contexts, true_rewards, epsilon=0.1):
    """
    ε-Greedy contextual bandit algorithm
    - With probability (1-ε): Exploit best-known action
    - With probability ε: Explore random action
    """
    n_rounds, n_arms = true_rewards.shape
    n_features = contexts.shape[1]

    # Initialize reward estimates and counts
    estimated_rewards = np.zeros((n_arms, n_features))
    counts = np.ones((n_arms, n_features))  # Avoid division by zero

    # Tracking variables
    cumulative_reward = 0
    optimal_cumulative = 0
    cumulative_regret = []
    chosen_arms = []

    for t in range(n_rounds):
        context = contexts[t]
        true_reward = true_rewards[t]

        # Predict rewards for current context
        predicted = np.sum(estimated_rewards * context, axis=1)

        # Exploration-exploitation tradeoff
        if np.random.rand() < epsilon:
            chosen_arm = np.random.choice(n_arms)  # Explore
        else:
            chosen_arm = np.argmax(predicted)  # Exploit

        # Get actual reward
        reward = true_reward[chosen_arm]
        cumulative_reward += reward
        optimal_reward = np.max(true_reward)
        optimal_cumulative += optimal_reward

        # Update estimates (incremental average)
        counts[chosen_arm] += context
        error = reward - np.dot(estimated_rewards[chosen_arm], context)
        estimated_rewards[chosen_arm] += context * error / counts[chosen_arm]

        # Track metrics
        cumulative_regret.append(optimal_cumulative - cumulative_reward)
        chosen_arms.append(chosen_arm)

    return cumulative_regret, chosen_arms, cumulative_reward


# =====================================================================
# 3. LINEAR UCB ALGORITHM IMPLEMENTATION
# =====================================================================
def linear_ucb(contexts, true_rewards, alpha=0.5):
    """
    Linear Upper Confidence Bound (UCB) algorithm
    - Uses confidence bounds for optimistic exploration
    - Balances exploration vs exploitation mathematically
    """
    n_rounds, n_arms = true_rewards.shape
    n_features = contexts.shape[1]

    # Initialize matrices (one per arm)
    A = [np.eye(n_features) for _ in range(n_arms)]  # Covariance matrices
    b = [np.zeros(n_features) for _ in range(n_arms)]  # Reward vectors

    # Tracking variables
    cumulative_reward = 0
    optimal_cumulative = 0
    cumulative_regret = []
    chosen_arms = []

    for t in range(n_rounds):
        context = contexts[t]
        true_reward = true_rewards[t]

        ucb_values = []
        for a in range(n_arms):
            # Calculate parameter estimates
            A_inv = inv(A[a])
            theta = A_inv @ b[a]

            # Compute UCB: estimated reward + confidence bound
            uncertainty = alpha * np.sqrt(context @ A_inv @ context)
            ucb = theta @ context + uncertainty
            ucb_values.append(ucb)

        # Choose arm with highest UCB
        chosen_arm = np.argmax(ucb_values)
        reward = true_reward[chosen_arm]
        cumulative_reward += reward
        optimal_cumulative += np.max(true_reward)

        # Update model
        A[chosen_arm] += np.outer(context, context)
        b[chosen_arm] += reward * context

        # Track metrics
        cumulative_regret.append(optimal_cumulative - cumulative_reward)
        chosen_arms.append(chosen_arm)

    return cumulative_regret, chosen_arms, cumulative_reward


# =====================================================================
# 4. THOMPSON SAMPLING IMPLEMENTATION
# =====================================================================
def thompson_sampling(contexts, true_rewards):
    """
    Thompson Sampling (Bayesian approach)
    - Uses posterior sampling for exploration
    - Naturally balances exploration-exploitation via uncertainty
    """
    n_rounds, n_arms = true_rewards.shape
    n_features = contexts.shape[1]

    # Initialize Bayesian priors
    mu = [np.zeros(n_features) for _ in range(n_arms)]  # Mean vectors
    cov = [np.eye(n_features) for _ in range(n_arms)]  # Covariance matrices

    # Tracking variables
    cumulative_reward = 0
    optimal_cumulative = 0
    cumulative_regret = []
    chosen_arms = []

    for t in range(n_rounds):
        context = contexts[t]
        true_reward = true_rewards[t]

        # Sample from posterior distributions
        sampled_vals = [np.random.multivariate_normal(mu[a], cov[a]) for a in range(n_arms)]
        theta_vals = [s @ context for s in sampled_vals]

        # Choose arm with highest sampled value
        chosen_arm = np.argmax(theta_vals)
        reward = true_reward[chosen_arm]
        cumulative_reward += reward
        optimal_cumulative += np.max(true_reward)

        # Bayesian update (posterior)
        cov_inv = inv(cov[chosen_arm])
        new_cov_inv = cov_inv + np.outer(context, context)
        cov[chosen_arm] = inv(new_cov_inv)

        residual = reward - mu[chosen_arm] @ context
        mu[chosen_arm] += cov[chosen_arm] @ context * residual

        # Track metrics
        cumulative_regret.append(optimal_cumulative - cumulative_reward)
        chosen_arms.append(chosen_arm)

    return cumulative_regret, chosen_arms, cumulative_reward


# =====================================================================
# 5. VISUALIZATION AND REPORTING FUNCTIONS
# =====================================================================
def plot_cumulative_regret(regrets, labels):
    """Plot cumulative regret comparison"""
    plt.figure(figsize=(10, 6))
    for regret, label in zip(regrets, labels):
        plt.plot(regret, label=label)

    plt.title("Cumulative Regret Comparison")
    plt.xlabel("Rounds")
    plt.ylabel("Cumulative Regret")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_arm_selections(chosen_arms_list, labels):
    """Visualize arm selection patterns"""
    n_algorithms = len(chosen_arms_list)
    fig, axes = plt.subplots(n_algorithms, 1, figsize=(10, 3 * n_algorithms), sharex=True)

    for i, (arms, label) in enumerate(zip(chosen_arms_list, labels)):
        axes[i].scatter(range(len(arms)), arms, s=20, alpha=0.7)
        axes[i].set_title(f"{label} Arm Selections")
        axes[i].set_ylabel("Arm")
        axes[i].set_yticks([0, 1, 2])
        axes[i].grid(True)

    plt.xlabel("Rounds")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(contexts, rewards):
    """Analyze context feature importance"""
    # Target = optimal reward per context
    y = np.max(rewards, axis=1)

    model = LinearRegression().fit(contexts, y)
    importance = np.abs(model.coef_)

    plt.figure(figsize=(8, 4))
    plt.bar([f"Feature {i + 1}" for i in range(len(importance))], importance, color='skyblue')
    plt.title("Context Feature Importance")
    plt.ylabel("Importance Coefficient")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()


def generate_pdf_report(figures, filename="contextual_bandits.pdf"):
    """Export all plots to PDF"""
    pdf = matplotlib.backends.backend_pdf.PdfPages(filename)
    for fig in figures:
        pdf.savefig(fig)
    pdf.close()


# =====================================================================
# 6. MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    # Create environment
    contexts, true_rewards = create_environment(n_rounds=500)

    # Run all algorithms
    eg_regret, eg_arms, eg_reward = epsilon_greedy(contexts, true_rewards)
    ucb_regret, ucb_arms, ucb_reward = linear_ucb(contexts, true_rewards)
    ts_regret, ts_arms, ts_reward = thompson_sampling(contexts, true_rewards)

    # Comparative analysis
    print(f"\n{' Algorithm ':-^40}")
    print(f"ε-Greedy Total Reward: {eg_reward:.2f}")
    print(f"Linear UCB Total Reward: {ucb_reward:.2f}")
    print(f"Thompson Sampling Total Reward: {ts_reward:.2f}")

    # Visualization
    plot_cumulative_regret(
        [eg_regret, ucb_regret, ts_regret],
        ["ε-Greedy", "Linear UCB", "Thompson Sampling"]
    )

    plot_arm_selections(
        [eg_arms, ucb_arms, ts_arms],
        ["ε-Greedy", "Linear UCB", "Thompson Sampling"]
    )

    plot_feature_importance(contexts, true_rewards)

    # Save final figures to PDF
    fig1 = plt.figure()
    plt.plot(eg_regret, label="ε-Greedy")
    plt.plot(ucb_regret, label="Linear UCB")
    plt.plot(ts_regret, label="Thompson Sampling")
    plt.title("Cumulative Regret Comparison")
    plt.legend()

    fig2, ax = plt.subplots(3, 1, figsize=(10, 8))
    for i, (arms, name) in enumerate(zip([eg_arms, ucb_arms, ts_arms],
                                         ["ε-Greedy", "Linear UCB", "Thompson Sampling"])):
        ax[i].scatter(range(len(arms)), arms, s=5)
        ax[i].set_title(f"{name} Arm Selection")

    generate_pdf_report([fig1, fig2])
    print("\nReport generated: contextual_bandits.pdf")