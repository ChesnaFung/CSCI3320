"""
CSCI3320 Assignment 1 Answers

Grok. (2026, Feb 1). Assistance with grammar correction and content guidance. Retrieved from https://grok.x.ai/.
    [Note: Content reviewed and modified by Fung Cheuk Nam to ensure originality.]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pla(X, y, max_iter=10000):
    """
    Perceptron Learning Algorithm.
    - X: Features with bias (N x 3, where column 0 is 1s).
    - y: Labels (+1 or -1).
    - Returns: final weights w, list of errors (misclassifications per iteration).
    """
    N, d = X.shape  # N samples, d features (including bias)
    w = np.zeros(d)  # Initialize weights to zero
    errors = []  # Track misclassifications per iteration
    
    for iteration in range(max_iter):
        misclassified = 0
        for i in range(N):
            prediction = np.sign(np.dot(w, X[i]))
            if prediction != y[i]:  # Misclassified
                w += y[i] * X[i]  # Update rule
                misclassified += 1
        errors.append(misclassified)
        if misclassified == 0:
            print(f"Converged after {iteration + 1} iterations.")
            break
    else:
        print(f"Did not converge after {max_iter} iterations (likely non-separable).")
    
    return w, errors

def plot_error(errors, title, filename):
    """Plot misclassifications vs iterations."""
    plt.figure()
    plt.plot(errors)
    plt.xlabel('Iterations')
    plt.ylabel('Number of Misclassifications')
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_boundary(X, y, w, title, filename):
    """Plot data points and decision boundary."""
    plt.figure()
    # Scatter points: +1 blue, -1 red
    plt.scatter(X[y == 1, 1], X[y == 1, 2], color='blue', label='+1')
    plt.scatter(X[y == -1, 1], X[y == -1, 2], color='red', label='-1')
    
    # Decision boundary: w0 + w1*x1 + w2*x2 = 0 => x2 = -(w0 + w1*x1)/w2
    x1_min, x1_max = np.min(X[:, 1]), np.max(X[:, 1])
    x1 = np.linspace(x1_min - 1, x1_max + 1, 100)  # Buffer for visibility
    if abs(w[2]) > 1e-8:  # Avoid division by zero
        x2 = -(w[0] + w[1] * x1) / w[2]
    else:
        x2 = np.zeros_like(x1)  # Fallback (vertical line unlikely)
    plt.plot(x1, x2, color='black', label='Decision Boundary')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

# Load and process a dataset
def load_dataset(filename):
    df = pd.read_csv(filename)
    X = df[['Feature 1', 'Feature 2']].values  # Features
    y = df['Target'].values  # Targets: 0 or 1
    y = np.where(y == 0, -1, 1)  # Map 0 -> -1, 1 -> +1
    N = X.shape[0]
    X = np.c_[np.ones(N), X]  # Add bias column (1s)
    return X, y

# Main execution
if __name__ == "__main__":
    # Linearly separable
    X_sep, y_sep = load_dataset("linearly_separable_data.csv")
    w_sep, errors_sep = pla(X_sep, y_sep)
    plot_error(errors_sep, 'Training Error (Linearly Separable)', 'error_separable.png')
    plot_boundary(X_sep, y_sep, w_sep, 'Decision Boundary (Linearly Separable)', 'boundary_separable.png')
    
    # Linearly inseparable
    X_insep, y_insep = load_dataset("linearly_inseparable_data.csv")
    w_insep, errors_insep = pla(X_insep, y_insep)
    plot_error(errors_insep, 'Training Error (Non-Linearly Separable)', 'error_inseparable.png')
    plot_boundary(X_insep, y_insep, w_insep, 'Decision Boundary (Non-Linearly Separable)', 'boundary_inseparable.png')
    
    
    