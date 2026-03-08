import numpy as np
import matplotlib.pyplot as plt

def load_data(x_file, y_file):
    # Load data and convert to numpy arrays
    x = np.loadtxt(x_file)
    y = np.loadtxt(y_file)
    # Create design matrix X, prepend a column of ones for the bias term (w0)
    X = np.column_stack((np.ones(len(x)), x))
    return x, y, X

def compute_residual_error(y, y_hat):
    # Calculate average residual error: 1/N * sum((y_i - y_hat_i)^2)
    return np.mean((y - y_hat)**2)

def solve_linear_regression(x_file, y_file, label_name):
    x, y, X = load_data(x_file, y_file)
    N = len(y)
    
    # --- 1. Theoretical Solution ---
    # Closed-form solution: w = (X^T * X)^-1 * X^T * y
    w_theory = np.linalg.inv(X.T @ X) @ X.T @ y
    y_hat_theory = X @ w_theory
    error_theory = compute_residual_error(y, y_hat_theory)
    
    print(f"\n--- {label_name} ---")
    print(f"Theoretical Weights (w0, w1): {w_theory}")
    print(f"Theoretical Residual Error: {error_theory:.6f}")

    # --- 2. Gradient Descent ---
    w = np.zeros(2) # Initial parameters w0=0, w1=0
    learning_rate = 0.01 # eta = 0.01
    tolerance = 0.0001 # Stop when Euclidean norm of gradient < 0.0001
    errors_history = []
    
    while True:
        y_hat = X @ w
        # Gradient of Loss L = 1/(2N) * sum((y - y_hat)^2)
        gradient = -(1/N) * (X.T @ (y - y_hat))
        
        # Update weights
        w = w - learning_rate * gradient
        
        # Record current residual error
        current_error = compute_residual_error(y, y_hat)
        errors_history.append(current_error)
        
        # Check convergence condition
        if np.linalg.norm(gradient) < tolerance:
            break

    print(f"GD Final Weights (w0, w1): {w}")
    print(f"GD Final Residual Error: {compute_residual_error(y, X @ w):.6f}")
    
    return errors_history, error_theory

def save_individual_plot(errors, theory_val, filename, title):
    plt.figure(figsize=(8, 5))
    plt.plot(errors, label='GD Residual Error', color='blue')
    plt.axhline(y=theory_val, color='red', linestyle='--', label='Theoretical Minimum')
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Residual Error')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(filename, dpi=300)
    print(f"Successfully saved plot to: {filename}")
    plt.close() # Close figure to free memory

# Execute Part 1: iid Noise
errors_q1, theory_q1 = solve_linear_regression('data_x_q1.txt', 'data_y_q1.txt', 'Part 1: iid Noise')
save_individual_plot(errors_q1, theory_q1, 'residual_error_q1.png', 'Part 1: iid Noise Convergence')

# Execute Part 2: Correlated Noise
errors_q2, theory_q2 = solve_linear_regression('data_x_q2.txt', 'data_y_q2.txt', 'Part 2: Correlated Noise')
save_individual_plot(errors_q2, theory_q2, 'residual_error_q2.png', 'Part 2: Correlated Noise Convergence')