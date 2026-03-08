import numpy as np
import matplotlib.pyplot as plt

# Parameters
K = 10000
x_grid = np.linspace(-1, 1, 1000)
f_x = x_grid**3

slopes = []
intercepts = []
e_out_list = []

for _ in range(K):
    # 1. Sample 2 points uniformly from [-1, 1]
    x_samp = np.random.uniform(-1, 1, 2)
    y_samp = x_samp**3
    
    # 2. Fit line g(x) = ax + b
    a = (y_samp[1] - y_samp[0]) / (x_samp[1] - x_samp[0])
    b = y_samp[0] - a * x_samp[0]
    slopes.append(a)
    intercepts.append(b)
    
    # 3. Numerical E_out for this specific g(x)
    g_x = a * x_grid + b
    e_out_list.append(np.mean((g_x - f_x)**2))

# Numerical Estimates
a_bar = np.mean(slopes)
b_bar = np.mean(intercepts)
g_bar_x = a_bar * x_grid + b_bar

avg_e_out = np.mean(e_out_list)
bias_x = (g_bar_x - f_x)**2
bias = np.mean(bias_x)

# Variance: E_D[(g_D(x) - g_bar(x))^2]
var_x = np.mean([(slopes[i]*x_grid + intercepts[i] - g_bar_x)**2 for i in range(K)], axis=0)
var = np.mean(var_x)

print(f"Numerical Results (K={K}):")
print(f"E_out: {avg_e_out:.4f}")
print(f"Bias:  {bias:.4f}")
print(f"Var:   {var:.4f}")
print(f"Bias + Var: {bias + var:.4f}")

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(x_grid, f_x, 'k-', label='$f(x) = x^3$', linewidth=2)
plt.plot(x_grid, g_bar_x, 'r--', label=r'$\bar{g}(x) \approx \frac{2}{3}x$')
plt.title("Target Function vs. Average Hypothesis")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()