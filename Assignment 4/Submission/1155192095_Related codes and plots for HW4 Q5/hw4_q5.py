import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Data
def load_data():
    print("Loading datasets...")
    x_train = pd.read_csv('q5_x_train.csv', header=None).values
    y_train = pd.read_csv('q5_y_train.csv', header=None).values.flatten()
    x_test = pd.read_csv('q5_x_test.csv', header=None).values
    y_test = pd.read_csv('q5_y_test.csv', header=None).values.flatten()
    
    # Add bias term (column of 1s)
    X_train = np.column_stack([np.ones(x_train.shape[0]), x_train])
    X_test = np.column_stack([np.ones(x_test.shape[0]), x_test])
    return X_train, y_train, X_test, y_test

# 2. Loss and Training Functions
def compute_loss(X, y, w, lam=0):
    m = X.shape[0]
    z = np.clip(y * (X @ w), -500, 500)
    loss = np.mean(np.log(1 + np.exp(-z)))
    if lam > 0:
        loss += (lam / (2 * m)) * np.sum(w[1:]**2) # w0 is not regularized
    return loss

def train(X, y, lam=0, eta=0.01, tol=1e-2):
    m, n = X.shape
    w = np.zeros(n)
    losses = []
    while True:
        z = np.clip(y * (X @ w), -500, 500)
        # Gradient formula from handout
        grad = -(1/m) * ((y / (1 + np.exp(z)))[:, None] * X).sum(axis=0)
        
        if lam > 0:
            reg_grad = (lam / m) * w
            reg_grad[0] = 0 # No regularization for bias
            grad += reg_grad
        
        losses.append(compute_loss(X, y, w, lam))
        if np.linalg.norm(grad) < tol:
            break
        w -= eta * grad
    return w, losses

def get_accuracy(X, y, w):
    preds = np.where(X @ w >= 0, 1, -1)
    return np.mean(preds == y)

# 3. Execution and Reporting
X_train, y_train, X_test, y_test = load_data()

# Part 1: No Regularization
print("Running Part 1 (No Regularization)...")
w1, losses1 = train(X_train, y_train, lam=0)

# Part 2: L2 Regularization
print("Running Part 2 (L2 Regularization, lambda=1)...")
w2, losses2 = train(X_train, y_train, lam=1)

# --- PRINT DATA RESULTS ---
print("\n" + "="*50)
print("FINAL RESULTS FOR YOUR REPORT")
print("="*50)

# Results for Part 1
acc_train1 = get_accuracy(X_train, y_train, w1)
acc_test1 = get_accuracy(X_test, y_test, w1)
norm1 = np.linalg.norm(w1[1:])
print(f"PART 1 (lambda=0):")
print(f"  - Training Accuracy: {acc_train1*100:.2f}%")
print(f"  - Test Accuracy:     {acc_test1*100:.2f}%")
print(f"  - Weight Norm ||w||: {norm1:.4f}")

print("-" * 50)

# Results for Part 2
acc_train2 = get_accuracy(X_train, y_train, w2)
acc_test2 = get_accuracy(X_test, y_test, w2)
norm2 = np.linalg.norm(w2[1:])
print(f"PART 2 (lambda=1):")
print(f"  - Training Accuracy: {acc_train2*100:.2f}%")
print(f"  - Test Accuracy:     {acc_test2*100:.2f}%")
print(f"  - Weight Norm ||w||: {norm2:.4f}")
print("="*50 + "\n")

# --- PLOT GRAPH ---
plt.figure(figsize=(10, 6))
plt.plot(losses1, label='No Regularization ($\lambda=0$)')
plt.plot(losses2, label='L2 Regularization ($\lambda=1$)')
plt.xlabel('Iterations')
plt.ylabel('Loss Value')
plt.title('Question 5: Logistic Regression Training Loss')
plt.legend()
plt.grid(True)

# Save the plot automatically
plt.savefig('training_loss_curve_for_hw4_q5.png')
print("Image saved as 'training_loss_curve_for_hw4_q5.png'. Showing plot now...")
plt.show()