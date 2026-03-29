import pandas as pd
import numpy as np

# 1. Load Data
def load_data():
    x_train = pd.read_csv('q5_x_train.csv', header=None).values
    y_train = pd.read_csv('q5_y_train.csv', header=None).values.flatten()
    x_test = pd.read_csv('q5_x_test.csv', header=None).values
    y_test = pd.read_csv('q5_y_test.csv', header=None).values.flatten()
    X_train = np.column_stack([np.ones(x_train.shape[0]), x_train])
    X_test = np.column_stack([np.ones(x_test.shape[0]), x_test])
    return X_train, y_train, X_test, y_test

# 2. Training Function with Numerical Stability
def train(X, y, lam=0, eta=0.01, tol=1e-2):
    m, n = X.shape
    w = np.zeros(n)
    while True:
        z = np.clip(y * (X @ w), -500, 500)
        grad = -(1/m) * ((y / (1 + np.exp(z)))[:, None] * X).sum(axis=0)
        if lam > 0:
            reg_grad = (lam / m) * w
            reg_grad[0] = 0
            grad += reg_grad
        if np.linalg.norm(grad) < tol: break
        w -= eta * grad
    return w

# 3. Main Execution
X_train, y_train, X_test, y_test = load_data()

# Part 1
w1 = train(X_train, y_train, lam=0)
train_acc1 = np.mean(np.where(X_train @ w1 >= 0, 1, -1) == y_train)
test_acc1 = np.mean(np.where(X_test @ w1 >= 0, 1, -1) == y_test)
norm1 = np.linalg.norm(w1[1:])

# Part 2
w2 = train(X_train, y_train, lam=1)
train_acc2 = np.mean(np.where(X_train @ w2 >= 0, 1, -1) == y_train)
test_acc2 = np.mean(np.where(X_test @ w2 >= 0, 1, -1) == y_test)
norm2 = np.linalg.norm(w2[1:])

# FINAL OUTPUT - COPY THESE TO YOUR PDF
print("\n" + "="*40)
print("FINAL RESULTS FOR YOUR PDF")
print("="*40)
print(f"PART 1 (No Regularization):")
print(f" - Training Accuracy: {train_acc1*100:.2f}%")
print(f" - Test Accuracy:     {test_acc1*100:.2f}%")
print(f" - Weight Norm ||w||: {norm1:.4f}")
print("-" * 40)
print(f"PART 2 (L2 Regularization lambda=1):")
print(f" - Training Accuracy: {train_acc2*100:.2f}%")
print(f" - Test Accuracy:     {test_acc2*100:.2f}%")
print(f" - Weight Norm ||w||: {norm2:.4f}")
print("="*40)