import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. 模型定義 (手動實作 GDA 與 LDA)
# ---------------------------------------------------------

def fit_gda(X, y):
    """
    GDA: 每個類別都有獨立的協方差矩陣 Sigma_c
    """
    n_samples, n_features = X.shape
    classes = np.unique(y)
    priors = {}
    means = {}
    covs = {}
    
    for c in classes:
        X_c = X[y == c]
        priors[c] = len(X_c) / n_samples
        means[c] = np.mean(X_c, axis=0)
        # MLE for Sigma_c
        diff = X_c - means[c]
        covs[c] = (diff.T @ diff) / len(X_c)
        
    return priors, means, covs, classes

def fit_lda(X, y):
    """
    LDA: 所有類別共享同一個協方差矩陣 shared_cov
    """
    n_samples, n_features = X.shape
    classes = np.unique(y)
    priors = {}
    means = {}
    shared_cov = np.zeros((n_features, n_features))
    
    for c in classes:
        X_c = X[y == c]
        priors[c] = len(X_c) / n_samples
        means[c] = np.mean(X_c, axis=0)
        # Accumulate scatter for shared covariance
        diff = X_c - means[c]
        shared_cov += (diff.T @ diff)
        
    shared_cov /= n_samples
    return priors, means, shared_cov, classes

def log_gaussian_pdf(x, mean, cov):
    """
    計算多元高斯分佈的對數概率密度 (來自 Q1.1 的推導)
    """
    d = len(mean)
    inv_cov = np.linalg.inv(cov)
    diff = x - mean
    log_det = np.log(np.linalg.det(cov))
    # 省略 (2*pi)^d/2 項，因為它在類別比較中是常數
    return -0.5 * (log_det + diff.T @ inv_cov @ diff)

def predict(X, priors, means, covs, classes, is_lda=False):
    preds = []
    for x in X:
        scores = []
        for c in classes:
            log_prior = np.log(priors[c])
            # 如果是 LDA，傳入的是單一的 shared_cov；GDA 則是 covs[c]
            sigma = covs if is_lda else covs[c]
            log_lik = log_gaussian_pdf(x, means[c], sigma)
            scores.append(log_prior + log_lik)
        preds.append(classes[np.argmax(scores)])
    return np.array(preds)

# ---------------------------------------------------------
# 2. 數據處理與執行
# ---------------------------------------------------------

if __name__ == "__main__":
    # 讀取 CSV
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    # 提取特徵與標籤
    X_train = train_df[['x1', 'x2']].values
    y_train = train_df['y'].values
    X_test = test_df[['x1', 'x2']].values

    # --- 執行 GDA ---
    p_gda, m_gda, c_gda, classes = fit_gda(X_train, y_train)
    gda_preds = predict(X_test, p_gda, m_gda, c_gda, classes)
    pd.DataFrame({'y': gda_preds}).to_csv('gda_predictions.csv', index=False)
    print("GDA predictions saved.")

    # --- 執行 LDA ---
    p_lda, m_lda, cov_lda, classes = fit_lda(X_train, y_train)
    lda_preds = predict(X_test, p_lda, m_lda, cov_lda, classes, is_lda=True)
    pd.DataFrame({'y': lda_preds}).to_csv('lda_predictions.csv', index=False)
    print("LDA predictions saved.")

    # ---------------------------------------------------------
    # 3. 繪圖 (Decision Boundaries)
    # ---------------------------------------------------------

    def plot_results(X, y_pred, model_type):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        
        if model_type == 'GDA':
            zz = predict(grid, p_gda, m_gda, c_gda, classes)
        else:
            zz = predict(grid, p_lda, m_lda, cov_lda, classes, is_lda=True)
            
        zz = zz.reshape(xx.shape)
        
        plt.figure(figsize=(10, 6))
        plt.contourf(xx, yy, zz, alpha=0.2, cmap='RdYlBu')
        plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], c='blue', label='Class 0', s=10, alpha=0.5)
        plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], c='red', label='Class 1', s=10, alpha=0.5)
        plt.title(f'{model_type} Decision Boundary on Test Data')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.savefig(f'{model_type.lower()}_plot.png')
        print(f"{model_type} plot generated.")

    plot_results(X_test, gda_preds, 'GDA')
    plot_results(X_test, lda_preds, 'LDA')