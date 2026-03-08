import matplotlib.pyplot as plt
import numpy as np

def plot_shattering_k3():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
    
    # --- 第一部分：6 個點 (N = 2k, k=3) ---
    # 在 2k 個點下，交替標籤產生 k 個正樣本區塊，剛好夠用
    x_6 = np.arange(1, 7)
    labels_6 = np.array([1, -1, 1, -1, 1, -1]) 
    
    ax1.hlines(0, 0.5, 6.5, colors='gray', zorder=1)
    pos_x6 = x_6[labels_6==1]
    neg_x6 = x_6[labels_6==-1]
    
    ax1.scatter(pos_x6, np.zeros_like(pos_x6), color='blue', s=200, label='+1 (Positive)', zorder=3)
    ax1.scatter(neg_x6, np.zeros_like(neg_x6), color='red', s=200, label='-1 (Negative)', zorder=3)
    
    # 繪製 3 個區間 (Intervals) 覆蓋藍色點
    for px in pos_x6:
        ax1.add_patch(plt.Rectangle((px-0.2, -0.05), 0.4, 0.1, color='green', alpha=0.3))
    
    ax1.set_title("Shattering 6 points (N=2k) with k=3 intervals")
    ax1.set_xlim(0, 7); ax1.set_ylim(-0.5, 0.5)
    ax1.get_yaxis().set_visible(False)
    ax1.legend(loc='upper right')

    # --- 第二部分：7 個點 (N = 2k+1, k=3) ---
    # 在 2k+1 個點下，交替標籤產生 k+1 (4) 個區塊，3 個區間不夠用
    x_7 = np.arange(1, 8)
    labels_7 = np.array([1, -1, 1, -1, 1, -1, 1]) 
    
    ax2.hlines(0, 0.5, 7.5, colors='gray', zorder=1)
    pos_x7 = x_7[labels_7==1]
    neg_x7 = x_7[labels_7==-1]
    
    ax2.scatter(pos_x7, np.zeros_like(pos_x7), color='blue', s=200, zorder=3)
    ax2.scatter(neg_x7, np.zeros_like(neg_x7), color='red', s=200, zorder=3)
    
    # 只能畫出 3 個綠色區間，最後一個藍色點會落空
    for px in pos_x7[:3]:
        ax2.add_patch(plt.Rectangle((px-0.2, -0.05), 0.4, 0.1, color='green', alpha=0.3))
        
    ax2.annotate('Missing 4th interval!', xy=(7, 0.05), xytext=(6, 0.3),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, color='darkred')
    
    ax2.set_title("Cannot shatter 7 points (N=2k+1): Needs 4 intervals > k=3")
    ax2.set_xlim(0, 8); ax2.set_ylim(-0.5, 0.5)
    ax2.get_yaxis().set_visible(False)

    plt.tight_layout()
    # 將 plt.show() 替換為以下內容
    save_path = "shattering_result_k3.png"
    plt.savefig(save_path)
    print(f"圖表已成功儲存至: {save_path}")
    # plt.show() # 如果想嘗試彈窗可以保留

if __name__ == "__main__":
    plot_shattering_k3()