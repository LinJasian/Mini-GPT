import matplotlib.pyplot as plt
import os

# Define results directory
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'results'))
os.makedirs(RESULTS_DIR, exist_ok=True)

def log_info(text, log_file="training_log.txt"):
    """将文本追加记录到日志文件中"""
    log_path = os.path.join(RESULTS_DIR, log_file)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(text + "\n")
    print(text)

def plot_loss(train_losses, val_losses, eval_steps):
    # ================= 中文字符异常显示解决 =================
    # 优先使用 Mac 的黑体(Heiti TC)和微软雅圆/黑体，最后用 sans-serif 兜底
    plt.rcParams['font.sans-serif'] = ['Heiti TC', 'SimHei', 'Arial Unicode MS', 'sans-serif']
    # 顺便修复负号（减号）在某些情况下显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False
    # ================= 作图 =================
    plt.figure(figsize=(10, 6))
    plt.plot(eval_steps, train_losses, label="Train Loss", marker='o')
    plt.plot(eval_steps, val_losses, label="Validation Loss", marker='x')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")

    plt.title("Training and Validation Loss 收敛曲线")
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(RESULTS_DIR, "loss_curve.png")
    plt.savefig(save_path, dpi=300)
    log_info(f"Loss 收敛曲线已保存至 {save_path}")
