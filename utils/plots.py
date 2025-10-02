import os
import matplotlib.pyplot as plt

def save_curve(values, title, ylabel, save_dir, fname):
    plt.figure()
    plt.plot(values)
    plt.title(title); plt.xlabel("Epoch"); plt.ylabel(ylabel)
    plt.tight_layout()
    path = os.path.join(save_dir, fname)
    plt.savefig(path); plt.close()
    print(f"Saved: {path}")

def save_pack(mse, mae, acc, rew, act, cri, save_dir, tol_pct):
    save_curve(mse, "MSE (actor vs. expert action)", "MSE", save_dir, "curve_mse.png")
    save_curve(mae, "MAE (actor vs. expert action)", "MAE", save_dir, "curve_mae.png")
    save_curve(acc, f"Decision Hit-Rate (±{int(tol_pct*100)}%)", "Hit-Rate", save_dir, "curve_hit.png")
    save_curve(rew, "Avg Reward", "Reward", save_dir, "curve_reward.png")
    save_curve(act, "Actor Loss", "Loss", save_dir, "curve_actor.png")
    save_curve(cri, "Critic Loss", "Loss", save_dir, "curve_critic.png")

    # 合并两张常用图
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(mse); plt.title("Training Loss (MSE)"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.subplot(1,2,2); plt.plot(acc); plt.title(f"Training Accuracy (Hit ±{int(tol_pct*100)}%)"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "train_curves.png")); plt.close()
