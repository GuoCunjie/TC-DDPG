import os, json, csv
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# 可调参数
IMG_DIR = "./data/imgs"
ENV_DIR = "./data/env"
ACT_DIR = "./data/actions"
META_MINMAX_PATH = "./data/meta/env_minmax.json"

BATCH_SIZE = 8
EPOCHS = 200
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "./results"

ACCURACY_TOL = 0.30          # 命中率容差（±30%）

# DDPG / 回放
REPLAY_SIZE = 50000
REPLAY_WARMUP = 1500         # 热身步数：先只做监督MSE
GAMMA = 0.9                  # TD 目标系数
TAU = 0.01                   # 目标网络软更新
W_BC, W_Q, W_PI = 1.0, 0.5, 0.2   # 监督/critic/actor 的损失权重（注意：W_Q只用于记录，不反传）
CLIP_GRAD = 1.0              # 梯度裁剪
REWARD_BETA = 0.02           # 奖励缩放：reward = 1 - beta * mse

SEED = 2025


torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(SAVE_DIR, exist_ok=True)

# 数据工具
def minmax_scale_scalar(x: float, lo: float, hi: float, eps: float = 1e-6) -> float:
    return float((x - lo) / max(hi - lo, eps))

# 数据集
class TomatoDataset(Dataset):
    def __init__(self, img_dir, env_dir, action_dir, transform=None, minmax_path=META_MINMAX_PATH):
        self.img_dir = img_dir
        self.env_dir = env_dir
        self.action_dir = action_dir
        self.img_files = sorted(os.listdir(img_dir))
        self.env_files = sorted(os.listdir(env_dir))
        self.action_files = sorted(os.listdir(action_dir))
        assert len(self.img_files)==len(self.env_files)==len(self.action_files), "数量不一致"

        if os.path.exists(minmax_path):
            with open(minmax_path, "r", encoding="utf-8") as f:
                self.minmax = json.load(f)
        else:
            # 没有统计文件也能跑
            self.minmax = {
                "Temperature": [10.0, 40.0],
                "Humidity": [30.0, 95.0],
                "LightIntensity": [0.0, 1200.0],
                "WindForce": [0.0, 6.0],
                "Precipitation": [0.0, 3.0]
            }

        self.transform = transform or transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        env_path = os.path.join(self.env_dir, self.env_files[idx])
        act_path = os.path.join(self.action_dir, self.action_files[idx])

        img = Image.open(img_path).convert("RGB")
        with open(env_path, "r", encoding="utf-8") as f:
            env = json.load(f)
        with open(act_path, "r", encoding="utf-8") as f:
            action = json.load(f)

        # 环境原始数值
        env_raw = {
            "Temperature": float(env["Temperature"]),
            "Humidity": float(env["Humidity"]),
            "LightIntensity": float(env["LightIntensity"]),
            "WindForce": float(env.get("WindForce", 0.0)),
            "Precipitation": float(env.get("Precipitation", 0.0)),
        }
        # 缩放到 [0,1]
        env_scaled = []
        for k in ["Temperature","Humidity","LightIntensity","WindForce","Precipitation"]:
            lo, hi = self.minmax[k][0], self.minmax[k][1]
            env_scaled.append(minmax_scale_scalar(env_raw[k], lo, hi))
        env_values = torch.tensor(env_scaled, dtype=torch.float32)

        # 专家动作（真实/规则）
        action_values = torch.tensor([
            float(action["irrigation"]),
            float(action["fertilizer"]),
            float(action["ventilation"]),
            float(action["light_supplement"])
        ], dtype=torch.float32)

        yield_score = float(action.get("yield_score", -1.0))

        img = self.transform(img)
        return img, env_values, action_values, yield_score

# 特征提取：CNN + Transformer(Encoder)
class EnvEncoder(nn.Module):
    def __init__(self, in_dim=5, d_model=128, nhead=4, nlayers=2):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

    def forward(self, env_b):   # [B,5]
        x = self.proj(env_b).unsqueeze(1)   # [B,1,d_model]
        x = self.enc(x)
        return x.squeeze(1)                 # [B,d_model]

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4,4))
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 256), nn.ReLU(inplace=True)
        )

    def forward(self, img_b):   # [B,3,H,W]
        x = self.body(img_b)
        x = self.head(x)        # [B,256]
        return x

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_enc = ImageEncoder()
        self.env_enc = EnvEncoder(in_dim=5, d_model=128, nhead=4, nlayers=2)
        self.fuse = nn.Sequential(
            nn.Linear(256 + 128, 256), nn.ReLU(inplace=True)
        )

    def forward(self, img_b, env_b):
        vi = self.img_enc(img_b)                 # [B,256]
        ve = self.env_enc(env_b)                 # [B,128]
        z = self.fuse(torch.cat([vi, ve], dim=1))# [B,256]
        return z

# DDPG 模块
class Actor(nn.Module):
    def __init__(self, state_dim=256, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, action_dim)
        )
        # 物理边界（和 data_create.py 中一致）
        self.bounds = {
            "irrigation": (0.5, 8.0),
            "fertilizer": (2.0, 20.0),
            "ventilation": (0.0, 24.0),
            "light_supplement": (0.0, 200.0)
        }

    def forward(self, s):
        x = self.net(s)  # logits
        # tanh 压到 [-1,1] 再线性映射到边界
        b = self.bounds
        a1 = (torch.tanh(x[:,0:1])+1)/2 * (b["irrigation"][1]-b["irrigation"][0]) + b["irrigation"][0]
        a2 = (torch.tanh(x[:,1:2])+1)/2 * (b["fertilizer"][1]-b["fertilizer"][0]) + b["fertilizer"][0]
        a3 = (torch.tanh(x[:,2:3])+1)/2 * (b["ventilation"][1]-b["ventilation"][0]) + b["ventilation"][0]
        a4 = (torch.tanh(x[:,3:4])+1)/2 * (b["light_supplement"][1]-b["light_supplement"][0]) + b["light_supplement"][0]
        return torch.cat([a1,a2,a3,a4], dim=1)

class Critic(nn.Module):
    def __init__(self, state_dim=256, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        return self.net(x)

class DDPG(nn.Module):
    def __init__(self, state_dim=256, action_dim=4):
        super().__init__()
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.actor_targ = Actor(state_dim, action_dim)
        self.critic_targ = Critic(state_dim, action_dim)
        self._hard_update(self.actor_targ, self.actor)
        self._hard_update(self.critic_targ, self.critic)

    @staticmethod
    def _hard_update(target, source):
        target.load_state_dict(source.state_dict())

    @torch.no_grad()
    def soft_update(self, tau=TAU):
        for t, s in zip(self.actor_targ.parameters(), self.actor.parameters()):
            t.data.mul_(1-tau).add_(tau*s.data)
        for t, s in zip(self.critic_targ.parameters(), self.critic.parameters()):
            t.data.mul_(1-tau).add_(tau*s.data)

# 回放池
class ReplayBuffer:
    def __init__(self, capacity=REPLAY_SIZE, state_dim=256, action_dim=4, device=DEVICE):
        self.capacity = capacity
        self.device = device
        self.ptr = 0; self.full = False
        self.s = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.a = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.r = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

    def push(self, s, a, r):
        n = s.shape[0]
        for i in range(n):
            self.s[self.ptr] = s[i]
            self.a[self.ptr] = a[i]
            self.r[self.ptr,0] = r[i]
            self.ptr = (self.ptr + 1) % self.capacity
            if self.ptr == 0: self.full = True

    def sample(self, batch):
        size = self.capacity if self.full else self.ptr
        idx = np.random.randint(0, size, size=batch)
        s = self.s[idx]; a = self.a[idx]; r = self.r[idx]
        return s, a, r

# 总模型
class NetAll(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat = FeatureExtractor()
        self.ddpg = DDPG(state_dim=256, action_dim=4)

    def forward(self, img_b, env_b):
        s = self.feat(img_b, env_b)          # [B,256]
        a = self.ddpg.actor(s)                # [B,4]
        return s, a

# 指标
def decision_hit_rate(pred, target, tol=ACCURACY_TOL):
    # 连续动作命中率：四维都满足 |pred-target| <= tol*|target|
    eps = 1e-6
    rel_ok = (torch.abs(pred - target) <= tol * (torch.abs(target)+eps))
    hit = torch.all(rel_ok, dim=1).float().mean().item()
    return hit

# 训练主流程
def main():
    # 保存配置
    with open(os.path.join(SAVE_DIR, "config.txt"), "w", encoding="utf-8") as f:
        f.write(
            f"DEVICE={DEVICE}\nEPOCHS={EPOCHS}\nBATCH_SIZE={BATCH_SIZE}\nLR={LR}\n"
            f"ACCURACY_TOL={ACCURACY_TOL}\nREWARD_BETA={REWARD_BETA}\n"
            f"GAMMA={GAMMA}\nTAU={TAU}\nW_BC={W_BC}\nW_Q={W_Q}\nW_PI={W_PI}\n"
        )

    dataset = TomatoDataset(IMG_DIR, ENV_DIR, ACT_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = NetAll().to(DEVICE)

    # 优化器
    optim_feat_actor = optim.Adam(
        list(model.feat.parameters()) + list(model.ddpg.actor.parameters()), lr=LR
    )
    optim_critic = optim.Adam(model.ddpg.critic.parameters(), lr=LR, weight_decay=1e-4)

    huber = torch.nn.SmoothL1Loss()

    mse_meter, mae_meter = [], []
    actor_loss_hist, critic_loss_hist = [], []
    acc_hist, reward_hist = [], []

    # 日志 CSV
    csv_path = os.path.join(SAVE_DIR, "train_log.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["epoch","mse","mae","hit_rate","avg_reward","actor_loss","critic_loss"])

    # 回放池
    rb = ReplayBuffer(device=DEVICE)

    for epoch in range(1, EPOCHS+1):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
        mse_epoch, mae_epoch, acc_epoch, rew_epoch = [], [], [], []
        act_loss_epoch, cri_loss_epoch = [], []

        for imgs, envs, actions_gt, yield_score in pbar:
            imgs = imgs.to(DEVICE)
            envs = envs.to(DEVICE)
            actions_gt = actions_gt.to(DEVICE)

            # 前向
            s, a_pred = model(imgs, envs)

            # 监督误差
            mse = torch.mean((a_pred - actions_gt)**2)
            mae = torch.mean(torch.abs(a_pred - actions_gt))

            # 奖励（0..1）：1 - beta * per-sample MSE
            with torch.no_grad():
                per_mse = torch.mean((a_pred - actions_gt)**2, dim=1)   # [B]
                reward = 1.0 - REWARD_BETA * per_mse
                reward = torch.clamp(reward, 0.0, 1.0)
                # 用专家动作当离线目标存池
                rb.push(s.detach(), actions_gt.detach(), reward.detach())

            actor_loss = torch.tensor(0.0, device=DEVICE)
            critic_loss = torch.tensor(0.0, device=DEVICE)

            if rb.ptr >= REPLAY_WARMUP:
                ss, aa, rr = rb.sample(BATCH_SIZE)

                # Critic 更新（有梯度）
                # TD 目标：y = r + γ * Q_targ(s, π_targ(s))
                with torch.no_grad():
                    aa_t = model.ddpg.actor_targ(ss)
                    q_next = model.ddpg.critic_targ(ss, aa_t)
                    y = torch.clamp(rr + GAMMA * q_next, 0.0, 1.0)

                q_pred = model.ddpg.critic(ss, aa)
                critic_loss = huber(q_pred, y)

                optim_critic.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.ddpg.critic.parameters(), CLIP_GRAD)
                optim_critic.step()

                # Actor + 特征 更新
                aa_actor = model.ddpg.actor(ss)
                q_actor = model.ddpg.critic(ss, aa_actor)
                actor_loss = - q_actor.mean()

                total = W_PI*actor_loss + W_BC*mse
                optim_feat_actor.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(list(model.feat.parameters())+list(model.ddpg.actor.parameters()), CLIP_GRAD)
                optim_feat_actor.step()

                # 软更新目标网络
                model.ddpg.soft_update(TAU)
            else:
                # 热身：只做监督
                optim_feat_actor.zero_grad(set_to_none=True)
                mse.backward()
                torch.nn.utils.clip_grad_norm_(list(model.feat.parameters())+list(model.ddpg.actor.parameters()), CLIP_GRAD)
                optim_feat_actor.step()

            # 统计
            hit_rate = decision_hit_rate(a_pred.detach(), actions_gt.detach(), tol=ACCURACY_TOL)
            mse_epoch.append(mse.item())
            mae_epoch.append(mae.item())
            acc_epoch.append(hit_rate)
            rew_epoch.append(reward.mean().item())
            act_loss_epoch.append(actor_loss.item())
            cri_loss_epoch.append(critic_loss.item())

            pbar.set_postfix(mse=np.mean(mse_epoch), mae=np.mean(mae_epoch),
                             hit=np.mean(acc_epoch), r=np.mean(rew_epoch))

        # 每个 epoch 汇总
        mse_epoch = float(np.mean(mse_epoch))
        mae_epoch = float(np.mean(mae_epoch))
        acc_epoch = float(np.mean(acc_epoch))
        rew_epoch = float(np.mean(rew_epoch))
        actor_l = float(np.mean(act_loss_epoch))
        critic_l = float(np.mean(cri_loss_epoch))

        mse_meter.append(mse_epoch)
        mae_meter.append(mae_epoch)
        acc_hist.append(acc_epoch)
        reward_hist.append(rew_epoch)
        actor_loss_hist.append(actor_l)
        critic_loss_hist.append(critic_l)

        # 记录到 CSV
        with open(csv_path, "a", newline="", encoding="utf-8") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([epoch, f"{mse_epoch:.6f}", f"{mae_epoch:.6f}",
                             f"{acc_epoch:.4f}", f"{rew_epoch:.4f}",
                             f"{actor_l:.6f}", f"{critic_l:.6f}"])

    # 可视化并保存
    def save_curve(values, title, ylabel, fname):
        plt.figure()
        plt.plot(values)
        plt.title(title); plt.xlabel("Epoch"); plt.ylabel(ylabel)
        plt.tight_layout()
        path = os.path.join(SAVE_DIR, fname)
        plt.savefig(path); plt.close()
        print(f"Saved: {path}")

    save_curve(mse_meter, "MSE (actor vs. expert action)", "MSE", "curve_mse.png")
    save_curve(mae_meter, "MAE (actor vs. expert action)", "MAE", "curve_mae.png")
    save_curve(acc_hist, f"Decision Hit-Rate (±{int(ACCURACY_TOL*100)}%)", "Hit-Rate", "curve_hit.png")
    save_curve(reward_hist, "Avg Reward", "Reward", "curve_reward.png")
    save_curve(actor_loss_hist, "Actor Loss", "Loss", "curve_actor.png")
    save_curve(critic_loss_hist, "Critic Loss", "Loss", "curve_critic.png")

    # 合并图
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(mse_meter); plt.title("Training Loss (MSE)"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.subplot(1,2,2); plt.plot(acc_hist); plt.title(f"Training Accuracy (Hit ±{int(ACCURACY_TOL*100)}%)"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "train_curves.png")); plt.close()

    # 保存模型
    torch.save({
        "feature_extractor": model.feat.state_dict(),
        "actor": model.ddpg.actor.state_dict(),
        "critic": model.ddpg.critic.state_dict(),
        "actor_target": model.ddpg.actor_targ.state_dict(),
        "critic_target": model.ddpg.critic_targ.state_dict(),
        "config": {
            "ACCURACY_TOL": ACCURACY_TOL,
            "REWARD_BETA": REWARD_BETA,
            "GAMMA": GAMMA, "TAU": TAU, "LR": LR,
            "W_BC": W_BC, "W_Q": W_Q, "W_PI": W_PI
        }
    }, os.path.join(SAVE_DIR, "checkpoint.pth"))

    print("\n✅ 训练完成，结果已保存到:", os.path.abspath(SAVE_DIR))
    print("  - 模型: checkpoint.pth")
    print("  - 日志: train_log.csv")
    print("  - 曲线: curve_*.png, train_curves.png")
    print("  - 配置: config.txt")

if __name__ == "__main__":
    main()
