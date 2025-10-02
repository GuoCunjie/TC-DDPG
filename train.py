import os, csv, numpy as np, torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    IMG_DIR, ENV_DIR, ACT_DIR, META_MINMAX_PATH, SAVE_DIR,
    BATCH_SIZE, EPOCHS, LR, DEVICE, SEED,
    ACCURACY_TOL, REWARD_BETA,
    REPLAY_SIZE, REPLAY_WARMUP, GAMMA, TAU, CLIP_GRAD,
    W_BC, W_Q, W_PI
)
from models import FeatureExtractor, DDPG
from utils.dataset import TomatoDataset
from utils.metrics import decision_hit_rate
from utils.replay_buffer import ReplayBuffer
from utils.plots import save_pack

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def build_model():
    feat = FeatureExtractor()
    ddpg = DDPG(state_dim=256, action_dim=4)
    class NetAll(torch.nn.Module):
        def __init__(self, f, d): super().__init__(); self.feat=f; self.ddpg=d
        def forward(self, img_b, env_b):
            s = self.feat(img_b, env_b)
            a = self.ddpg.actor(s)
            return s, a
    return NetAll(feat, ddpg)

def train():
    set_seed(SEED)

    dataset = TomatoDataset(IMG_DIR, ENV_DIR, ACT_DIR, minmax_path=META_MINMAX_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = build_model().to(DEVICE)
    optim_feat_actor = optim.Adam(
        list(model.feat.parameters()) + list(model.ddpg.actor.parameters()), lr=LR
    )
    optim_critic = optim.Adam(model.ddpg.critic.parameters(), lr=LR, weight_decay=1e-4)
    huber = torch.nn.SmoothL1Loss()

    # meters
    mse_meter, mae_meter = [], []
    actor_loss_hist, critic_loss_hist = [], []
    acc_hist, reward_hist = [], []

    # save config snapshot
    with open(os.path.join(SAVE_DIR, "config.txt"), "w", encoding="utf-8") as f:
        f.write(
            f"DEVICE={DEVICE}\nEPOCHS={EPOCHS}\nBATCH_SIZE={BATCH_SIZE}\nLR={LR}\n"
            f"ACCURACY_TOL={ACCURACY_TOL}\nREWARD_BETA={REWARD_BETA}\n"
            f"GAMMA={GAMMA}\nTAU={TAU}\nW_BC={W_BC}\nW_Q={W_Q}\nW_PI={W_PI}\n"
        )

    # csv log
    csv_path = os.path.join(SAVE_DIR, "train_log.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        csv.writer(fcsv).writerow(
            ["epoch","mse","mae","hit_rate","avg_reward","actor_loss","critic_loss"]
        )

    # replay buffer
    rb = ReplayBuffer(capacity=REPLAY_SIZE, state_dim=256, action_dim=4, device=DEVICE)

    for epoch in range(1, EPOCHS+1):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
        mse_epoch, mae_epoch, acc_epoch, rew_epoch = [], [], [], []
        act_loss_epoch, cri_loss_epoch = [], []

        for imgs, envs, actions_gt, yield_score in pbar:
            imgs = imgs.to(DEVICE); envs = envs.to(DEVICE); actions_gt = actions_gt.to(DEVICE)

            # forward
            s, a_pred = model(imgs, envs)
            mse = torch.mean((a_pred - actions_gt)**2)
            mae = torch.mean(torch.abs(a_pred - actions_gt))

            # reward + push expert pair
            with torch.no_grad():
                per_mse = torch.mean((a_pred - actions_gt)**2, dim=1)
                reward = torch.clamp(1.0 - REWARD_BETA * per_mse, 0.0, 1.0)
                rb.push(s.detach(), actions_gt.detach(), reward.detach())

            actor_loss = torch.tensor(0.0, device=DEVICE)
            critic_loss = torch.tensor(0.0, device=DEVICE)

            if rb.ptr >= REPLAY_WARMUP:
                ss, aa, rr = rb.sample(BATCH_SIZE)

                # critic
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

                # actor + feature
                aa_actor = model.ddpg.actor(ss)
                q_actor = model.ddpg.critic(ss, aa_actor)
                actor_loss = - q_actor.mean()

                total = W_PI*actor_loss + W_BC*mse
                optim_feat_actor.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.feat.parameters())+list(model.ddpg.actor.parameters()),
                    CLIP_GRAD
                )
                optim_feat_actor.step()

                # target soft update
                model.ddpg.soft_update(TAU)
            else:
                # warmup: supervised only
                optim_feat_actor.zero_grad(set_to_none=True)
                mse.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.feat.parameters())+list(model.ddpg.actor.parameters()),
                    CLIP_GRAD
                )
                optim_feat_actor.step()

            # stats
            hit = decision_hit_rate(a_pred.detach(), actions_gt.detach(), ACCURACY_TOL)
            mse_epoch.append(mse.item()); mae_epoch.append(mae.item())
            acc_epoch.append(hit); rew_epoch.append(reward.mean().item())
            act_loss_epoch.append(actor_loss.item()); cri_loss_epoch.append(critic_loss.item())

            pbar.set_postfix(mse=np.mean(mse_epoch), mae=np.mean(mae_epoch),
                             hit=np.mean(acc_epoch), r=np.mean(rew_epoch))

        # epoch summary
        mse_m = float(np.mean(mse_epoch)); mae_m = float(np.mean(mae_epoch))
        acc_m = float(np.mean(acc_epoch)); rew_m = float(np.mean(rew_epoch))
        act_m = float(np.mean(act_loss_epoch)); cri_m = float(np.mean(cri_loss_epoch))

        mse_meter.append(mse_m); mae_meter.append(mae_m)
        acc_hist.append(acc_m); reward_hist.append(rew_m)
        actor_loss_hist.append(act_m); critic_loss_hist.append(cri_m)

        with open(csv_path, "a", newline="", encoding="utf-8") as fcsv:
            csv.writer(fcsv).writerow(
                [epoch, f"{mse_m:.6f}", f"{mae_m:.6f}", f"{acc_m:.4f}", f"{rew_m:.4f}", f"{act_m:.6f}", f"{cri_m:.6f}"]
            )

    # save plots
    save_pack(mse_meter, mae_meter, acc_hist, reward_hist, actor_loss_hist, critic_loss_hist,
              SAVE_DIR, ACCURACY_TOL)

    # save checkpoint
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
