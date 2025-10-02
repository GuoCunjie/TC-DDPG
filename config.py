import os
import torch

# 数据与输出路径
IMG_DIR = "./data/imgs"
ENV_DIR = "./data/env"
ACT_DIR = "./data/actions"
META_MINMAX_PATH = "./data/meta/env_minmax.json"

SAVE_DIR = "./results/run2"
os.makedirs(SAVE_DIR, exist_ok=True)

# 训练超参
BATCH_SIZE = 8
EPOCHS = 200
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2025

# 命中率判定（相对误差容差）
ACCURACY_TOL = 0.30

# DDPG/回放
REPLAY_SIZE = 50_000
REPLAY_WARMUP = 1500
GAMMA = 0.9
TAU = 0.01
CLIP_GRAD = 1.0

# 损失权重
W_BC = 1.0   # 行为克隆（监督MSE）
W_Q  = 0.5   # 仅记录，不反传
W_PI = 0.2   # 策略（-Q）

# 奖励缩放
REWARD_BETA = 0.02  # reward = clamp(1 - beta*mse, 0, 1)
