import torch
import torch.nn as nn

class EnvEncoder(nn.Module):
    def __init__(self, in_dim=5, d_model=128, nhead=4, nlayers=2):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)

    def forward(self, env_b):         # [B,5]
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

    def forward(self, img_b):         # [B,3,H,W]
        x = self.body(img_b)
        x = self.head(x)              # [B,256]
        return x

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_enc = ImageEncoder()
        self.env_enc = EnvEncoder()
        self.fuse = nn.Sequential(
            nn.Linear(256 + 128, 256), nn.ReLU(inplace=True)
        )

    def forward(self, img_b, env_b):
        vi = self.img_enc(img_b)                 # [B,256]
        ve = self.env_enc(env_b)                 # [B,128]
        z = self.fuse(torch.cat([vi, ve], dim=1))# [B,256]
        return z
