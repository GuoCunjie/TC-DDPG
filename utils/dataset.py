import os, json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

DEFAULT_MINMAX = {
    "Temperature": [10.0, 40.0],
    "Humidity": [30.0, 95.0],
    "LightIntensity": [0.0, 1200.0],
    "WindForce": [0.0, 6.0],
    "Precipitation": [0.0, 3.0]
}

def minmax_scale_scalar(x: float, lo: float, hi: float, eps: float = 1e-6) -> float:
    return float((x - lo) / max(hi - lo, eps))

class TomatoDataset(Dataset):
    def __init__(self, img_dir, env_dir, action_dir, transform=None, minmax_path=None):
        self.img_dir = img_dir
        self.env_dir = env_dir
        self.action_dir = action_dir
        self.img_files = sorted(os.listdir(img_dir))
        self.env_files = sorted(os.listdir(env_dir))
        self.action_files = sorted(os.listdir(action_dir))
        assert len(self.img_files)==len(self.env_files)==len(self.action_files), "样本数不一致"

        if minmax_path and os.path.exists(minmax_path):
            with open(minmax_path, "r", encoding="utf-8") as f:
                self.minmax = json.load(f)
        else:
            self.minmax = DEFAULT_MINMAX

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

        env_raw = {
            "Temperature": float(env["Temperature"]),
            "Humidity": float(env["Humidity"]),
            "LightIntensity": float(env["LightIntensity"]),
            "WindForce": float(env.get("WindForce", 0.0)),
            "Precipitation": float(env.get("Precipitation", 0.0)),
        }
        env_scaled = []
        for k in ["Temperature","Humidity","LightIntensity","WindForce","Precipitation"]:
            lo, hi = self.minmax[k][0], self.minmax[k][1]
            env_scaled.append(minmax_scale_scalar(env_raw[k], lo, hi))
        env_values = torch.tensor(env_scaled, dtype=torch.float32)

        action_values = torch.tensor([
            float(action["irrigation"]),
            float(action["fertilizer"]),
            float(action["ventilation"]),
            float(action["light_supplement"])
        ], dtype=torch.float32)

        yield_score = float(action.get("yield_score", -1.0))  # 未用到，但保留
        img = self.transform(img)
        return img, env_values, action_values, yield_score
