# minnen20_pipeline.py

import os
import sys
import csv

# ---------------------- ENV CHECK ----------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    from torchvision.utils import save_image
    import torch.nn.functional as F
    from PIL import Image
except ModuleNotFoundError:
    print("[ERROR] This script requires PyTorch and torchvision.")
    print("Please install them by running: pip install torch torchvision")
    sys.exit(1)

from minnen20 import Minnen2020LRP

# ---------------------- CONFIG ----------------------
CONFIG = {
    'epochs': 75,
    'batch_size': 4,
    'learning_rate': 1e-4,
    'image_size': (256, 256),
    'data_path': './data',
    'save_path': './checkpoints',
    'model_name': 'minnen20lrp.pth',
    'eval_output_dir': './eval_results',
    'eval_csv': './eval_results/metrics.csv'
}

# ---------------------- TRANSFORMS ----------------------
transform = transforms.Compose([
    transforms.Resize(CONFIG['image_size']),
    transforms.CenterCrop(CONFIG['image_size']),
    transforms.ToTensor(),
])

# ---------------------- DATASET ----------------------
if not os.path.exists(CONFIG['data_path']):
    print(f"[ERROR] Dataset path {CONFIG['data_path']} does not exist.")
    sys.exit(1)

dataset = ImageFolder(root=CONFIG['data_path'], transform=transform)
dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)

# ---------------------- MODEL ----------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Minnen2020LRP(N=192, M=320).to(device)
model.train()

# ---------------------- LOSS ----------------------
class RateDistortionLoss(nn.Module):
    def __init__(self, lmbda=0.01):
        super().__init__()
        self.lmbda = lmbda
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        num_pixels = target.size(2) * target.size(3)
        bpp_y = torch.sum(torch.log(output['likelihoods']['y'] + 1e-9)) / (-torch.log(torch.tensor(2.0)) * num_pixels)
        bpp_z = torch.sum(torch.log(output['likelihoods']['z'] + 1e-9)) / (-torch.log(torch.tensor(2.0)) * num_pixels)
        bpp = bpp_y + bpp_z
        mse_loss = self.mse(output['x_hat'], target)
        loss = self.lmbda * 255**2 * mse_loss + bpp
        return loss, bpp, mse_loss

criterion = RateDistortionLoss(lmbda=0.01)
optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

# ---------------------- PSNR & MS-SSIM ----------------------
def compute_psnr(a, b):
    mse = F.mse_loss(a, b)
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.item()

try:
    from pytorch_msssim import ms_ssim
    def compute_msssim(a, b):
        return ms_ssim(a, b, data_range=1.0).item()
except ImportError:
    print("[WARN] pytorch-msssim not installed. MS-SSIM will be skipped.")
    def compute_msssim(a, b):
        return -1.0

# ---------------------- TRAIN ----------------------
print("[INFO] Starting training...")
os.makedirs(CONFIG['save_path'], exist_ok=True)
for epoch in range(CONFIG['epochs']):
    total_loss, total_bpp, total_mse = 0, 0, 0
    for i, (x, _) in enumerate(dataloader):
        x = x.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss, bpp, mse = criterion(output, x)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_bpp += bpp.item()
        total_mse += mse.item()

        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{CONFIG['epochs']}], Step [{i}], Loss: {loss.item():.4f}, BPP: {bpp.item():.4f}, MSE: {mse.item():.4f}")

    torch.save(model.state_dict(), os.path.join(CONFIG['save_path'], CONFIG['model_name']))
    print(f"[INFO] Epoch {epoch+1} completed. Avg Loss: {total_loss/len(dataloader):.4f}")

print("[INFO] Training complete.")
model.update()


# ---------------------- SINGLE TEST ----------------------
def test(model, image_path, out_path):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out_enc = model.compress(x)
        out_dec = model.decompress(out_enc['strings'], out_enc['shape'])
        x_hat = out_dec['x_hat'].clamp(0, 1)
        psnr = compute_psnr(x_hat, x)
        msssim = compute_msssim(x_hat, x)
        save_image(x_hat, out_path)
        print(f"[INFO] Image reconstructed and saved to {out_path}")
        print(f"[EVAL] PSNR: {psnr:.2f} dB, MS-SSIM: {msssim:.4f}")

# ---------------------- BATCH TEST ----------------------
def batch_test_all_images(model, input_dir, output_dir, csv_path):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'PSNR (dB)', 'MS-SSIM'])

        for fname in sorted(os.listdir(input_dir)):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, f"recon_{fname}")
            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out_enc = model.compress(x)
                out_dec = model.decompress(out_enc['strings'], out_enc['shape'])
                x_hat = out_dec['x_hat'].clamp(0, 1)
                psnr = compute_psnr(x_hat, x)
                msssim = compute_msssim(x_hat, x)
                save_image(x_hat, out_path)
                writer.writerow([fname, f"{psnr:.2f}", f"{msssim:.4f}"])
                print(f"[EVAL] {fname}: PSNR={psnr:.2f} dB, MS-SSIM={msssim:.4f}")

# Example single test:
# test(model, "./data/kodak/images/class1/kodim01.png", "./recon_kodim01.png")

# Example batch test:
batch_test_all_images(model, "./data/kodak/images/class1", CONFIG['eval_output_dir'], CONFIG['eval_csv'])
