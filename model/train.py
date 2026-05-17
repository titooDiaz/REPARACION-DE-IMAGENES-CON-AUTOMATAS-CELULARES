import torch
import os
import random
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from nca import NCA
from masking import random_mask_batch
from losses import TotalLoss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = "data/img_align_celeba"
SAVE_PATH = "models/nca.pth"
BATCH_SIZE = 8
IMG_SIZE = 120
NUM_WORKERS = 2

class CelebADataset(Dataset):
    def __init__(self, folder_path, size):
        self.folder_path = folder_path
        self.size = size
        self.files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.files[idx])
        img = Image.open(img_path).convert("RGB").resize((self.size, self.size))
        img = np.array(img) / 255.0
        return torch.tensor(img).permute(2, 0, 1).float()

def train():
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    dataset = CelebADataset(DATASET_PATH, size=IMG_SIZE)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    torch.backends.cudnn.benchmark = True

    model = NCA().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    loss_fn = TotalLoss(DEVICE)
    scaler = GradScaler()

    epochs = 200
    global_step = 0

    print(f"starting training on {DEVICE}...")

    for epoch in range(epochs):
        for batch_imgs in dataloader:
            batch_imgs = batch_imgs.to(DEVICE, non_blocking=True)

            corrupted, mask_t = random_mask_batch(batch_imgs)
            state = NCA.init_state(corrupted, mask_t)

            steps = random.randint(48, 96)

            with autocast():
                for _ in range(steps):
                    state = model(state)

                    rgb = state[:, :3]
                    other = state[:, 3:]
                    rgb = rgb * (1 - mask_t) + corrupted * mask_t
                    state = torch.cat([rgb, other], dim=1)

                output = torch.clamp(state[:, :3], 0, 1)

                loss = loss_fn(output, batch_imgs, mask_t)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            if global_step % 20 == 0:
                print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item():.4f}")

            global_step += 1

        scheduler.step()

        torch.save(model.state_dict(), SAVE_PATH)
        print(f"model save - {epoch}")

if __name__ == '__main__':
    train()