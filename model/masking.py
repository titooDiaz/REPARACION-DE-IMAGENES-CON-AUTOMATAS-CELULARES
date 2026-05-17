import torch
import random

def random_mask_batch(imgs):
    B, C, H, W = imgs.shape
    masks = torch.ones((B, 1, H, W), device=imgs.device)

    for i in range(B):
        for _ in range(random.randint(2, 5)):
            box_w = random.randint(10, 30)
            box_h = random.randint(10, 30)

            x1 = random.randint(0, max(1, W - box_w))
            y1 = random.randint(0, max(1, H - box_h))

            masks[i, 0, y1:y1+box_h, x1:x1+box_w] = 0

    corrupted = imgs.clone()
    corrupted = corrupted * masks

    return corrupted, masks