import torch
import torch.nn as nn
import torch.nn.functional as F

class NCA(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.channels = channels

        self.conv1 = nn.Conv2d(channels, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 1)
        self.conv3 = nn.Conv2d(128, channels, 1)

        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x):
        dx = F.relu(self.conv1(x))
        dx = F.relu(self.conv2(dx))
        dx = self.conv3(dx)

        B, C, H, W = x.shape
        update_mask = (torch.rand((B, 1, H, W), device=x.device) < 0.5).float()

        new_x = x + dx * update_mask

        return new_x

    @staticmethod
    def init_state(img, mask, channels=32):
        B, C, H, W = img.shape
        state = torch.zeros((B, channels, H, W), device=img.device)
        state[:, :3] = img
        state[:, 3:4] = mask
        return state