import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.transforms import Normalize

class TotalLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16].eval().to(device)
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def perceptual_loss(self, output, target):
        f1 = self.vgg(self.norm(output))
        f2 = self.vgg(self.norm(target))
        return F.mse_loss(f1, f2)

    def forward(self, output, target, mask):
        hole_mask = 1.0 - mask
        
        mse = (F.mse_loss(output, target, reduction='none') * hole_mask).sum() / (hole_mask.sum() * 3 + 1e-8)

        perc = self.perceptual_loss(output, target)

        tv_loss = torch.mean(torch.abs(output[:, :, :, :-1] - output[:, :, :, 1:])) + \
                  torch.mean(torch.abs(output[:, :, :-1, :] - output[:, :, 1:, :]))

        return mse + 0.05 * perc + 0.01 * tv_loss