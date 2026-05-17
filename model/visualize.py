import torch
import cv2
import numpy as np
from PIL import Image

from nca import NCA
from masking import random_mask_batch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "model/models/nca.pth"
TEST_IMG = "model/test.jpeg"

def load_image(path, size=120):
    img = Image.open(path).convert("RGB").resize((size, size))
    img = np.array(img) / 255.0
    img = torch.tensor(img).permute(2,0,1).float()
    return img

model = NCA().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

img = load_image(TEST_IMG).unsqueeze(0).to(DEVICE)

corrupted, mask_t = random_mask_batch(img)

state = NCA.init_state(corrupted, mask_t)

frames = []

with torch.no_grad():
    for i in range(80):
        state = model(state)

        state[:, :3] = state[:, :3] * (1 - mask_t) + corrupted * mask_t

        output = state[:, :3]
        output = torch.clamp(output, 0, 1)

        frame = output[0].permute(1,2,0).cpu().numpy()
        frame = (frame * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame_bgr = cv2.resize(frame_bgr, (256, 256), interpolation=cv2.INTER_NEAREST)
        frames.append(frame_bgr)

h, w, _ = frames[0].shape
video = cv2.VideoWriter("reconstruction.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 15, (w,h))

for f in frames:
    video.write(f)

video.release()
print("The video has been saved as reconstruction.mp4")