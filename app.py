import io
import base64
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import torch
import numpy as np

from model.nca import NCA

app = Flask(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "REPARACION-DE-IMAGENES-CON-AUTOMATAS-CELULARES-main/model/nca.pth"
IMG_SIZE = 120

model = NCA().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def preprocess_image(pil_img):
    pil_img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img = np.array(pil_img) / 255.0
    img = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
    return img

def postprocess_tensor(tensor):
    img = tensor.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    img = (img * 255).clip(0,255).astype(np.uint8)
    return Image.fromarray(img)

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/reconstruct", methods=["POST"])
def reconstruct():
    data = request.json
    img_data = data["image"]

    header, encoded = img_data.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    pil_img = Image.open(io.BytesIO(img_bytes))

    img = preprocess_image(pil_img)

    corrupted = img.clone()
    mask = data["mask"]
    mask = np.array(mask).astype(np.float32)
    mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0).to(DEVICE)

    corrupted = corrupted * mask

    state = NCA.init_state(corrupted, mask)

    with torch.no_grad():
        for _ in range(60):
            state = model(state)
            rgb = state[:, :3]
            other = state[:, 3:]
            rgb = rgb * (1 - mask) + corrupted * mask
            state = torch.cat([rgb, other], dim=1)

        output = torch.clamp(state[:, :3], 0, 1)

    out_img = postprocess_tensor(output)

    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    result_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return jsonify({"image": "data:image/png;base64," + result_base64})

if __name__ == "__main__":
    app.run(debug=True)