import os
from glob import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm

from model.lle_6channel import MobileIELLENetS as MobileIELLENetS6


MODEL_PATH = "./result/model_best.pt"
INPUT_DIR = "./competition/low"
OUTPUT_DIR = "./competition/enhanced_pt"
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileIELLENetS6(channels=32)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    img_paths = []
    for ext in IMAGE_EXTENSIONS:
        img_paths.extend(glob(os.path.join(INPUT_DIR, ext)))
    img_paths = sorted(img_paths)

    print(f"Device: {device}")
    print(f"Found {len(img_paths)} images in {INPUT_DIR}")

    for img_path in tqdm(img_paths, desc="Enhancing"):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: failed to read {img_path}, skipping.")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img_tensor)

        out_img = (out.clamp(0, 1)[0] * 255).permute(1, 2, 0).cpu().numpy()
        out_img = out_img.astype(np.uint8)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

        save_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
        cv2.imwrite(save_path, out_img)

    print(f"Done. Enhanced images saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
