import cv2
import json
from pathlib import Path

SUBMISSION_ROOT = Path("sub")
with open("target_sizes.json", "r") as f:
    target_sizes = json.load(f)

for scene, (target_w, target_h) in target_sizes.items():
    scene_dir = SUBMISSION_ROOT / scene
    if not scene_dir.exists():
        continue
    for img_path in scene_dir.glob("*.*"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        # Resize to exact target dimensions
        resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(img_path), resized)
        print(f"Resized {scene}/{img_path.name} to {target_w}x{target_h}")

print("All images resized to 4x original dimensions.")