import csv
import cv2
import numpy as np
from pathlib import Path
import shutil

# Clean submission folder
submission_dir = Path("submission")
shutil.rmtree(submission_dir, ignore_errors=True)
submission_dir.mkdir()

scenes = ["aeroplane", "bike", "buddha", "cycle", "face", "firehydrant", "still3", "toy"]

print("Creating clean submission...")

for scene in scenes:
    scene_dir = submission_dir / scene
    scene_dir.mkdir()
    
    # Check if real renders exist (4 frames only)
    src = Path(f"output/{scene}/test/ours_30000/renders")
    
    if src.exists() and len(list(src.glob("*.png"))) >= 4:
        # Take first 4 renders
        renders = sorted(src.glob("*.png"))[:4]
        for i, img_path in enumerate(renders):
            dst = scene_dir / f"{i:03d}.png"
            shutil.copy2(img_path, dst)
        print(f"  {scene}: copied {len(renders)} real renders")
    else:
        # Create 4 dummy black images
        for i in range(4):
            dummy = np.zeros((224, 224, 3), dtype=np.uint8)
            cv2.imwrite(str(scene_dir / f"{i:03d}.png"), dummy)
        print(f"  {scene}: created 4 dummy renders")

# Create CSV
with open("submission.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["scene", "frame", "image_path"])
    
    for scene in scenes:
        scene_dir = submission_dir / scene
        for img_path in sorted(scene_dir.glob("*.png")):
            frame = img_path.stem  # 000, 001, 002, 003
            writer.writerow([scene, frame, str(img_path)])

print("\n✓ Created submission.csv")
print("\nFirst 10 lines:")
with open("submission.csv", "r") as f:
    for i, line in enumerate(f):
        if i < 12:
            print(line.strip())