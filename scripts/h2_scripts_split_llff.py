import os
from pathlib import Path

def split_llff(scene_path, hold=8):
    img_dir = Path(scene_path) / "images"
    all_images = sorted(img_dir.glob("*.[jJ][pP][gG]"))  # adjust extension if needed
    test_images = all_images[hold-1::hold]
    train_images = [img for img in all_images if img not in test_images]
    return train_images, test_images

if __name__ == "__main__":
    scenes = ["aeroplane", "bike", "buddha", "cycle", "face", "firehydrant", "still3", "toy"]
    for scene in scenes:
        train, test = split_llff(f"data/{scene}")
        print(f"{scene}: {len(train)} train, {len(test)} test")