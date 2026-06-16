"""
LLFF-style train/test split utility.
Every llffhold-th image (0-indexed) is held out as test.
llffhold=8 means indices 0,8,16,24,... are test views.
"""

import os
import re
from pathlib import Path


SCENES = ["aeroplane", "bike", "buddha", "cycle", "face", "firehydrant", "still3", "toy"]
LLFFHOLD = 8


def natural_sort_key(s):
    """Sort filenames naturally so image10 comes after image9."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


def get_split(scene_dir: str, llffhold: int = LLFFHOLD):
    """
    Returns (train_images, test_images) as lists of filenames (no path).
    Test images are every llffhold-th image by sorted order.
    """
    img_dir = Path(scene_dir) / "images"
    all_imgs = sorted(
        [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
        key=natural_sort_key
    )
    test_imgs  = [f for i, f in enumerate(all_imgs) if i % llffhold == 0]
    train_imgs = [f for i, f in enumerate(all_imgs) if i % llffhold != 0]
    return train_imgs, test_imgs


def print_splits(data_root: str):
    print(f"{'Scene':<15} {'Total':>6} {'Train':>6} {'Test':>6}  Test files")
    print("-" * 80)
    for scene in SCENES:
        scene_dir = os.path.join(data_root, scene)
        if not os.path.exists(scene_dir):
            print(f"{scene:<15} NOT FOUND")
            continue
        train, test = get_split(scene_dir)
        print(f"{scene:<15} {len(train)+len(test):>6} {len(train):>6} {len(test):>6}  {test[:3]}{'...' if len(test)>3 else ''}")


def copy_train_images(data_root: str, out_root: str, upsample: bool = False):
    """
    Copies train images for each scene into out_root/<scene>/images/
    This is the input folder structure 3DGS expects.
    Set upsample=True if you've already SR'd the images and want to use those.
    """
    from shutil import copy2
    for scene in SCENES:
        scene_dir = os.path.join(data_root, scene)
        if not os.path.exists(scene_dir):
            continue
        train_imgs, _ = get_split(scene_dir)
        out_img_dir = os.path.join(out_root, scene, "images")
        os.makedirs(out_img_dir, exist_ok=True)
        for fname in train_imgs:
            src = os.path.join(scene_dir, "images", fname)
            dst = os.path.join(out_img_dir, fname)
            copy2(src, dst)
        print(f"[{scene}] Copied {len(train_imgs)} train images → {out_img_dir}")


if __name__ == "__main__":
    import sys
    data_root = sys.argv[1] if len(sys.argv) > 1 else "data_all"
    print_splits(data_root)
