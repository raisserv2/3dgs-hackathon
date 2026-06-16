"""
Step 2b: Run hloc (SuperPoint + SuperGlue) for scenes where COLMAP failed.
Targets: aeroplane, face, still3

Uses neural feature matching instead of SIFT — works on low-res, low-texture images.
Output: data_ready/<scene>/sparse/0/ + data_ready/<scene>/images/
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch.set_num_threads(4)

import sys
import shutil
import struct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "Hierarchical-Localization"))

from hloc import extract_features, match_features, reconstruction, visualization

SCENES       = ["aeroplane", "face", "still3"]
DATA_SR      = Path("data_sr")
DATA_ALL     = Path("data_all")
READY_ROOT   = Path("data_ready")
COLMAP_PATH  = "colmap-x64-windows-cuda/bin/colmap.exe"

# hloc configs — SuperPoint + SuperGlue (best for low-res)
FEATURE_CONF = extract_features.confs["superpoint_aachen"]
MATCHER_CONF = match_features.confs["superglue"]


def process_scene(scene):
    print(f"\n{'='*60}")
    print(f"  hloc: {scene}")
    print(f"{'='*60}")

    dst_scene  = READY_ROOT / scene
    images_dir = dst_scene / "images"
    sfm_dir    = dst_scene / "sfm"
    hloc_dir   = dst_scene / "hloc"

    images_dir.mkdir(parents=True, exist_ok=True)
    sfm_dir.mkdir(parents=True, exist_ok=True)
    hloc_dir.mkdir(parents=True, exist_ok=True)

    # Copy ALL images from data_sr into images/
    src_imgs = DATA_SR / scene / "images"
    if not src_imgs.exists():
        src_imgs = DATA_ALL / scene / "images"

    copied = 0
    for f in src_imgs.iterdir():
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            dst = images_dir / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
            copied += 1
    print(f"  Copied {copied} images")

    # 1. Extract SuperPoint features
    features_path = hloc_dir / "features.h5"
    print("  [1/3] Extracting SuperPoint features...")
    extract_features.main(
        FEATURE_CONF,
        images_dir,
        feature_path=features_path
    )
    # Force CUDA and clean cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_device(0)

    # 2. Match with SuperGlue (exhaustive for small scenes)
    pairs_path  = hloc_dir / "pairs.txt"
    matches_path = hloc_dir / "matches.h5"

    # Write exhaustive pairs
    img_files = sorted([f.name for f in images_dir.iterdir()
                        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    with open(pairs_path, "w") as f:
        for i, a in enumerate(img_files):
            for b in img_files[i+1:]:
                f.write(f"{a} {b}\n")
    print(f"  [2/3] Matching {len(img_files)*(len(img_files)-1)//2} pairs with SuperGlue...")

    match_features.main(
        MATCHER_CONF,
        pairs_path,
        features=features_path,
        matches=matches_path
    )

    # 3. Reconstruct with COLMAP using hloc matches
    print("  [3/3] Running COLMAP reconstruction with neural matches...")
    sfm_model = reconstruction.main(
        sfm_dir,
        images_dir,
        pairs_path,
        features_path,
        matches_path,
    )

    if sfm_model is None:
        print(f"  ✗ Reconstruction failed for {scene}")
        return False

    # Move to sparse/0/ structure that 3DGS expects
    sparse_0 = dst_scene / "sparse" / "0"
    sparse_0.mkdir(parents=True, exist_ok=True)

    # sfm_model is a pycolmap Reconstruction — export it
    sfm_model.write(str(sparse_0))
    print(f"  Exported sparse/0/")

    # Check camera count
    images_bin = sparse_0 / "images.bin"
    if images_bin.exists():
        with open(images_bin, "rb") as fid:
            n = struct.unpack("<Q", fid.read(8))[0]
        print(f"  ✓ Reconstructed {n} cameras")
        if n < 5:
            print(f"  ⚠ Still few cameras — scene may be too challenging")

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", default=SCENES)
    args = parser.parse_args()

    for scene in args.scenes:
        process_scene(scene)

    print("\n[✓] hloc done. Now run: python step3_train_3dgs.py")


if __name__ == "__main__":
    main()
