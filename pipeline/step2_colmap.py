"""
Step 2: Prepare scene data for 3DGS training.

For scenes WITH sparse/: convert to 3DGS-compatible format using skip_matching.
For scenes WITHOUT sparse/: run full COLMAP (feature extract → match → map → undistort).

This script prepares a `data_ready/<scene>/` folder that train.py can directly use.

IMPORTANT: Only train images (LLFF split) are fed to COLMAP / 3DGS.

Usage:
    python step2_colmap.py
    python step2_colmap.py --scenes aeroplane cycle     # specific scenes only
    python step2_colmap.py --colmap_path "C:/path/to/colmap.exe"  # if not in PATH
    python step2_colmap.py --no_gpu                    # CPU-only COLMAP
"""

import os
import sys
import argparse
import shutil
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.llff_split import get_split, natural_sort_key

# ── Config ──────────────────────────────────────────────────────────────────
SR_ROOT    = "data_sr"      # input: SR'd images (output of step1)
READY_ROOT = "data_ready"   # output: 3DGS-ready data

SCENES_WITH_SPARSE    = ["bike", "buddha", "firehydrant", "toy"]
SCENES_WITHOUT_SPARSE = ["aeroplane", "cycle", "face", "still3"]
ALL_SCENES = SCENES_WITH_SPARSE + SCENES_WITHOUT_SPARSE

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--scenes",       nargs="+", default=ALL_SCENES)
parser.add_argument("--sr_root",      default=SR_ROOT)
parser.add_argument("--ready_root",   default=READY_ROOT)
parser.add_argument("--colmap_path",  default=r"colmap-x64-windows-cuda\bin\colmap.exe",  help="Path to colmap executable")
parser.add_argument("--no_gpu",       action="store_true")
args = parser.parse_args()

GPU_FLAG = "1" if not args.no_gpu else "0"  # kept for reference but not used in new COLMAP


def run(cmd, cwd=None):
    print(f"  $ {cmd}")
    ret = subprocess.run(cmd, shell=True, cwd=cwd)
    if ret.returncode != 0:
        print(f"[!] Command failed with code {ret.returncode}")
        sys.exit(ret.returncode)


def copy_train_images(scene, src_root, dst_dir):
    """
    Copy ALL images into dst_dir/input/ — 3DGS applies its own LLFF hold
    internally when --eval is set, so it needs all images present.
    Also copies all images into dst_dir/images/ so train.py can find them.
    """
    input_dir = Path(dst_dir) / "input"
    images_dir = Path(dst_dir) / "images"
    input_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    scene_src = Path(src_root) / scene
    # Use original data_all for full image set (SR root may only have train)
    scene_orig = Path("data_all") / scene
    train_imgs, test_imgs = get_split(scene_orig)
    all_imgs = train_imgs + test_imgs

    for fname in all_imgs:
        # Copy SR version if available, else fall back to original
        sr_path = scene_src / "images" / fname
        orig_path = scene_orig / "images" / fname
        src = sr_path if sr_path.exists() else orig_path
        shutil.copy2(src, input_dir / fname)
        shutil.copy2(src, images_dir / fname)

    print(f"  Copied {len(all_imgs)} images ({len(train_imgs)} train + {len(test_imgs)} test)")
    return train_imgs, test_imgs


def prepare_with_sparse(scene):
    """
    For scenes with precomputed sparse/: just copy images + sparse directly.
    No undistortion needed — these are already standard pinhole images.
    """
    print(f"\n[{scene}] Has sparse data — skipping COLMAP matching")

    dst_scene = Path(args.ready_root) / scene
    dst_scene.mkdir(parents=True, exist_ok=True)

    # Copy ALL images directly into images/ (3DGS reads from here)
    copy_train_images(scene, args.sr_root, dst_scene)

    # Copy sparse reconstruction from original data
    sparse_src = Path("data_all") / scene / "sparse" / "0"
    sparse_dst = dst_scene / "sparse" / "0"
    if sparse_dst.exists():
        shutil.rmtree(sparse_dst)
    shutil.copytree(sparse_src, sparse_dst)
    print(f"  Copied sparse/ from original data")
    print(f"[{scene}] ✓ Ready at {dst_scene}")


def prepare_without_sparse(scene):
    """
    For scenes without sparse/: run full COLMAP pipeline.
    feature_extractor → exhaustive_matcher → mapper → image_undistorter
    """
    print(f"\n[{scene}] No sparse data — running full COLMAP")

    src_scene = Path(args.sr_root) / scene
    dst_scene = Path(args.ready_root) / scene
    dst_scene.mkdir(parents=True, exist_ok=True)

    # Copy train images
    copy_train_images(scene, args.sr_root, dst_scene)

    distorted_dir = dst_scene / "distorted"
    db_path       = distorted_dir / "database.db"
    sparse_dir    = distorted_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    input_dir = dst_scene / "input"

    # 1. Feature extraction (new COLMAP 3.10+: GPU auto-detected, no explicit flag needed)
    run(
        f"{args.colmap_path} feature_extractor"
        f" --database_path \"{db_path}\""
        f" --image_path \"{input_dir}\""
        f" --ImageReader.single_camera 1"
    )

    # 2. Feature matching — sequential is better for ordered image captures
    run(
        f"{args.colmap_path} sequential_matcher"
        f" --database_path \"{db_path}\""
    )

    # 3. Sparse reconstruction (mapper)
    run(
        f"{args.colmap_path} mapper"
        f" --database_path \"{db_path}\""
        f" --image_path \"{input_dir}\""
        f" --output_path \"{sparse_dir}\""
        f" --Mapper.ba_global_function_tolerance=0.000001"
    )

    # 4. Undistort images — find first valid sparse reconstruction (0, 1, ...)
    sparse_model = None
    for i in range(5):
        candidate = sparse_dir / str(i)
        if candidate.exists() and (candidate / "images.bin").exists():
            sparse_model = candidate
            break
    if sparse_model is None:
        print(f"[{scene}] ✗ No valid sparse reconstruction found in {sparse_dir}")
        return

    run(
        f"{args.colmap_path} image_undistorter"
        f" --image_path \"{input_dir}\""
        f" --input_path \"{sparse_model}\""
        f" --output_path \"{dst_scene}\""
        f" --output_type COLMAP"
    )

    # Ensure images/ has ALL images (undistorter may miss some)
    images_out = dst_scene / "images"
    images_out.mkdir(exist_ok=True)
    for fname in os.listdir(input_dir):
        dst_img = images_out / fname
        if not dst_img.exists():
            shutil.copy2(input_dir / fname, dst_img)

    # 5. New COLMAP puts sparse files directly in sparse/ not sparse/0/ — fix it
    sparse_out = dst_scene / "sparse" / "0"
    sparse_flat = dst_scene / "sparse"
    if not sparse_out.exists() and (sparse_flat / "images.bin").exists():
        sparse_out.mkdir(parents=True, exist_ok=True)
        for f in ["cameras.bin", "images.bin", "points3D.bin", "frames.bin", "rigs.bin"]:
            src_f = sparse_flat / f
            if src_f.exists():
                shutil.move(str(src_f), str(sparse_out / f))
        print(f"  Fixed sparse/0/ structure")

    # 6. Save estimated poses for submission
    poses_dir = Path("estimated_poses") / scene
    poses_dir.mkdir(parents=True, exist_ok=True)
    if sparse_out.exists():
        for f in ["cameras.txt", "images.txt", "points3D.txt",
                  "cameras.bin", "images.bin", "points3D.bin"]:
            src_f = sparse_out / f
            if src_f.exists():
                shutil.copy2(src_f, poses_dir / f)
        print(f"  Estimated poses saved → {poses_dir}")

    # 7. Verify camera count
    images_bin = sparse_out / "images.bin"
    if images_bin.exists():
        import struct
        with open(images_bin, "rb") as fid:
            n_cams = struct.unpack("<Q", fid.read(8))[0]
        print(f"  Reconstructed {n_cams} cameras")
        if n_cams < 5:
            print(f"  ⚠ WARNING: Very few cameras — COLMAP may have failed for {scene}")

    print(f"[{scene}] ✓ Ready at {dst_scene}")


def main():
    for scene in args.scenes:
        if scene in SCENES_WITH_SPARSE:
            prepare_with_sparse(scene)
        else:
            prepare_without_sparse(scene)

    print(f"\n[✓] All scenes ready in {args.ready_root}/")
    print("Next step: python step3_train_3dgs.py")


if __name__ == "__main__":
    main()
