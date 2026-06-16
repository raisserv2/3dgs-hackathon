"""
Step 3: Train 3D Gaussian Splatting on all scenes.

Trains gaussian-splatting/train.py sequentially on each scene using
the prepared data in data_ready/<scene>/

Key flags used:
  --iterations 30000        standard, gives good quality
  --eval                    enables train/test split logging
  --white_background        improves results for object-centric scenes

Output: output/<scene>/  (contains point_cloud/, cameras.json, etc.)

Usage:
    python step3_train_3dgs.py
    python step3_train_3dgs.py --scenes bike buddha   # subset
    python step3_train_3dgs.py --iterations 15000     # faster, lower quality
    python step3_train_3dgs.py --fast                 # 7000 iter quick test
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

READY_ROOT  = "data_ready"
OUTPUT_ROOT = "output"
GS_TRAIN    = "gaussian-splatting/train.py"  # relative to hackathon root
SCENES      = ["aeroplane", "bike", "buddha", "cycle", "face", "firehydrant", "still3", "toy"]

parser = argparse.ArgumentParser()
parser.add_argument("--scenes",     nargs="+", default=SCENES)
parser.add_argument("--ready_root", default=READY_ROOT)
parser.add_argument("--output",     default=OUTPUT_ROOT)
parser.add_argument("--iterations", default=30000, type=int)
parser.add_argument("--fast",       action="store_true", help="7000 iterations for quick test")
parser.add_argument("--resolution", default=-1, type=int,
                    help="Downscale resolution (-1=auto, 1=full, 2=half, etc.)")
args = parser.parse_args()

if args.fast:
    args.iterations = 7000
    print("[!] Fast mode: 7000 iterations")


def train_scene(scene):
    src = Path(args.ready_root) / scene
    out = Path(args.output)     / scene

    if not src.exists():
        print(f"[{scene}] SKIP — {src} not found. Run step2_colmap.py first.")
        return False

    print(f"\n{'='*60}")
    print(f"  Training: {scene}  ({args.iterations} iterations)")
    print(f"  Source:   {src}")
    print(f"  Output:   {out}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, GS_TRAIN,
        "--source_path", str(src),
        "--model_path",  str(out),
        "--iterations",  str(args.iterations),
        "--eval",                      # compute metrics on test split
        "--resolution", str(args.resolution),
    ]
    if scene == "aeroplane":
        cmd.append("--sh_degree")
        cmd.append("0")
    # Small scenes: reduce densification thresholds slightly for stability
    img_count_dir = src / "images"
    if img_count_dir.exists():
        n_imgs = len(list(img_count_dir.glob("*")))
        if n_imgs < 25:
            cmd += ["--densify_until_iter", "15000"]

    print("  CMD:", " ".join(cmd))
    ret = subprocess.run(cmd)

    if ret.returncode != 0:
        print(f"[{scene}] ✗ Training failed (code {ret.returncode})")
        return False

    print(f"[{scene}] ✓ Training complete → {out}")
    return True


def main():
    failed = []
    for scene in args.scenes:
        ok = train_scene(scene)
        if not ok:
            failed.append(scene)

    print(f"\n{'='*60}")
    print(f"[✓] Training done. {len(args.scenes)-len(failed)}/{len(args.scenes)} scenes succeeded.")
    if failed:
        print(f"[!] Failed scenes: {failed}")
    print("Next step: python step4_render_submit.py")


if __name__ == "__main__":
    main()
