"""
Step 4: Render test views + build submission.

This script:
  1. Calls gaussian-splatting/render.py for each scene to render test views
  2. Optionally applies a 2nd-pass SR (Real-ESRGAN) on rendered outputs
  3. Copies rendered test images into submission/<scene>/ with EXACT original filenames
  4. Also copies .ply point clouds (required for visualization)
  5. Calls imgs2csv.py to generate submission.csv

Usage:
    python step4_render_submit.py
    python step4_render_submit.py --post_sr          # apply SR on rendered outputs
    python step4_render_submit.py --scenes bike toy  # subset
"""

import os
import sys
import re
import shutil
import subprocess
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.llff_split import get_split, natural_sort_key, SCENES

DATA_ALL    = "data_all"
DATA_SR     = "data_sr"
READY_ROOT  = "data_ready"
OUTPUT_ROOT = "output"
SUBMIT_DIR  = "submission"
GS_RENDER   = "gaussian-splatting/render.py"

parser = argparse.ArgumentParser()
parser.add_argument("--scenes",      nargs="+", default=SCENES)
parser.add_argument("--post_sr",     action="store_true", help="Apply Real-ESRGAN on rendered outputs")
parser.add_argument("--data_root",   default=DATA_ALL)
parser.add_argument("--sr_root",     default=DATA_SR)
parser.add_argument("--ready_root",  default=READY_ROOT)
parser.add_argument("--output_root", default=OUTPUT_ROOT)
parser.add_argument("--submit_dir",  default=SUBMIT_DIR)
args = parser.parse_args()


# ── Post-SR helper ────────────────────────────────────────────────────────────
def load_realesrgan():
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        import torch
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        wp = Path("weights/RealESRGAN_x4plus.pth")
        if not wp.exists():
            import urllib.request
            wp.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve(
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                wp
            )
        return RealESRGANer(scale=4, model_path=str(wp), model=model,
                            tile=256, tile_pad=10, pre_pad=0,
                            half=torch.cuda.is_available())
    except ImportError:
        print("[!] Real-ESRGAN not available for post-SR. Skipping.")
        return None


# ── Render a scene ─────────────────────────────────────────────────────────────
def render_scene(scene):
    model_path = Path(args.output_root) / scene
    src_path   = Path(args.ready_root) / scene

    if not model_path.exists():
        print(f"[{scene}] SKIP — no trained model at {model_path}")
        return None

    print(f"\n[{scene}] Rendering test views...")
    cmd = [
        sys.executable, GS_RENDER,
        "--model_path",  str(model_path),
        "--source_path", str(src_path),
        "--skip_train",              # only render test split
        "--quiet",
    ]
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"[{scene}] ✗ Render failed (code {ret.returncode})")
        return None

    # rendered images are in output/<scene>/test/ours_<iter>/renders/
    render_dirs = sorted((model_path / "test").glob("ours_*"))
    if not render_dirs:
        print(f"[{scene}] ✗ No render output found in {model_path / 'test'}")
        return None

    latest = render_dirs[-1] / "renders"
    print(f"[{scene}] ✓ Rendered → {latest}")
    return latest


# ── Match rendered images to original test filenames ──────────────────────────
def copy_to_submission(scene, render_dir, upsampler=None):
    """
    3DGS render.py names outputs as 00000.png, 00001.png, etc.
    We need to rename them back to the original test image filenames.
    """
    # Get original test filenames in sorted order
    scene_src = Path(args.data_root) / scene
    _, test_imgs = get_split(scene_src)

    # Get rendered files in sorted order
    rendered = sorted(
        [f for f in os.listdir(render_dir) if f.lower().endswith(('.png', '.jpg'))],
        key=natural_sort_key
    )

    if len(rendered) != len(test_imgs):
        print(f"[{scene}] ⚠ Mismatch: {len(rendered)} rendered vs {len(test_imgs)} test images expected")
        # Use min to avoid crash
        n = min(len(rendered), len(test_imgs))
        rendered  = rendered[:n]
        test_imgs = test_imgs[:n]

    dst_scene_dir = Path(args.submit_dir) / scene
    dst_scene_dir.mkdir(parents=True, exist_ok=True)

    for rfile, tfile in zip(rendered, test_imgs):
        src = Path(render_dir) / rfile
        # Keep original extension from test filename
        dst = dst_scene_dir / tfile

        if upsampler is not None:
            # Apply Real-ESRGAN on the rendered output
            import cv2
            img = cv2.imread(str(src))
            out, _ = upsampler.enhance(img, outscale=1)  # already at 4x from SR input
            cv2.imwrite(str(dst), out)
        else:
            shutil.copy2(src, dst)

    print(f"[{scene}] ✓ {len(rendered)} test views → {dst_scene_dir}")


# ── Copy point clouds ──────────────────────────────────────────────────────────
def copy_pointclouds():
    ply_dir = Path("pointclouds")
    ply_dir.mkdir(exist_ok=True)

    for scene in args.scenes:
        model_path = Path(args.output_root) / scene
        # 3DGS saves point clouds at output/<scene>/point_cloud/iteration_<N>/point_cloud.ply
        ply_files = sorted(model_path.glob("point_cloud/iteration_*/point_cloud.ply"))
        if ply_files:
            latest_ply = ply_files[-1]
            dst = ply_dir / f"{scene}.ply"
            shutil.copy2(latest_ply, dst)
            print(f"[{scene}] Point cloud → {dst}")
        else:
            print(f"[{scene}] ⚠ No point cloud found in {model_path}")


# ── Generate submission.csv ───────────────────────────────────────────────────
def generate_csv():
    csv_script = Path("imgs2csv.py")
    if not csv_script.exists():
        print("[!] imgs2csv.py not found. Skipping CSV generation.")
        return

    # Patch PRED_ROOT in the script temporarily or just run it with env
    import importlib.util, sys as _sys
    spec = importlib.util.spec_from_file_location("imgs2csv", csv_script)
    mod  = importlib.util.module_from_spec(spec)

    # Monkey-patch the PRED_ROOT inside the script
    original = csv_script.read_text()
    patched  = original.replace('PRED_ROOT = "submission"', f'PRED_ROOT = "{args.submit_dir}"')
    patched_path = Path("imgs2csv_patched.py")
    patched_path.write_text(patched)

    ret = subprocess.run([sys.executable, str(patched_path)])
    patched_path.unlink(missing_ok=True)

    if ret.returncode == 0:
        print("[✓] submission.csv generated!")
    else:
        print("[!] imgs2csv.py failed — check submission folder structure")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    upsampler = None
    if args.post_sr:
        print("[*] Loading Real-ESRGAN for post-render SR...")
        upsampler = load_realesrgan()

    for scene in args.scenes:
        render_dir = render_scene(scene)
        if render_dir:
            copy_to_submission(scene, render_dir, upsampler)

    print("\n[*] Copying point clouds...")
    copy_pointclouds()

    print("\n[*] Generating submission.csv...")
    generate_csv()

    print(f"\n[✓] Submission ready in {args.submit_dir}/")
    print("Upload submission.csv (or zip it) to Kaggle!")


if __name__ == "__main__":
    main()
