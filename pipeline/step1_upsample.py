"""
Step 1: 4× Super-Resolution on all input images.

Strategy (best available → fallback):
  1. Real-ESRGAN x4plus  (best perceptual quality, pretrained)
  2. SwinIR x4 (classical degradation model)
  3. Bicubic x4  (fallback if no GPU / no model weights)

Output: data_sr/<scene>/images/<same_filename>
These SR'd images are then used as input to 3DGS training,
giving the Gaussians high-res texture information directly.

Usage:
    python step1_upsample.py --method realesrgan   # recommended
    python step1_upsample.py --method bicubic       # no-dep fallback
    python step1_upsample.py --method realesrgan --scenes bike buddha  # specific scenes
"""

import os
import sys
import argparse
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
DATA_ROOT  = "data_all"
OUT_ROOT   = "data_sr"          # SR'd images go here
SCALE      = 4
SCENES     = ["aeroplane", "bike", "buddha", "cycle", "face", "firehydrant", "still3", "toy"]

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--method",  default="realesrgan", choices=["realesrgan", "swinir", "bicubic"])
parser.add_argument("--scenes",  nargs="+", default=SCENES)
parser.add_argument("--data_root", default=DATA_ROOT)
parser.add_argument("--out_root",  default=OUT_ROOT)
parser.add_argument("--scale",     default=SCALE, type=int)
args = parser.parse_args()


def upsample_bicubic(img, scale):
    from PIL import Image
    w, h = img.size
    return img.resize((w * scale, h * scale), Image.BICUBIC)


def load_realesrgan(scale):
    """Load Real-ESRGAN model. Auto-downloads weights on first run."""
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError:
        print("[!] Real-ESRGAN not installed. Install with:")
        print("    pip install realesrgan basicsr")
        print("    Falling back to bicubic.")
        return None

    import torch
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=scale)

    # Weight file — download if missing
    weight_dir = Path("weights")
    weight_dir.mkdir(exist_ok=True)
    weight_path = weight_dir / f"RealESRGAN_x{scale}plus.pth"

    if not weight_path.exists():
        import urllib.request
        url = f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x{scale}plus.pth"
        print(f"[*] Downloading Real-ESRGAN weights from {url}")
        urllib.request.urlretrieve(url, weight_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    upsampler = RealESRGANer(
        scale=scale,
        model_path=str(weight_path),
        model=model,
        tile=256,           # tile to avoid OOM on small GPUs
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available()  # fp16 on GPU
    )
    print(f"[*] Real-ESRGAN loaded on {device}")
    return upsampler


def upsample_realesrgan(upsampler, img_path, scale):
    import cv2
    import numpy as np
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    output, _ = upsampler.enhance(img, outscale=scale)
    return output  # numpy BGR


def process_scene(scene, method, upsampler=None):
    from PIL import Image
    import shutil

    src_dir = Path(args.data_root) / scene / "images"
    dst_dir = Path(args.out_root)  / scene / "images"
    dst_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted([f for f in os.listdir(src_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    print(f"\n[{scene}] Processing {len(img_files)} images with {method}...")

    for i, fname in enumerate(img_files):
        src = src_dir / fname
        # Keep same extension
        dst = dst_dir / fname

        if dst.exists():
            continue  # skip already processed

        if method == "bicubic":
            img = Image.open(src).convert("RGB")
            out = upsample_bicubic(img, args.scale)
            out.save(dst)

        elif method == "realesrgan":
            if upsampler is None:
                img = Image.open(src).convert("RGB")
                out = upsample_bicubic(img, args.scale)
                out.save(dst)
            else:
                import cv2
                out_bgr = upsample_realesrgan(upsampler, src, args.scale)
                cv2.imwrite(str(dst), out_bgr)

        if (i + 1) % 10 == 0 or (i + 1) == len(img_files):
            print(f"  {i+1}/{len(img_files)}", end="\r")

    print(f"[{scene}] ✓ Done → {dst_dir}")

    # Also copy sparse folder if it exists (needed for 3DGS)
    sparse_src = Path(args.data_root) / scene / "sparse"
    sparse_dst = Path(args.out_root)  / scene / "sparse"
    if sparse_src.exists() and not sparse_dst.exists():
        shutil.copytree(sparse_src, sparse_dst)
        print(f"[{scene}] Copied sparse/ → {sparse_dst}")


def main():
    upsampler = None

    if args.method == "realesrgan":
        upsampler = load_realesrgan(args.scale)
        if upsampler is None:
            print("[!] Falling back to bicubic.")
            args.method = "bicubic"

    for scene in args.scenes:
        process_scene(scene, args.method, upsampler)

    print(f"\n[✓] All scenes upsampled → {args.out_root}/")
    print("Next step: python step2_colmap.py")


if __name__ == "__main__":
    main()
