#!/usr/bin/env python3
"""
MASt3R Reconstruction — Group B Scenes (HPC / Maximum Quality)
===============================================================
Scenes : buddha, firehydrant, toy  (configurable via --scenes)
Strategy: Complete graph (all N^2 pairs), full batch inference,
          ALL pairs used for global alignment, no descriptor stripping,
          COLMAP text output for 3DGS.

Differences from the Kaggle-optimised version
----------------------------------------------
  - No descriptor stripping  : full desc + desc_conf retained in memory
  - No pair subsampling      : every pair participates in global alignment
  - No model CPU offloading  : model stays on GPU the whole time
  - No chunking overhead     : pairs collated and sent as one batch
  - Higher iteration default : niter=500 (was 300)
  - Zero confidence filtering: conf_percentile=0 by default (keep all pts)
  - Multi-GPU support        : --device cuda for DataParallel across all GPUs
  - Higher-quality PLY       : binary PLY output (smaller, faster I/O)

Usage
-----
    python run_mast3r_groupb_hpc.py \
        --data_root /path/to/data-given-3dgs \
        --out_root  /path/to/mast3r_sparse \
        --mast3r_dir /path/to/mast3r \
        --scenes buddha,firehydrant,toy \
        --niter 500
"""

import argparse
import gc
import os
import struct
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage


# ─── ARGUMENT PARSING ────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MASt3R sparse reconstruction for 3DGS — HPC maximum quality."
    )
    parser.add_argument(
        "--data_root", type=Path, required=True,
        help="Root directory containing scene folders (each with images/ subfolder)."
    )
    parser.add_argument(
        "--out_root", type=Path, default=Path("mast3r_sparse"),
        help="Output directory for sparse reconstruction results."
    )
    parser.add_argument(
        "--mast3r_dir", type=Path, required=True,
        help="Path to the cloned mast3r repository."
    )
    parser.add_argument(
        "--scenes", type=str, default="buddha,firehydrant,toy",
        help="Comma-separated scene names to process."
    )
    parser.add_argument(
        "--image_size", type=int, default=512,
        help="MASt3R image processing size (default: 512)."
    )
    parser.add_argument(
        "--scene_graph", type=str, default="complete",
        help="Scene graph type: 'complete' = all N^2 pairs."
    )
    parser.add_argument(
        "--niter", type=int, default=500,
        help="Global alignment iterations (500 for HPC quality)."
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device string. 'cuda' uses all visible GPUs via DataParallel; "
             "'cuda:0' pins to a single card."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to MASt3R checkpoint (.pth). If not provided, downloads to mast3r_dir/checkpoints/."
    )
    parser.add_argument(
        "--conf_percentile", type=int, default=0,
        help="Confidence percentile threshold for point cloud filtering. "
             "0 = keep every point (HPC default). Set to 10 to discard bottom decile."
    )
    parser.add_argument(
        "--lr", type=float, default=0.01,
        help="Learning rate for global alignment (default 0.01)."
    )
    parser.add_argument(
        "--schedule", type=str, default="cosine",
        help="LR schedule for global alignment: 'cosine' or 'linear'."
    )
    parser.add_argument(
        "--no_zip", action="store_true",
        help="Skip zipping the output at the end."
    )
    return parser.parse_args()


# ─── GPU INFO ─────────────────────────────────────────────────────────────────

def print_gpu_info():
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    print(f"GPU count       : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} — {props.total_memory / 1e9:.1f} GB")


# ─── CHECKPOINT ───────────────────────────────────────────────────────────────

CKPT_NAME = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
CKPT_URL  = f"https://download.europe.naverlabs.com/ComputerVision/MASt3R/{CKPT_NAME}"


def ensure_checkpoint(mast3r_dir: Path, checkpoint_override: str = None) -> str:
    if checkpoint_override and os.path.exists(checkpoint_override):
        print(f"Using provided checkpoint: {checkpoint_override}")
        return checkpoint_override

    ckpt_dir  = mast3r_dir / "checkpoints"
    ckpt_path = ckpt_dir / CKPT_NAME
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        print(f"Downloading checkpoint (~1.1 GB) to {ckpt_path} ...")
        urllib.request.urlretrieve(CKPT_URL, str(ckpt_path))
        print(f"Saved to {ckpt_path}")
    else:
        print(f"Checkpoint already exists: {ckpt_path}")

    size_gb = os.path.getsize(str(ckpt_path)) / 1e9
    print(f"Checkpoint size: {size_gb:.2f} GB")
    return str(ckpt_path)


# ─── MODEL LOADING ───────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: str):
    """
    Load AsymmetricMASt3R and, if multiple GPUs are available and the user
    has not pinned to a specific card, wrap with DataParallel.
    """
    from mast3r.model import AsymmetricMASt3R

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = AsymmetricMASt3R(
        enc_depth=24, dec_depth=12,
        enc_embed_dim=1024, dec_embed_dim=768,
        enc_num_heads=16, dec_num_heads=12,
        pos_embed='RoPE100', img_size=(512, 512),
        head_type='catmlp+dpt', output_mode='pts3d+desc24',
        depth_mode=('exp', float('-inf'), float('inf')),
        conf_mode=('exp', 1, float('inf')),
        patch_embed_cls='PatchEmbedDust3R',
        two_confs=True,
        desc_conf_mode=('exp', 0, float('inf'))
    )
    model.load_state_dict(ckpt['model'], strict=False)

    # Resolve primary device
    primary = torch.device(device if device != "cuda" else "cuda:0")
    model   = model.to(primary).eval()

    # Wrap with DataParallel when >1 GPU is available and no card was pinned
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1 and device == "cuda":
        model = torch.nn.DataParallel(model)
        print(f"DataParallel across {n_gpus} GPUs: "
              f"{[torch.cuda.get_device_name(i) for i in range(n_gpus)]}")
    else:
        print(f"Model loaded on {next(model.parameters()).device}")

    return model


# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def get_image_paths(scene_dir):
    img_dir = Path(scene_dir) / "images"
    if not img_dir.exists():
        img_dir = Path(scene_dir)
        print(f"  Warning: no 'images' subfolder, using scene root")
    exts  = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    paths = sorted([p for p in img_dir.iterdir() if p.suffix in exts])
    print(f"  Found {len(paths)} images in {img_dir}")
    return [str(p) for p in paths]


def get_original_size(img_path):
    with PILImage.open(img_path) as im:
        return im.size  # (W, H)


def rotation_matrix_to_quaternion(R):
    """3x3 rotation matrix → [qw, qx, qy, qz]"""
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s  = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2,1] - R[1,2]) * s
        qy = (R[0,2] - R[2,0]) * s
        qz = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s  = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        qw = (R[2,1] - R[1,2]) / s
        qx = 0.25 * s
        qy = (R[0,1] + R[1,0]) / s
        qz = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s  = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        qw = (R[0,2] - R[2,0]) / s
        qx = (R[0,1] + R[1,0]) / s
        qy = 0.25 * s
        qz = (R[1,2] + R[2,1]) / s
    else:
        s  = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        qw = (R[1,0] - R[0,1]) / s
        qx = (R[0,2] + R[2,0]) / s
        qy = (R[1,2] + R[2,1]) / s
        qz = 0.25 * s
    return np.array([qw, qx, qy, qz])


def write_ply_binary(path, points, colors):
    """
    Binary PLY — significantly faster to write and smaller on disk than ASCII.
    points : (N, 3) float32
    colors : (N, 3) uint8
    """
    n = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode()

    # interleave xyz + rgb into a structured array
    dt    = np.dtype([('x','f4'),('y','f4'),('z','f4'),
                      ('r','u1'),('g','u1'),('b','u1')])
    verts = np.empty(n, dtype=dt)
    pts32 = points.astype(np.float32)
    verts['x'] = pts32[:,0]; verts['y'] = pts32[:,1]; verts['z'] = pts32[:,2]
    verts['r'] = colors[:,0]; verts['g'] = colors[:,1]; verts['b'] = colors[:,2]

    with open(path, 'wb') as f:
        f.write(header)
        f.write(verts.tobytes())


def save_pointcloud(pts3d, imgs, confidence, out_dir, conf_percentile=0):
    pts_all, cols_all, conf_all = [], [], []
    for pt, im, cf in zip(pts3d, imgs, confidence):
        pts_all.append(pt.reshape(-1, 3))
        cols_all.append((np.clip(im, 0, 1).reshape(-1, 3) * 255).astype(np.uint8))
        conf_all.append(cf.reshape(-1))

    pts_all  = np.concatenate(pts_all)
    cols_all = np.concatenate(cols_all)
    conf_all = np.concatenate(conf_all)

    if conf_percentile > 0:
        mask    = conf_all > np.percentile(conf_all, conf_percentile)
        pts_all  = pts_all[mask]
        cols_all = cols_all[mask]
        print(f"  Confidence filter (>{conf_percentile}th pct): "
              f"{mask.sum():,} / {len(mask):,} points kept")
    else:
        print(f"  No confidence filtering — keeping all {len(pts_all):,} points")

    ply_path = os.path.join(out_dir, "pointcloud.ply")
    write_ply_binary(ply_path, pts_all, cols_all)
    print(f"  Saved {len(pts_all):,} points → pointcloud.ply "
          f"({os.path.getsize(ply_path)/1e6:.1f} MB, binary)")
    return ply_path


def save_colmap_cameras(focals, poses, img_paths,
                         orig_W, orig_H, mast3r_W, out_dir):
    sparse_dir = os.path.join(out_dir, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)

    scale  = orig_W / mast3r_W
    cx, cy = orig_W / 2.0, orig_H / 2.0

    with open(os.path.join(sparse_dir, "cameras.txt"), 'w') as f:
        f.write("# CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy\n")
        for i, focal_m in enumerate(focals.flatten()):
            focal_orig = float(focal_m) * scale
            f.write(f"{i+1} PINHOLE {orig_W} {orig_H} "
                    f"{focal_orig:.6f} {focal_orig:.6f} "
                    f"{cx:.6f} {cy:.6f}\n")

    with open(os.path.join(sparse_dir, "images.txt"), 'w') as f:
        f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")
        f.write("# POINTS2D[] as (X Y POINT3D_ID)\n")
        for i, (c2w, img_path) in enumerate(zip(poses, img_paths)):
            w2c  = np.linalg.inv(c2w)
            R, t = w2c[:3, :3], w2c[:3, 3]
            q    = rotation_matrix_to_quaternion(R)
            name = os.path.basename(img_path)
            f.write(f"{i+1} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} "
                    f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {i+1} {name}\n")
            f.write("\n")

    with open(os.path.join(sparse_dir, "points3D.txt"), 'w') as f:
        f.write("# empty — 3DGS initialised from pointcloud.ply\n")

    focal_range = focals.flatten() * scale
    print(f"  COLMAP sparse → {sparse_dir}")
    print(f"  Resolution: {orig_W}x{orig_H} | "
          f"focal range: [{focal_range.min():.1f}, {focal_range.max():.1f}]")


# ─── FULL BATCH INFERENCE ─────────────────────────────────────────────────────

def inference_full(pairs, model, device):
    """
    Run MASt3R inference on ALL pairs in a single collated batch.
    Descriptors are kept intact — no stripping, no chunking.
    On multi-GPU setups the DataParallel wrapper distributes the batch
    across cards automatically.
    """
    from tqdm import tqdm
    from dust3r.utils.device import collate_with_cat
    from dust3r.inference import loss_of_one_batch

    n_pairs = len(pairs)
    print(f"  Full batch inference: {n_pairs} pairs (no chunking, no descriptor stripping)")

    # Collect results pair-by-pair but keep everything on CPU to avoid
    # OOM during the accumulation phase; the model itself stays on GPU.
    all_view1, all_view2, all_pred1, all_pred2 = {}, {}, {}, {}

    def accumulate(target, source):
        for k, v in source.items():
            if isinstance(v, torch.Tensor):
                target.setdefault(k, []).append(v.cpu())
            elif isinstance(v, list):
                target.setdefault(k, []).extend(v)
            else:
                target.setdefault(k, []).append(v)

    with torch.no_grad():
        for pair in tqdm(pairs, desc="  Inference", unit="pair"):
            result = loss_of_one_batch(
                collate_with_cat([pair]),
                model, None, device
            )
            # accumulate on CPU — descriptors kept
            accumulate(all_view1, result['view1'])
            accumulate(all_view2, result['view2'])
            accumulate(all_pred1, result['pred1'])
            accumulate(all_pred2, result['pred2'])
            del result

    print("  Concatenating results...")

    def final_cat(d):
        out = {}
        for k, vlist in d.items():
            if vlist and isinstance(vlist[0], torch.Tensor):
                out[k] = torch.cat(vlist, dim=0)
            else:
                out[k] = vlist
        return out

    output = {
        'view1': final_cat(all_view1),
        'view2': final_cat(all_view2),
        'pred1': final_cat(all_pred1),
        'pred2': final_cat(all_pred2),
    }

    n_assembled = output['pred1']['pts3d'].shape[0]
    print(f"  Assembled {n_assembled} pairs.")
    print(f"  pts3d shape  : {output['pred1']['pts3d'].shape}")
    if 'desc' in output['pred1']:
        print(f"  desc shape   : {output['pred1']['desc'].shape}")
    return output


# ─── SCENE PROCESSING ────────────────────────────────────────────────────────

def run_scene(model, scene_name, dataset_path, out_root,
              image_size, scene_graph, niter, device, conf_percentile,
              lr, schedule):
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from dust3r.utils.device import to_numpy

    print(f"\n{'='*60}")
    print(f"  Scene: {scene_name}")
    print(f"{'='*60}")

    scene_dir = os.path.join(str(dataset_path), scene_name)
    if not os.path.exists(scene_dir):
        print(f"  ERROR: scene directory not found: {scene_dir}")
        print(f"  Available: {os.listdir(str(dataset_path))}")
        return

    out_dir = os.path.join(str(out_root), scene_name)
    os.makedirs(out_dir, exist_ok=True)

    img_paths = get_image_paths(scene_dir)
    if len(img_paths) == 0:
        print(f"  ERROR: no images found in {scene_dir}")
        return

    orig_W, orig_H = get_original_size(img_paths[0])
    n = len(img_paths)
    n_pairs = n * (n - 1)   # symmetrize=True → N*(N-1) directed pairs

    print(f"  Original res : {orig_W}x{orig_H}")
    print(f"  Submission   : {orig_W*4}x{orig_H*4}")
    print(f"  Images       : {n}")
    print(f"  Pairs        : {n_pairs} ({scene_graph} graph, ALL used for alignment)")

    images = load_images(img_paths, size=image_size, verbose=False)
    mast3r_H, mast3r_W = images[0]['img'].shape[-2:]
    print(f"  MASt3R res   : {mast3r_W}x{mast3r_H}")
    print(f"  Focal scale  : {orig_W/mast3r_W:.4f} (MASt3R → original)")

    pairs = make_pairs(
        images,
        scene_graph=scene_graph,
        prefilter=None,
        symmetrize=True      # full directed graph
    )
    print(f"  Total pairs after symmetrize: {len(pairs)}")

    # ── Full inference — no chunking, no descriptor stripping ──────────────
    output = inference_full(pairs, model, device)

    # ── Global alignment on ALL pairs ─────────────────────────────────────
    # Resolve primary device (DataParallel puts params on cuda:0)
    align_device = (torch.device("cuda:0")
                    if device == "cuda" and torch.cuda.device_count() > 1
                    else torch.device(device))

    print(f"\n  Global alignment on {align_device}")
    print(f"  Pairs used   : {output['pred1']['pts3d'].shape[0]} "
          f"(no subsampling)")
    print(f"  Iterations   : {niter}, schedule={schedule}, lr={lr}")

    scene_obj = global_aligner(
        output,
        device=align_device,
        mode=GlobalAlignerMode.PointCloudOptimizer
    )

    # output still needed by scene_obj internally; do not delete yet
    loss = scene_obj.compute_global_alignment(
        init='mst',
        niter=niter,
        schedule=schedule,
        lr=lr
    )
    print(f"  Alignment loss: {loss:.6f}")

    # Extract results
    imgs       = to_numpy(scene_obj.imgs)
    focals     = to_numpy(scene_obj.get_focals())
    poses      = to_numpy(scene_obj.get_im_poses())   # c2w (N,4,4)
    pts3d      = to_numpy(scene_obj.get_pts3d())
    confidence = to_numpy(scene_obj.get_conf())

    # Save
    save_pointcloud(pts3d, imgs, confidence, out_dir,
                    conf_percentile=conf_percentile)
    save_colmap_cameras(focals, poses, img_paths,
                        orig_W, orig_H, mast3r_W, out_dir)

    del scene_obj, output, imgs, pts3d, confidence
    gc.collect()
    torch.cuda.empty_cache()

    print(f"  Scene {scene_name} complete.")


# ─── VERIFY OUTPUTS ──────────────────────────────────────────────────────────

def verify_outputs(scenes, out_root):
    print("\nOutput structure:")
    for scene in scenes:
        scene_out = os.path.join(str(out_root), scene)
        if not os.path.exists(scene_out):
            print(f"  {scene}: MISSING")
            continue
        sparse = os.path.join(scene_out, "sparse", "0")
        ply    = os.path.join(scene_out, "pointcloud.ply")
        print(f"  {scene}/")
        if os.path.exists(ply):
            print(f"    pointcloud.ply : OK ({os.path.getsize(ply)/1e6:.1f} MB)")
        else:
            print(f"    pointcloud.ply : MISSING")
        for fname in ["cameras.txt", "images.txt", "points3D.txt"]:
            fpath = os.path.join(sparse, fname)
            print(f"    sparse/0/{fname}: {'OK' if os.path.exists(fpath) else 'MISSING'}")


# ─── ZIP RESULTS ──────────────────────────────────────────────────────────────

def zip_results(scenes, out_root):
    zip_path = str(out_root) + ".zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for scene in scenes:
            scene_out = os.path.join(str(out_root), scene)
            if not os.path.exists(scene_out):
                continue
            for root, dirs, files in os.walk(scene_out):
                for file in files:
                    fpath   = os.path.join(root, file)
                    arcname = os.path.relpath(fpath, str(out_root))
                    zf.write(fpath, arcname)
    size_mb = os.path.getsize(zip_path) / 1e6
    print(f"\nZipped output: {zip_path} ({size_mb:.1f} MB)")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    scenes     = [s.strip() for s in args.scenes.split(",") if s.strip()]
    mast3r_dir = args.mast3r_dir.resolve()
    data_root  = args.data_root.resolve()
    out_root   = args.out_root.resolve()

    print(f"Data root  : {data_root}")
    print(f"Output root: {out_root}")
    print(f"MASt3R dir : {mast3r_dir}")
    print(f"Scenes     : {scenes}")
    print(f"Device     : {args.device}")
    print()

    sys.path.insert(0, str(mast3r_dir))
    sys.path.insert(0, str(mast3r_dir / "dust3r"))

    os.makedirs(str(out_root), exist_ok=True)

    print_gpu_info()
    print()

    ckpt_path = ensure_checkpoint(mast3r_dir, args.checkpoint)
    print()

    model = load_model(ckpt_path, args.device)

    for scene in scenes:
        try:
            run_scene(
                model, scene, data_root, out_root,
                image_size=args.image_size,
                scene_graph=args.scene_graph,
                niter=args.niter,
                device=args.device,
                conf_percentile=args.conf_percentile,
                lr=args.lr,
                schedule=args.schedule,
            )
        except Exception as e:
            print(f"\n  ERROR on {scene}: {e}")
            import traceback
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("All scenes processed.")
    print(f"Results in: {out_root}")

    verify_outputs(scenes, out_root)

    if not args.no_zip:
        zip_results(scenes, out_root)


if __name__ == "__main__":
    main()