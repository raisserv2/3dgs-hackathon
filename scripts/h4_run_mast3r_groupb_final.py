#!/usr/bin/env python3
"""
MASt3R-SfM final pipeline for Group B scenes.

This version is the full-quality path:
- complete scene graph
- no pair subsampling
- no chunking
- no descriptor stripping
- sparse_global_alignment from the MASt3R repo
- dense point cloud export plus COLMAP text sparse model

It is intended for HPC runs where quality matters more than runtime.
"""

from __future__ import annotations

import argparse
import json
import gc
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage


ROOT_DIR = Path(__file__).resolve().parent
HPC_HOME_ROOT = Path(os.environ.get("HPC_HOME_ROOT", "/home/da24b009"))
HPC_STORAGE_ROOT = Path(os.environ.get("HPC_STORAGE_ROOT", "/storage/da24b009"))
CKPT_NAME = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
CKPT_URL = f"https://download.europe.naverlabs.com/ComputerVision/MASt3R/{CKPT_NAME}"
DEFAULT_DATA_ROOT = HPC_STORAGE_ROOT / "data-given-3dgs"
DEFAULT_OUT_ROOT = HPC_STORAGE_ROOT / "mast3r_sparse_final"
DEFAULT_MAST3R_DIR = HPC_HOME_ROOT / "mast3r"
DEFAULT_GSCPR_DIR = HPC_HOME_ROOT / "GS-CPR"
DEFAULT_CHECKPOINT = HPC_STORAGE_ROOT / "checkpoints" / CKPT_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MASt3R-SfM final quality pipeline for Group B scenes")
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--mast3r_dir", type=Path, default=DEFAULT_MAST3R_DIR)
    parser.add_argument("--gscpr_dir", type=Path, default=DEFAULT_GSCPR_DIR)
    parser.add_argument("--scenes", type=str, default="buddha,firehydrant,toy")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--niter1", type=int, default=500)
    parser.add_argument("--niter2", type=int, default=500)
    parser.add_argument("--lr1", type=float, default=0.07)
    parser.add_argument("--lr2", type=float, default=0.01)
    parser.add_argument("--matching_conf_thr", type=float, default=0.0)
    parser.add_argument("--conf_percentile", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--shared_intrinsics", action="store_true", default=True)
    parser.add_argument("--no_shared_intrinsics", dest="shared_intrinsics", action="store_false")
    parser.add_argument("--clean_depth", action="store_true", default=True)
    parser.add_argument("--no_clean_depth", dest="clean_depth", action="store_false")
    parser.add_argument("--tsdf_thresh", type=float, default=0.0)
    parser.add_argument("--no_zip", action="store_true")
    parser.add_argument("--gscpr_cmd", type=str, default=None,
                        help="Optional external GSCPR command to run after each scene. If set, it receives the scene sparse dir as the last argument.")
    return parser.parse_args()


def bootstrap_mast3r_paths(mast3r_dir: Path) -> None:
    mast3r_dir = mast3r_dir.resolve()
    if str(mast3r_dir) not in sys.path:
        sys.path.insert(0, str(mast3r_dir))


def print_gpu_info() -> None:
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    print(f"GPU count       : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} | {props.total_memory / 1e9:.1f} GB")


def ensure_checkpoint(mast3r_dir: Path, checkpoint_override: Path | None = None) -> str:
    if checkpoint_override and checkpoint_override.exists():
        print(f"Using provided checkpoint: {checkpoint_override}")
        return str(checkpoint_override)

    ckpt_path = checkpoint_override or (mast3r_dir / "checkpoints" / CKPT_NAME)
    ckpt_dir = ckpt_path.parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        print(f"Downloading checkpoint to {ckpt_path} ...")
        urllib.request.urlretrieve(CKPT_URL, str(ckpt_path))
    else:
        print(f"Checkpoint already exists: {ckpt_path}")

    print(f"Checkpoint size: {os.path.getsize(str(ckpt_path)) / 1e9:.2f} GB")
    return str(ckpt_path)


def load_model(ckpt_path: str, device: str):
    from mast3r.model import AsymmetricMASt3R

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = AsymmetricMASt3R(
        enc_depth=24,
        dec_depth=12,
        enc_embed_dim=1024,
        dec_embed_dim=768,
        enc_num_heads=16,
        dec_num_heads=12,
        pos_embed="RoPE100",
        img_size=(512, 512),
        head_type="catmlp+dpt",
        output_mode="pts3d+desc24",
        depth_mode=("exp", float("-inf"), float("inf")),
        conf_mode=("exp", 1, float("inf")),
        patch_embed_cls="PatchEmbedDust3R",
        two_confs=True,
        desc_conf_mode=("exp", 0, float("inf")),
    )
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(torch.device(device)).eval()
    print(f"Model loaded on {next(model.parameters()).device}")
    return model


def get_image_paths(scene_dir: Path) -> list[str]:
    img_dir = scene_dir / "images"
    if not img_dir.exists():
        img_dir = scene_dir
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    paths = sorted([p for p in img_dir.iterdir() if p.suffix in exts])
    print(f"  Found {len(paths)} images in {img_dir}")
    return [str(p) for p in paths]


def get_original_size(img_path: str) -> tuple[int, int]:
    with PILImage.open(img_path) as im:
        return im.size


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return np.array([qw, qx, qy, qz])


def write_ply_binary(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    n = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode()
    dt = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    verts = np.empty(n, dtype=dt)
    pts32 = points.astype(np.float32)
    verts["x"] = pts32[:, 0]
    verts["y"] = pts32[:, 1]
    verts["z"] = pts32[:, 2]
    verts["r"] = colors[:, 0]
    verts["g"] = colors[:, 1]
    verts["b"] = colors[:, 2]

    with open(path, "wb") as f:
        f.write(header)
        f.write(verts.tobytes())


def flatten_cloud_blocks(pts3d, imgs, confidence):
    pts_all = []
    cols_all = []
    conf_all = []

    for pt, im, cf in zip(pts3d, imgs, confidence):
        pt_np = np.asarray(pt).reshape(-1, 3)
        im_np = np.asarray(im)
        cf_np = np.asarray(cf).reshape(-1)
        pts_all.append(pt_np)
        cols_all.append((np.clip(im_np, 0, 1).reshape(-1, 3) * 255).astype(np.uint8))
        conf_all.append(cf_np)

    return np.concatenate(pts_all), np.concatenate(cols_all), np.concatenate(conf_all)


def save_pointcloud(pts3d, imgs, confidence, out_dir: Path, conf_percentile: float = 0.0) -> Path:
    pts_all, cols_all, conf_all = flatten_cloud_blocks(pts3d, imgs, confidence)

    if conf_percentile > 0:
        threshold = np.percentile(conf_all, conf_percentile)
        mask = conf_all > threshold
        pts_all = pts_all[mask]
        cols_all = cols_all[mask]
        print(f"  Confidence filter > {conf_percentile}th percentile: kept {mask.sum():,}/{len(mask):,}")
    else:
        print(f"  No confidence filtering: keeping all {len(pts_all):,} points")

    ply_path = out_dir / "pointcloud.ply"
    write_ply_binary(ply_path, pts_all, cols_all)
    print(f"  Saved pointcloud: {ply_path} ({ply_path.stat().st_size / 1e6:.1f} MB)")
    return ply_path


def save_colmap_cameras(
    focals,
    principal_points,
    poses,
    img_paths: list[str],
    orig_w: int,
    orig_h: int,
    mast3r_w: int,
    mast3r_h: int,
    out_dir: Path,
) -> Path:
    sparse_dir = out_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    scale_x = orig_w / mast3r_w
    scale_y = orig_h / mast3r_h
    focal_scale = 0.5 * (scale_x + scale_y)

    with open(sparse_dir / "cameras.txt", "w", encoding="utf-8") as f:
        f.write("# Camera list: CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy\n")
        for i, (focal, pp) in enumerate(zip(np.asarray(focals).reshape(-1), np.asarray(principal_points))):
            focal_orig = float(focal) * focal_scale
            cx = float(pp[0]) * scale_x
            cy = float(pp[1]) * scale_y
            f.write(
                f"{i + 1} PINHOLE {orig_w} {orig_h} "
                f"{focal_orig:.6f} {focal_orig:.6f} {cx:.6f} {cy:.6f}\n"
            )

    with open(sparse_dir / "images.txt", "w", encoding="utf-8") as f:
        f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")
        f.write("# POINTS2D[] as (X Y POINT3D_ID)\n")
        for i, (c2w, img_path) in enumerate(zip(np.asarray(poses), img_paths)):
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            q = rotation_matrix_to_quaternion(R)
            name = os.path.basename(img_path)
            f.write(
                f"{i + 1} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} "
                f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {i + 1} {name}\n"
            )
            f.write("\n")

    with open(sparse_dir / "points3D.txt", "w", encoding="utf-8") as f:
        f.write("# 3D point list - 3DGS uses pointcloud.ply for the dense init\n")

    print(f"  Saved COLMAP text sparse model to {sparse_dir}")
    return sparse_dir


def write_gscpr_manifest(scene_name: str, out_dir: Path, sparse_dir: Path, img_paths: list[str], mast3r_dir: Path) -> Path:
    manifest = {
        "scene_name": scene_name,
        "out_dir": str(out_dir),
        "sparse_dir": str(sparse_dir),
        "images_dir": str(Path(img_paths[0]).parent),
        "image_paths": img_paths,
        "cameras_txt": str(sparse_dir / "cameras.txt"),
        "images_txt": str(sparse_dir / "images.txt"),
        "points3D_txt": str(sparse_dir / "points3D.txt"),
        "pointcloud_ply": str(out_dir / "pointcloud.ply"),
        "mast3r_dir": str(mast3r_dir),
    }
    manifest_path = out_dir / "gscpr_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def run_gscpr_hook(scene_sparse_dir: Path, gscpr_cmd: str | None) -> None:
    if not gscpr_cmd:
        return
    print(f"  Running external GSCPR command: {gscpr_cmd}")
    env = os.environ.copy()
    env["MAST3R_SPARSE_DIR"] = str(scene_sparse_dir)
    subprocess.run(f'{gscpr_cmd} "{scene_sparse_dir}"', shell=True, check=True, env=env)


def run_scene(model, scene_name: str, data_root: Path, out_root: Path, args: argparse.Namespace) -> None:
    from dust3r.utils.device import to_numpy
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment

    print("\n" + "=" * 72)
    print(f"Scene: {scene_name}")
    print("=" * 72)

    scene_dir = data_root / scene_name
    if not scene_dir.exists():
        print(f"  Missing scene directory: {scene_dir}")
        return

    img_paths = get_image_paths(scene_dir)
    if not img_paths:
        print(f"  No images found in {scene_dir}")
        return

    out_dir = out_root / scene_name
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_w, orig_h = get_original_size(img_paths[0])
    print(f"  Original image size: {orig_w}x{orig_h}")

    imgs = load_images(img_paths, size=args.image_size, verbose=True)
    if len(imgs) == 1:
        imgs = [imgs[0], imgs[0].copy()]
        img_paths = [img_paths[0], img_paths[0] + "_dup"]

    mast3r_h, mast3r_w = imgs[0]["img"].shape[-2:]
    print(f"  MASt3R working size: {mast3r_w}x{mast3r_h}")
    print(f"  Scene graph: complete (symmetrized)")
    print(f"  Pair count : {len(imgs) * (len(imgs) - 1)} directed pairs")

    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    scene = sparse_global_alignment(
        img_paths,
        pairs,
        str(cache_dir),
        model,
        subsample=args.subsample,
        device=args.device,
        lr1=args.lr1,
        niter1=args.niter1,
        lr2=args.lr2,
        niter2=args.niter2,
        matching_conf_thr=args.matching_conf_thr,
        shared_intrinsics=args.shared_intrinsics,
        opt_depth=True,
        kinematic_mode="mst",
    )

    focals = to_numpy(scene.get_focals().detach().cpu())
    principal_points = to_numpy(scene.get_principal_points().detach().cpu())
    poses = to_numpy(scene.get_im_poses().detach().cpu())

    if args.tsdf_thresh > 0:
        from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess

        tsdf = TSDFPostProcess(scene, subsample=args.subsample, TSDF_thresh=args.tsdf_thresh)
        pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=args.clean_depth))
    else:
        pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=args.clean_depth, subsample=args.subsample))

    save_pointcloud(pts3d, scene.imgs, confs, out_dir, conf_percentile=args.conf_percentile)
    sparse_dir = save_colmap_cameras(
        focals,
        principal_points,
        poses,
        img_paths,
        orig_w,
        orig_h,
        mast3r_w,
        mast3r_h,
        out_dir,
    )

    manifest_path = write_gscpr_manifest(scene_name, out_dir, sparse_dir, img_paths, args.mast3r_dir)
    print(f"  GSCPR manifest written: {manifest_path}")

    run_gscpr_hook(sparse_dir, args.gscpr_cmd)

    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Scene complete: {scene_name}")


def verify_outputs(scenes: list[str], out_root: Path) -> None:
    print("\nOutput check:")
    for scene in scenes:
        scene_out = out_root / scene
        if not scene_out.exists():
            print(f"  {scene}: MISSING")
            continue
        sparse = scene_out / "sparse" / "0"
        ply = scene_out / "pointcloud.ply"
        print(f"  {scene}/")
        print(f"    pointcloud.ply : {'OK' if ply.exists() else 'MISSING'}")
        for fname in ["cameras.txt", "images.txt", "points3D.txt"]:
            print(f"    sparse/0/{fname}: {'OK' if (sparse / fname).exists() else 'MISSING'}")


def zip_results(scenes: list[str], out_root: Path) -> None:
    zip_path = Path(str(out_root) + ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for scene in scenes:
            scene_out = out_root / scene
            if not scene_out.exists():
                continue
            for root, _, files in os.walk(scene_out):
                for file in files:
                    fpath = Path(root) / file
                    arcname = fpath.relative_to(out_root)
                    zf.write(fpath, arcname.as_posix())
    print(f"\nZipped output: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    args = parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass

    bootstrap_mast3r_paths(args.mast3r_dir)

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    data_root = args.data_root.resolve()
    out_root = args.out_root.resolve()
    mast3r_dir = args.mast3r_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print_gpu_info()
    ckpt_path = ensure_checkpoint(mast3r_dir, args.checkpoint)
    model = load_model(ckpt_path, args.device)

    for scene in scenes:
        try:
            run_scene(model, scene, data_root, out_root, args)
        except Exception as exc:
            print(f"  ERROR on scene {scene}: {exc}")
            raise

    verify_outputs(scenes, out_root)
    if not args.no_zip:
        zip_results(scenes, out_root)


if __name__ == "__main__":
    main()