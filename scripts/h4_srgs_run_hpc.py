#!/usr/bin/env python3
"""
SRGS Training & Rendering — Group A + Group B (Merged, Unoptimized for HPC)
============================================================================

Group A scenes: bike, buddha, firehydrant, toy
  - Uses COLMAP sparse from competition data (cameras.bin, images.bin)
  - Uses MVS .ply point clouds from groupaclouds dataset (<scene>_mvs.ply)
  - NO voxel subsampling, NO confidence pruning — full point clouds used

Group B scenes: aeroplane, cycle, face, still3
  - Uses COLMAP sparse from group-b-dense dataset
  - Uses pointcloud.ply from group-b-dense dataset
  - NO voxel subsampling, NO confidence pruning — full point clouds used

Usage
-----
    python run_srgs.py \
        --srgs_dir      ~/SRGS \
        --competition_data /storage/data-given-3dgs \
        --groupa_clouds    /storage/groupaclouds \
        --groupb_dense     /storage/group-b-dense \
        --work_dir         /storage/srgs_work \
        --scenes bike,buddha,firehydrant,toy,aeroplane,cycle,face,still3
"""

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════
import argparse
import os
import shutil
import struct
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

# ═══════════════════════════════════════════════════════════════════════════════
# TUNABLES — override via CLI args, these are the defaults
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULT_GROUP_A_SCENES = ["bike", "buddha", "firehydrant", "toy"]
DEFAULT_GROUP_B_SCENES = ["aeroplane", "cycle", "face", "still3"]
DEFAULT_ALL_SCENES = DEFAULT_GROUP_A_SCENES + DEFAULT_GROUP_B_SCENES

# SRGS training hyperparameters
TRAIN_ITERATIONS = 30000
TEST_ITERATIONS = [7000, 30000]
SAVE_ITERATIONS = [30000]
DENSIFY_GRAD_THRESHOLD = 0.0002  # slightly higher than default 0.0001 for dense init
RESOLUTION = 1
TRAIN_TIMEOUT_HOURS = 8  # per-scene timeout
RENDER_TIMEOUT_SECONDS = 3600  # per-scene render timeout

# SwinIR-L weights filename
SWINIR_WEIGHTS_NAME = "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
SWINIR_WEIGHTS_URL = f"https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{SWINIR_WEIGHTS_NAME}"

# diff-gaussian-rasterization commit (pre-antialiasing)
RASTERIZER_COMMIT = "9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0"


# ═══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ═══════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SRGS train + render for Group A & B scenes (HPC, unoptimized)."
    )
    # Required paths
    p.add_argument(
        "--srgs_dir",
        type=Path,
        required=True,
        help="Path to the cloned SRGS repository.",
    )
    p.add_argument(
        "--competition_data",
        type=Path,
        required=True,
        help="Path to data-given-3dgs/ (LR images + Group A sparse).",
    )
    p.add_argument(
        "--groupa_clouds",
        type=Path,
        required=True,
        help="Path to groupaclouds/ dataset (contains <scene>_mvs.ply files).",
    )
    p.add_argument(
        "--groupb_dense",
        type=Path,
        required=True,
        help="Path to group-b-dense/ dataset (sparse + pointcloud.ply per scene).",
    )

    # Working / output directory
    p.add_argument(
        "--work_dir",
        type=Path,
        default=Path("srgs_work"),
        help="Working directory for assembled data, models, and results.",
    )

    # Scene selection
    p.add_argument(
        "--scenes",
        type=str,
        default=",".join(DEFAULT_ALL_SCENES),
        help="Comma-separated scene names to process.",
    )

    # Training overrides
    p.add_argument(
        "--iterations",
        type=int,
        default=TRAIN_ITERATIONS,
        help="SRGS training iterations (default: 30000).",
    )
    p.add_argument(
        "--densify_grad_threshold",
        type=float,
        default=DENSIFY_GRAD_THRESHOLD,
        help="Densification gradient threshold (default: 0.0002).",
    )
    p.add_argument(
        "--target_points",
        type=int,
        default=0,
        help="Target number of points for voxel subsampling (default: 0 = no subsampling. e.g., 75000).",
    )

    # Control flags
    p.add_argument(
        "--skip_setup",
        action="store_true",
        help="Skip data assembly (assume work_dir/data/ already prepared).",
    )
    p.add_argument(
        "--skip_patch",
        action="store_true",
        help="Skip SRGS source patching (assume already patched).",
    )
    p.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip training (go straight to rendering).",
    )
    p.add_argument("--skip_render", action="store_true", help="Skip rendering.")
    p.add_argument(
        "--no_zip", action="store_true", help="Skip zipping the final submission."
    )
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def write_points3d_bin(positions: np.ndarray, colors: np.ndarray, bin_path):
    """Write a COLMAP-compatible points3D.bin from positions (N,3) and colors (N,3)."""
    n = len(positions)
    with open(bin_path, "wb") as f:
        f.write(struct.pack("<Q", n))
        for i in range(n):
            f.write(struct.pack("<Q", i + 1))
            f.write(struct.pack("<ddd", *positions[i].tolist()))
            f.write(struct.pack("<BBB", *colors[i].tolist()))
            f.write(struct.pack("<d", 0.0))  # error
            f.write(struct.pack("<Q", 0))  # track length


def save_ply(positions: np.ndarray, colors: np.ndarray, ply_path: Path):
    from plyfile import PlyData, PlyElement

    n = len(positions)
    dt = np.dtype(
        [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    verts = np.empty(n, dtype=dt)
    verts["x"] = positions[:, 0]
    verts["y"] = positions[:, 1]
    verts["z"] = positions[:, 2]
    verts["nx"] = 0.0
    verts["ny"] = 0.0
    verts["nz"] = 0.0
    verts["red"] = colors[:, 0]
    verts["green"] = colors[:, 1]
    verts["blue"] = colors[:, 2]

    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(str(ply_path))


def voxel_subsample_target(xyz, target_n, seed=42):
    """
    User's Kaggle binary search voxel subsampler to hit a specific point count.
    """
    lo = np.linalg.norm(xyz.max(0) - xyz.min(0)) / (target_n * 10)
    hi = np.linalg.norm(xyz.max(0) - xyz.min(0))
    best = None
    for _ in range(25):
        mid = (lo + hi) / 2.0
        vc = np.floor(xyz / mid).astype(np.int32)
        vm = {}
        for i, v in enumerate(vc):
            k = (v[0], v[1], v[2])
            if k not in vm:
                vm[k] = i
        idx = np.array(list(vm.values()), dtype=np.int64)
        best = idx
        if abs(len(idx) - target_n) < target_n * 0.03:
            break
        if len(idx) > target_n:
            lo = mid
        else:
            hi = mid
    if len(best) > target_n:
        rng = np.random.default_rng(seed)
        best = best[rng.choice(len(best), target_n, replace=False)]
    return best


def is_group_a(scene: str) -> bool:
    return scene in DEFAULT_GROUP_A_SCENES


def is_group_b(scene: str) -> bool:
    return scene in DEFAULT_GROUP_B_SCENES


# ═══════════════════════════════════════════════════════════════════════════════
# DATA ASSEMBLY — no subsampling, no confidence pruning
# ═══════════════════════════════════════════════════════════════════════════════
def setup_data(scenes, args):
    """
    Assemble the data directory for each scene.
    Unoptimized: full point clouds, no voxel subsampling, no confidence pruning.
    """
    from plyfile import PlyData

    data_base = args.work_dir / "data"
    data_base.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Data Assembly (unoptimized — full point clouds)")
    print("=" * 60)

    for scene in scenes:
        print(f"\n  [{scene}]")
        dst = data_base / scene
        src_comp = args.competition_data / scene

        if not src_comp.exists():
            print(f"    ✗ Competition data not found: {src_comp}")
            continue

        # Clean destination
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)

        # Copy LR images (common to both groups)
        shutil.copytree(src_comp / "images", dst / "images")
        print("    ✓ Images copied")

        if is_group_a(scene):
            # Group A: sparse from competition data, PLY from groupaclouds
            shutil.copytree(src_comp / "sparse", dst / "sparse")
            print("    ✓ COLMAP sparse copied (from competition data)")

            src_ply = args.groupa_clouds / f"{scene}_mvs.ply"
            if not src_ply.exists():
                print(f"    ✗ MVS PLY not found: {src_ply}")
                continue

            dst_dir = dst / "sparse" / "0"
            dst_dir.mkdir(parents=True, exist_ok=True)

            # Read the PLY
            plydata = PlyData.read(str(src_ply))
            v = plydata["vertex"]
            n = len(v)
            pos = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float64)
            col = np.vstack([v["red"], v["green"], v["blue"]]).T.astype(np.uint8)

            if args.target_points > 0 and n > args.target_points:
                li = voxel_subsample_target(pos, args.target_points)
                pos = pos[li]
                col = col[li]
                n_down = len(pos)
                print(
                    f"    PLY: {n:,} points — downsampled to {n_down:,} (target_points={args.target_points})"
                )
                save_ply(pos, col, dst_dir / "points3D.ply")
                write_points3d_bin(pos, col, dst_dir / "points3D.bin")
                print(f"    ✓ PLY + BIN written ({n_down:,} pts)")
            else:
                print(f"    PLY: {n:,} points — using ALL (no subsampling)")
                # Copy PLY as-is
                shutil.copy2(src_ply, dst_dir / "points3D.ply")
                # Write matching .bin
                write_points3d_bin(pos, col, dst_dir / "points3D.bin")
                print(f"    ✓ PLY + BIN written ({n:,} pts)")

        elif is_group_b(scene):
            # Group B: sparse + pointcloud from group-b-dense
            src_b = args.groupb_dense / scene
            if not src_b.exists():
                print(f"    ✗ Group B dense data not found: {src_b}")
                continue

            shutil.copytree(src_b / "sparse", dst / "sparse")
            print("    ✓ COLMAP sparse copied (from group-b-dense)")

            src_ply = src_b / "pointcloud.ply"
            if not src_ply.exists():
                print(f"    ✗ pointcloud.ply not found: {src_ply}")
                continue

            dst_dir = dst / "sparse" / "0"
            dst_dir.mkdir(parents=True, exist_ok=True)

            # Read PLY
            plydata = PlyData.read(str(src_ply))
            v = plydata["vertex"]
            n = len(v)
            fields = v.data.dtype.names
            pos = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float64)
            col = np.vstack([v["red"], v["green"], v["blue"]]).T.astype(np.uint8)

            if args.target_points > 0 and n > args.target_points:
                li = voxel_subsample_target(pos, args.target_points)
                pos = pos[li]
                col = col[li]
                n_down = len(pos)
                print(f"    PLY: {n:,} points | fields: {fields}")
                print(
                    f"    Downsampled to {n_down:,} (target_points={args.target_points})"
                )
                save_ply(pos, col, dst_dir / "points3D.ply")
                write_points3d_bin(pos, col, dst_dir / "points3D.bin")
                print(f"    ✓ PLY + BIN written ({n_down:,} pts)")
            else:
                print(f"    PLY: {n:,} points | fields: {fields}")
                print(
                    "    Using ALL points (no confidence pruning, no voxel subsampling)"
                )

                # Write PLY (binary, preserving all fields)
                from plyfile import PlyElement

                PlyData([PlyElement.describe(v.data, "vertex")], text=False).write(
                    str(dst_dir / "points3D.ply")
                )

                # Write matching .bin
                write_points3d_bin(pos, col, dst_dir / "points3D.bin")
                print(f"    ✓ PLY + BIN written ({n:,} pts)")

        else:
            print(f"    ✗ Unknown group for scene '{scene}' — skipping")

    print("\n  Data assembly complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# SRGS PATCHING — 3 patches + patch 2b
# ═══════════════════════════════════════════════════════════════════════════════
def patch_srgs(srgs_root: Path):
    """
    Apply 4 patches to the SRGS source code:
      1. fetchPly normals fallback (MVS/MASt3R PLY may lack nx/ny/nz)
      2. Rasterizer return unpacking (pre-AA commit returns 2-tuple)
      2b. antialiasing=False argument added to renderer call
      3. SwinIR-L config (large model with correct weights path)
    """
    S = str(srgs_root)
    print("\n" + "=" * 60)
    print("Patching SRGS source")
    print("=" * 60)

    # ── Patch 1: fetchPly normals fallback ─────────────────────────────────
    dr = os.path.join(S, "scene", "dataset_readers.py")
    with open(dr) as f:
        c = f.read()
    old = (
        "def fetchPly(path):\n"
        "    plydata = PlyData.read(path)\n"
        "    vertices = plydata['vertex']\n"
        "    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T\n"
        "    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0\n"
        "    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T\n"
        "    return BasicPointCloud(points=positions, colors=colors, normals=normals)"
    )
    new = (
        "def fetchPly(path):\n"
        "    plydata = PlyData.read(path)\n"
        "    vertices = plydata['vertex']\n"
        "    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T\n"
        "    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0\n"
        "    try:\n"
        "        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T\n"
        "    except ValueError:\n"
        "        normals = np.zeros_like(positions)\n"
        "    return BasicPointCloud(points=positions, colors=colors, normals=normals)"
    )
    if old in c:
        c = c.replace(old, new)
        with open(dr, "w") as f:
            f.write(c)
        print("  ✓ Patch 1: fetchPly normals fallback")
    elif "except ValueError" in c:
        print("  ✓ Patch 1: already applied")
    else:
        print("  ✗ Patch 1: pattern not found — MANUAL CHECK NEEDED")

    # ── Patch 2: rasterizer return unpacking ───────────────────────────────
    gr = os.path.join(S, "gaussian_renderer", "__init__.py")
    with open(gr) as f:
        c = f.read()
    if "    rendered_image, radii = rasterizer(" in c:
        c = c.replace(
            "    rendered_image, radii = rasterizer(",
            "    rendered_image, radii, *_ = rasterizer(",
        )
        with open(gr, "w") as f:
            f.write(c)
        print("  ✓ Patch 2: rasterizer return unpacking")
    elif "*_" in c:
        print("  ✓ Patch 2: already applied")
    else:
        print("  ✗ Patch 2: pattern not found — MANUAL CHECK NEEDED")

    # ── Patch 2b: antialiasing=False in renderer ───────────────────────────
    with open(gr) as f:
        c = f.read()
    old_2b = "        prefiltered=False,\n        debug=pipe.debug\n    )"
    new_2b = "        prefiltered=False,\n        debug=pipe.debug,\n        antialiasing=False\n    )"
    if old_2b in c:
        c = c.replace(old_2b, new_2b)
        with open(gr, "w") as f:
            f.write(c)
        print("  ✓ Patch 2b: antialiasing=False added to renderer")
    elif "antialiasing=False" in c:
        print("  ✓ Patch 2b: already applied")
    else:
        print("  ✗ Patch 2b: pattern not found — MANUAL CHECK NEEDED")

    # ── Patch 3: SwinIR-L config ───────────────────────────────────────────
    cam = os.path.join(S, "scene", "cameras.py")
    with open(cam) as f:
        c = f.read()
    old3 = (
        "self.SR_model = net(upscale=4, in_chans=3, img_size=64, window_size=8,\n"
        "                       img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],\n"
        "                       mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')\n"
        "        param_key_g = 'params'\n"
        '        pretrained_model = torch.load("./model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth")'
    )
    new3 = (
        "self.SR_model = net(upscale=4, in_chans=3, img_size=64, window_size=8,\n"
        "                       img_range=1., depths=[6,6,6,6,6,6,6,6,6], embed_dim=240, num_heads=[8,8,8,8,8,8,8,8,8],\n"
        "                       mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')\n"
        "        param_key_g = 'params_ema'\n"
        '        pretrained_model = torch.load("./model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth")'
    )
    if old3 in c:
        c = c.replace(old3, new3)
        with open(cam, "w") as f:
            f.write(c)
        print("  ✓ Patch 3: SwinIR-L config")
    elif "003_realSR_BSRGAN" in c:
        print("  ✓ Patch 3: already applied")
    else:
        print("  ✗ Patch 3: pattern not found — MANUAL CHECK NEEDED")

    print("  Patching complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
def train_scenes(scenes, args):
    data_base = args.work_dir / "data"
    output_base = args.work_dir / "output"
    output_base.mkdir(parents=True, exist_ok=True)

    training_results = {}
    start_total = time.time()
    print("\n" + "=" * 60)
    print("SRGS Training (unoptimized — full point clouds)")
    print("=" * 60)

    for idx, scene in enumerate(scenes, 1):
        print(f"\n  [{idx}/{len(scenes)}] {scene.upper()}")
        src = data_base / scene
        out = output_base / scene

        has_cams = (src / "sparse" / "0" / "cameras.bin").exists()
        has_ply = (src / "sparse" / "0" / "points3D.ply").exists()
        print(f"    cameras={'✓' if has_cams else '✗'}  PLY={'✓' if has_ply else '✗'}")

        if not has_cams or not has_ply:
            training_results[scene] = {"status": "skipped"}
            print("    ✗ SKIP: missing prerequisites")
            continue

        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            "python",
            str(args.srgs_dir / "train.py"),
            "-s",
            str(src),
            "-m",
            str(out),
            "--images",
            "images",
            "--resolution",
            str(RESOLUTION),
            "--iterations",
            str(args.iterations),
            "--test_iterations",
            *[str(i) for i in TEST_ITERATIONS],
            "--save_iterations",
            *[str(i) for i in SAVE_ITERATIONS],
            "--densify_grad_threshold",
            str(args.densify_grad_threshold),
        ]
        print(f"    Command: {' '.join(cmd)}")

        # Set env var to disable TF32 on A100 to prevent PSNR loss
        env = os.environ.copy()
        env["NVIDIA_TF32_OVERRIDE"] = "0"

        t0 = time.time()
        try:
            res = subprocess.run(
                cmd, cwd=str(args.srgs_dir), timeout=TRAIN_TIMEOUT_HOURS * 3600, env=env
            )
            elapsed = time.time() - t0
            ok = res.returncode == 0
            status = "success" if ok else "failed"
            print(f"    {'✓' if ok else '✗'} {status} in {elapsed / 60:.1f} min")
            training_results[scene] = {"status": status, "elapsed": elapsed}
        except subprocess.TimeoutExpired:
            print(f"    ✗ Timeout after {TRAIN_TIMEOUT_HOURS}h")
            training_results[scene] = {"status": "timeout"}

    print("\n" + "=" * 60)
    print("Training Summary:")
    for scene, r in training_results.items():
        s = r["status"]
        t = f"{r['elapsed'] / 60:.1f} min" if s == "success" else s
        print(f"  {'✓' if s == 'success' else '✗'} {scene:<15} {t}")
    print(f"Total: {(time.time() - start_total) / 3600:.1f}h")
    return training_results


# ═══════════════════════════════════════════════════════════════════════════════
# RENDERING
# ═══════════════════════════════════════════════════════════════════════════════
def render_scenes(scenes, args):
    data_base = args.work_dir / "data"
    output_base = args.work_dir / "output"
    results_base = args.work_dir / "results"
    results_base.mkdir(parents=True, exist_ok=True)

    rendering_results = {}
    print("\n" + "=" * 60)
    print("Rendering Novel Views")
    print("=" * 60)

    for idx, scene in enumerate(scenes, 1):
        print(f"\n  [{idx}/{len(scenes)}] {scene.upper()}")
        model_path = output_base / scene
        result_path = results_base / scene

        if not (model_path / "point_cloud").exists():
            print(f"    ✗ SKIP: no trained model at {model_path}")
            rendering_results[scene] = {"status": "skipped"}
            continue

        result_path.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python",
            str(args.srgs_dir / "render.py"),
            "-s",
            str(data_base / scene),
            "-m",
            str(model_path),
            "--skip_test",
        ]

        t0 = time.time()
        try:
            res = subprocess.run(
                cmd,
                cwd=str(args.srgs_dir),
                capture_output=True,
                text=True,
                timeout=RENDER_TIMEOUT_SECONDS,
            )
            elapsed = time.time() - t0

            # Find renders (no --eval → all images in train set)
            renders_src = None
            train_dir = model_path / "train"
            if train_dir.exists():
                ours_dirs = sorted(train_dir.glob("ours_*"))
                if ours_dirs:
                    renders_src = ours_dirs[-1] / "renders"

            if res.returncode == 0 and renders_src and renders_src.exists():
                shutil.copytree(
                    renders_src, result_path / "renders", dirs_exist_ok=True
                )
                n_renders = len(list((result_path / "renders").glob("*.png")))
                print(f"    ✓ {n_renders} images rendered in {elapsed / 60:.1f} min")
                rendering_results[scene] = {"status": "success", "n_renders": n_renders}
            else:
                print(f"    ✗ Failed (code {res.returncode})")
                if res.stderr:
                    print(f"    STDERR: {res.stderr[-600:]}")
                rendering_results[scene] = {"status": "failed"}
        except subprocess.TimeoutExpired:
            rendering_results[scene] = {"status": "timeout"}
            print("    ✗ Timeout")

    print("\n" + "=" * 60)
    print("Rendering Summary:")
    for scene, r in rendering_results.items():
        s = r.get("status")
        print(f"  {'✓' if s == 'success' else '✗'} {scene:<15} {s}")
    return rendering_results


# ═══════════════════════════════════════════════════════════════════════════════
# ZIP SUBMISSION
# ═══════════════════════════════════════════════════════════════════════════════
def zip_submission(scenes, args):
    results_base = args.work_dir / "results"
    zip_path = args.work_dir / "submission.zip"
    print(f"\nCreating {zip_path} ...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for scene in scenes:
            rd = results_base / scene / "renders"
            if not rd.exists():
                print(f"  ✗ {scene}: renders not found")
                continue
            pngs = sorted(rd.glob("*.png"))
            for p in pngs:
                zf.write(p, f"{scene}/{p.name}")
            print(f"  ✓ {scene}: {len(pngs)} renders added")
    size_mb = zip_path.stat().st_size / 1e6
    print(f"\nSubmission: {zip_path} ({size_mb:.1f} MB)")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]

    print("=" * 60)
    print("SRGS Pipeline — Group A + B (Unoptimized for HPC)")
    print("=" * 60)
    print(f"  SRGS dir        : {args.srgs_dir.resolve()}")
    print(f"  Competition data: {args.competition_data.resolve()}")
    print(f"  Group A clouds  : {args.groupa_clouds.resolve()}")
    print(f"  Group B dense   : {args.groupb_dense.resolve()}")
    print(f"  Work dir        : {args.work_dir.resolve()}")
    print(f"  Scenes          : {scenes}")
    print(f"  Iterations      : {args.iterations}")
    print(f"  Densify thresh  : {args.densify_grad_threshold}")
    print(
        f"  GPU             : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
    print()

    group_a = [s for s in scenes if is_group_a(s)]
    group_b = [s for s in scenes if is_group_b(s)]
    unknown = [s for s in scenes if not is_group_a(s) and not is_group_b(s)]
    if unknown:
        print(f"  WARNING: Unknown scenes will be skipped: {unknown}")
    print(f"  Group A scenes  : {group_a}")
    print(f"  Group B scenes  : {group_b}")

    # Step 1: Data assembly
    if not args.skip_setup:
        setup_data(scenes, args)
    else:
        print("\n  Skipping data setup (--skip_setup)")

    # Step 2: Patch SRGS source
    if not args.skip_patch:
        patch_srgs(args.srgs_dir)
    else:
        print("\n  Skipping SRGS patching (--skip_patch)")

    # Step 3: Train
    if not args.skip_train:
        train_scenes(scenes, args)
    else:
        print("\n  Skipping training (--skip_train)")

    # Step 4: Render
    if not args.skip_render:
        render_scenes(scenes, args)
    else:
        print("\n  Skipping rendering (--skip_render)")

    # Step 5: Zip
    if not args.no_zip:
        zip_submission(scenes, args)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()