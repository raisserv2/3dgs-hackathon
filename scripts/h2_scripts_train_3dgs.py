import os
import shutil
import subprocess
import argparse
import sys
from pathlib import Path


def ensure_colmap_layout(src: Path):
    """Ensure gaussian-splatting can find COLMAP model under sparse/0."""
    sparse0 = src / "sparse" / "0"
    required = ["images.bin", "cameras.bin", "points3D.bin"]

    if all((sparse0 / name).exists() for name in required):
        return True

    sfm = src / "sfm"
    if not sfm.exists():
        return False

    # Prefer undistorting from sfm model so camera type becomes PINHOLE/SIMPLE_PINHOLE.
    colmap_exe = shutil.which("colmap")
    if colmap_exe is not None:
        image_path = src / "images"
        if image_path.exists():
            undist_cmd = [
                colmap_exe,
                "image_undistorter",
                "--image_path", str(image_path),
                "--input_path", str(sfm),
                "--output_path", str(src),
                "--output_type", "COLMAP",
            ]
            subprocess.run(undist_cmd, check=True)

            # COLMAP may place model files under sparse/ instead of sparse/0.
            sparse_root = src / "sparse"
            sparse0.mkdir(parents=True, exist_ok=True)
            for name in required:
                root_file = sparse_root / name
                dst_file = sparse0 / name
                if root_file.exists() and not dst_file.exists():
                    shutil.move(str(root_file), str(dst_file))

            if all((sparse0 / name).exists() for name in required):
                return True

    sparse0.mkdir(parents=True, exist_ok=True)
    # Fallback for environments without COLMAP: copy raw model files.
    for name in required:
        src_file = sfm / name
        dst_file = sparse0 / name
        if not src_file.exists():
            return False
        if not dst_file.exists():
            shutil.copy2(src_file, dst_file)

    return True


def train_scene(scene, resolution="1"):
    src = Path(f"data_processed/{scene}")
    out = Path(f"output/{scene}")
    out.mkdir(parents=True, exist_ok=True)
    if not ensure_colmap_layout(src):
        print(f"Skipping {scene}: missing prepared COLMAP data in {src}")
        return False
    
    # Check sparse point count to estimate densification aggressiveness.
    num_points = -1
    try:
        gs_root = Path("gaussian-splatting").resolve()
        if str(gs_root) not in sys.path:
            sys.path.append(str(gs_root))
        from scene.colmap_loader import read_points3D_binary
        pts = read_points3D_binary(str(src / "sparse" / "0" / "points3D.bin"))
        num_points = len(pts[0])
    except Exception:
        num_points = -1

    # For scenes with many sparse points, use memory-safe settings.
    local_resolution = resolution
    densify_until = 15000
    use_data_device_cpu = False
    if num_points > 10000:
        densify_until = 6000
        use_data_device_cpu = True
        if resolution == "1":
            local_resolution = "2"
            print(f"{scene}: high sparse point count ({num_points}), using --resolution 2 and reduced densification.")
    elif num_points > 7000:
        densify_until = 10000
        use_data_device_cpu = True

    if num_points < 0:
        densify_until = 15000
    
    cmd = [
        "python", "gaussian-splatting/train.py",
        "--source_path", str(src),
        "--model_path", str(out),
        "--iterations", "30000",
        "--eval",
        "--resolution", local_resolution,
        "--densify_until_iter", str(densify_until)
    ]

    if use_data_device_cpu:
        cmd += ["--data_device", "cpu"]
    
    # Avoid SH gradient bug on Windows (use degree 0)
    if os.name == "nt":
        cmd += ["--sh_degree", "0"]
    
    print("Launching:", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            if resolution == "1":
                print(f"Out of memory on {scene} at full resolution. Retrying at 1/2 resolution...")
                return train_scene(scene, resolution="2")
            else:
                print(f"Out of memory on {scene} even at reduced resolution. Skipping.")
                return False
        else:
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_scene",
        default="aeroplane",
        choices=["aeroplane", "bike", "buddha", "cycle", "face", "firehydrant", "still3", "toy"],
        help="First scene to train in the batch.",
    )
    args = parser.parse_args()

    scenes = ["aeroplane", "bike", "buddha", "cycle", "face", "firehydrant", "still3", "toy"]
    scenes = scenes[scenes.index(args.start_scene):]
    completed = 0
    skipped = 0
    for scene in scenes:
        if train_scene(scene):
            completed += 1
        else:
            skipped += 1

    print(f"Finished: {completed} scenes trained, {skipped} scenes skipped")