from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
KAGGLE_OUTPUT_ROOT = ROOT / "kaggle_output"
DATA_KAGGLE_ROOT = ROOT / "data_kaggle"
DATA_PROCESSED_ROOT = ROOT / "data_processed"
RENDER_SCRIPT = ROOT / "gaussian-splatting" / "render.py"


SCENE_MODEL_PATHS = {
    "aeroplane": KAGGLE_OUTPUT_ROOT / "aeroplane_buddha" / "runs_s2" / "aeroplane",
    "buddha": KAGGLE_OUTPUT_ROOT / "aeroplane_buddha" / "runs_s2" / "buddha",
    "bike": KAGGLE_OUTPUT_ROOT / "bike_cycle_face_toy" / "runs_s2" / "bike",
    "cycle": KAGGLE_OUTPUT_ROOT / "bike_cycle_face_toy" / "runs_s2" / "cycle",
    "face": KAGGLE_OUTPUT_ROOT / "bike_cycle_face_toy" / "runs_s2" / "face",
    "toy": KAGGLE_OUTPUT_ROOT / "bike_cycle_face_toy" / "runs_s2" / "toy",
    "firehydrant": KAGGLE_OUTPUT_ROOT / "firehydrant" / "runs_s2" / "firehydrant",
    "still3": KAGGLE_OUTPUT_ROOT / "still3" / "runs_s2" / "still3",
}

SCENE_SOURCE_ROOTS = {
    "aeroplane": DATA_PROCESSED_ROOT,
    "cycle": DATA_PROCESSED_ROOT,
    "face": DATA_PROCESSED_ROOT,
    "still3": DATA_PROCESSED_ROOT,
    "toy": DATA_PROCESSED_ROOT,
    "firehydrant": DATA_PROCESSED_ROOT,
    "buddha": DATA_KAGGLE_ROOT,
    "bike": DATA_KAGGLE_ROOT,
}


def render_scene(scene: str, model_path: Path) -> None:
    if not model_path.exists():
        print(f"{scene}: missing model path, skipping: {model_path}")
        return

    if not (model_path / "point_cloud").exists():
        print(f"{scene}: no trained model found under {model_path}, skipping")
        return

    render_dir = model_path / "test" / "ours_30000" / "renders"
    if render_dir.exists() and any(render_dir.glob("*.png")):
        print(f"{scene}: renders already exist at {render_dir}, skipping")
        return

    source_root = SCENE_SOURCE_ROOTS[scene]
    source_path = source_root / scene
    if not source_path.exists():
        raise FileNotFoundError(
            f"Source scene folder not found for {scene}: {source_path}. "
            "Create the matching source folder first or update SCENE_SOURCE_ROOTS."
        )

    cfg_args_path = model_path / "cfg_args"
    cfg_args_path.write_text(
        "Namespace("
        "sh_degree=3, "
        f"source_path={str(source_path)!r}, "
        f"model_path={str(model_path)!r}, "
        "images='images', "
        "depths='', "
        "resolution=-1, "
        "white_background=False, "
        "train_test_exp=False, "
        "data_device='cuda', "
        "eval=True"
        ")\n"
    )
    print(f"{scene}: wrote cfg_args at {cfg_args_path}")

    if not RENDER_SCRIPT.exists():
        raise FileNotFoundError(f"Render script not found: {RENDER_SCRIPT}")

    print(f"Rendering {scene} from {model_path}...")
    cmd = [
        sys.executable,
        str(RENDER_SCRIPT),
        "--model_path",
        str(model_path),
        "--skip_train",
        "--quiet",
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    print(f"Finished {scene}")


def main() -> None:
    if not KAGGLE_OUTPUT_ROOT.exists():
        raise FileNotFoundError(f"kaggle_output folder not found: {KAGGLE_OUTPUT_ROOT}")

    for scene, model_path in SCENE_MODEL_PATHS.items():
        render_scene(scene, model_path)


if __name__ == "__main__":
    main()