import subprocess
from pathlib import Path

def render_scene(scene):
    model_path = Path(f"output/{scene}")
    render_path = model_path / "test" / "ours_30000" / "renders"
    if render_path.exists():
        print(f"Renders already exist for {scene}, skipping.")
        return
    cmd = [
        "python", "gaussian-splatting/render.py",
        "--model_path", str(model_path),
        "--skip_train",   # only render test views
        "--quiet"
    ]
    subprocess.run(cmd, check=True)

def copy_to_submission(scene):
    src = Path(f"output/{scene}/test/ours_30000/renders")
    dst = Path(f"submission/{scene}")
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*"):
        # Keep the original file names (e.g. "000.png")
        f.rename(dst / f.name)
        print(f"Copied {f.name} to {dst}")

if __name__ == "__main__":
    scenes = ["aeroplane", "bike", "buddha", "cycle", "face", "firehydrant", "still3", "toy"]
    for scene in scenes:
        render_scene(scene)
        copy_to_submission(scene)