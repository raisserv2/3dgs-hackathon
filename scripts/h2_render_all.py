import subprocess
from pathlib import Path

OUTPUT_ROOT = "output"
scenes = ["aeroplane", "bike", "buddha", "cycle", "face", "firehydrant", "still3", "toy"]

for scene in scenes:
    model_path = Path(OUTPUT_ROOT) / scene
    if not (model_path / "point_cloud").exists():
        print(f"{scene}: no trained model, skipping")
        continue
    
    # Check if renders already exist
    render_dir = model_path / "test" / "ours_30000" / "renders"
    if render_dir.exists() and len(list(render_dir.glob("*.png"))) > 0:
        print(f"{scene}: renders already exist, skipping")
        continue
    
    print(f"Rendering {scene}...")
    cmd = [
        "python", "gaussian-splatting/render.py",
        "--model_path", str(model_path),
        "--skip_train",   # only render test views
        "--quiet"
    ]
    subprocess.run(cmd)
    print(f"Finished {scene}")