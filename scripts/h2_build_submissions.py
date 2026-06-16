import json
import shutil
import cv2
import numpy as np
from pathlib import Path

OUTPUT_ROOT = Path("output")
SUBMISSION_ROOT = Path("submission")
ITER_NUM = 30000

with open("test_filenames.json", "r") as f:
    mapping = json.load(f)

shutil.rmtree(SUBMISSION_ROOT, ignore_errors=True)
SUBMISSION_ROOT.mkdir()

for scene, test_files in mapping.items():
    dst_scene = SUBMISSION_ROOT / scene
    dst_scene.mkdir(parents=True, exist_ok=True)
    
    render_dir = OUTPUT_ROOT / scene / "test" / f"ours_{ITER_NUM}" / "renders"
    if render_dir.exists():
        # Get all render files (they are named like 00000.png, 00001.png, ...)
        render_files = sorted(render_dir.glob("*.png"))
        if len(render_files) >= len(test_files):
            for idx, orig_name in enumerate(test_files):
                # Render files are zero-padded to 5 digits (e.g., 00000.png)
                src = render_dir / f"{idx:05d}.png"
                if src.exists():
                    shutil.copy2(src, dst_scene / orig_name)
                    print(f"Copied {scene}/{orig_name}")
                else:
                    # Fallback: try any pattern? But should not happen
                    dummy = np.zeros((224,224,3), dtype=np.uint8)
                    cv2.imwrite(str(dst_scene / orig_name), dummy)
                    print(f"Missing {src}, using dummy for {orig_name}")
        else:
            print(f"Not enough renders for {scene} (need {len(test_files)}, have {len(render_files)})")
            for orig_name in test_files:
                dummy = np.zeros((224,224,3), dtype=np.uint8)
                cv2.imwrite(str(dst_scene / orig_name), dummy)
    else:
        print(f"Render dir not found: {render_dir}")
        for orig_name in test_files:
            dummy = np.zeros((224,224,3), dtype=np.uint8)
            cv2.imwrite(str(dst_scene / orig_name), dummy)

print("Submission folder ready. Now run: python imgs2csv.py")