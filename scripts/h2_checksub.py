import json
from pathlib import Path

SUBMISSION_ROOT = Path("submission")

# Load the test filenames you saved earlier
with open("test_filenames.json", "r") as f:
    expected = json.load(f)

all_ok = True
for scene, expected_files in expected.items():
    scene_dir = SUBMISSION_ROOT / scene
    if not scene_dir.exists():
        print(f"❌ {scene}: folder missing")
        all_ok = False
        continue
    actual_files = sorted([f.name for f in scene_dir.glob("*") if f.is_file()])
    if actual_files != expected_files:
        print(f"❌ {scene}: file mismatch")
        print(f"   Expected: {expected_files}")
        print(f"   Actual:   {actual_files}")
        all_ok = False
    else:
        print(f"✅ {scene}: {len(actual_files)} files OK")

if all_ok:
    print("\n✅ Submission folder is correctly structured.")
else:
    print("\n❌ Fix the mismatches above before proceeding.")