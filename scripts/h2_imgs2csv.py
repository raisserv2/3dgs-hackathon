import os
import cv2
import pandas as pd


PRED_ROOT = "sub" # Modify this
OUTPUT = "submissionpeek.csv"


EXPECTED_SCENES = [
    "aeroplane",
    "bike",
    "buddha",
    "cycle",
    "face",
    "firehydrant",
    "still3",
    "toy"
]


def image_to_string(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return " ".join(map(str, img.flatten()))

# -------------------------------
# Validate scenes
# -------------------------------
found_scenes = [
    d for d in os.listdir(PRED_ROOT)
    if os.path.isdir(os.path.join(PRED_ROOT, d))
]

missing = set(EXPECTED_SCENES) - set(found_scenes)
extra = set(found_scenes) - set(EXPECTED_SCENES)

if missing:
    raise ValueError(
        f"Missing scenes: {sorted(missing)}\n"
        f"You must submit ALL scenes: {EXPECTED_SCENES}"
    )

if extra:
    raise ValueError(
        f" Unexpected scenes found: {sorted(extra)}\n"
        f"Only these are allowed: {EXPECTED_SCENES}"
    )

print("Scene check passed")

# -------------------------------
# Process images
# -------------------------------
rows = []

for scene in sorted(EXPECTED_SCENES):
    scene_path = os.path.join(PRED_ROOT, scene)

    image_files = sorted([
        f for f in os.listdir(scene_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    if len(image_files) == 0:
        raise ValueError(f"No images found in scene: {scene}")

    for fname in image_files:
        img_path = os.path.join(scene_path, fname)
        img = cv2.imread(img_path)

        if img is None:
            raise ValueError(f"Failed to read: {img_path}")

        img_str = image_to_string(img)

        base = os.path.splitext(fname)[0]

        rows.append({
            "id": f"{scene}_{base}",
            "scene": scene,
            "image_name": base,
            "image": img_str
        })

print(f" Processed images: {len(rows)}")

# -------------------------------
# Create DataFrame
# -------------------------------
df = pd.DataFrame(rows)

# Ensure consistent ordering
df = df.sort_values(by=["scene", "image_name"]).reset_index(drop=True)

# -------------------------------
# Save CSV
# -------------------------------
df.to_csv(OUTPUT, index=False)

print(f"submission.csv ready: {OUTPUT} ({len(df)} rows)")