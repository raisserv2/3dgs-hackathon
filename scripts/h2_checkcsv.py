import pandas as pd
import json

# Load expected test filenames
with open("test_filenames.json", "r") as f:
    expected = json.load(f)

# Read the CSV
df = pd.read_csv("submission.csv")

print("CSV columns:", list(df.columns))
print("Total rows:", len(df))

# Check if columns are correct
required_cols = ["id", "scene", "image_name", "image"]
if list(df.columns) != required_cols:
    print(f"❌ Wrong columns: {list(df.columns)}")
else:
    print("✅ Columns correct")

# Check number of rows per scene
row_count = df.groupby("scene").size().to_dict()
for scene in expected:
    expected_count = len(expected[scene])
    actual_count = row_count.get(scene, 0)
    if actual_count != expected_count:
        print(f"❌ {scene}: expected {expected_count} rows, got {actual_count}")
    else:
        print(f"✅ {scene}: {actual_count} rows")

# Check a sample id format
sample_id = df.iloc[0]["id"]
print(f"\nSample id: {sample_id}")
print("(Should look like: scene_imagefilename_without_extension)")

# Check if image column has non-empty strings
sample_img = df.iloc[0]["image"]
print(f"Sample image data length: {len(sample_img.split())} numbers")