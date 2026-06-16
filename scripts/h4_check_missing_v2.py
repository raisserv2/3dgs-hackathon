import os, sys, glob

if len(sys.argv) != 3:
    print("Usage: python check_missing_v2.py <scene_name> <dataset_root>")
    print("Example: python check_missing_v2.py aeroplane C:\\main\\IITM\\SEM_4\\Modern Computer Vision\\hackathon_final\\data-given-3dgs")
    sys.exit(1)

scene = sys.argv[1]
dataset_root = sys.argv[2]   # e.g., C:\...\data-given-3dgs

scene_image_dir = os.path.join(dataset_root, scene)
images_txt_path = f"group_b_colmap/{scene}/sparse/0_txt/images.txt"

# 1) Gather original image filenames (recursive, strip extension, lower)
original = set()
for root, dirs, files in os.walk(scene_image_dir):
    for f in files:
        if f.lower().endswith(('.jpg','.jpeg','.png')):
            # Store the relative path string (without extension) for accurate comparison
            rel_path = os.path.relpath(os.path.join(root, f), scene_image_dir)
            name = os.path.splitext(rel_path)[0].replace('\\', '/').lower()
            original.add(name)

# 2) Parse COLMAP images.txt to get registered image names
registered = set()
with open(images_txt_path, 'r') as f:
    lines = f.readlines()

# In images.txt, an image block is:
#   <IMAGE_ID> <QW> <QX> <QY> <QZ> <TX> <TY> <TZ> <CAMERA_ID> <NAME>
#   <point2D lines...>
# We only care about the first line of each block, which has 10 tokens.
i = 0
while i < len(lines):
    line = lines[i].strip()
    if line == '' or line.startswith('#'):
        i += 1
        continue
    parts = line.split()
    # The image line has exactly 10 tokens (ID, 4 quaternion, 3 translation, camera_id, name)
    if len(parts) == 10:
        name = parts[-1]  # The last field is the image name (relative path)
        # Remove extension if present, but COLMAP stores without extension
        name = os.path.splitext(name)[0].replace('\\', '/').lower()
        registered.add(name)
        # Skip the following point lines (they have only 3 tokens)
        i += 1
        while i < len(lines) and lines[i].strip() != '' and not lines[i].startswith('#'):
            i += 1
    else:
        i += 1

missing = original - registered

print(f"Scene: {scene}")
print(f"Original images found: {len(original)}")
print(f"Registered images: {len(registered)}")
if missing:
    print(f"\nMISSING ({len(missing)}):")
    for m in sorted(missing):
        print(f"  {m}")
else:
    print("\n✓ All original images are registered!")