"""
Create minimal COLMAP binary files for 3DGS
"""

import struct
import numpy as np
from pathlib import Path

def create_cameras_bin(path, width, height, fx, fy, cx, cy):
    """Create cameras.bin with PINHOLE model"""
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', 1))  # 1 camera
        f.write(struct.pack('<I', 1))  # camera_id = 1
        f.write(struct.pack('<I', 1))  # model_id = PINHOLE
        f.write(struct.pack('<Q', width))
        f.write(struct.pack('<Q', height))
        f.write(struct.pack('<Q', 4))  # 4 params
        f.write(struct.pack('<dddd', fx, fy, cx, cy))
    print(f"  Created {path}")

def create_images_bin(path, num_images, image_names):
    """Create images.bin with placeholder poses"""
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', num_images))
        for i, name in enumerate(image_names, 1):
            f.write(struct.pack('<I', i))           # image_id
            f.write(struct.pack('<dddd', 1, 0, 0, 0))  # quat (identity)
            f.write(struct.pack('<ddd', 0, 0, 0))   # translation (zero)
            f.write(struct.pack('<I', 1))           # camera_id
            f.write(struct.pack('<I', len(name)))   # name length
            f.write(name.encode('utf-8'))           # name
            f.write(b'\x00')                        # null terminator
            f.write(struct.pack('<Q', 0))           # no 2D points
    print(f"  Created {path}")

def create_points_bin(path, num_points=2000):
    """Create points3D.bin with random points"""
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', num_points))
        for i in range(num_points):
            x = np.random.uniform(-2, 2)
            y = np.random.uniform(-2, 2)
            z = np.random.uniform(-2, 2)
            r = np.random.randint(100, 200)
            g = np.random.randint(100, 200)
            b = np.random.randint(100, 200)
            f.write(struct.pack('<Q', i+1))      # point_id
            f.write(struct.pack('<ddd', x, y, z)) # XYZ
            f.write(struct.pack('<BBB', r, g, b)) # RGB
            f.write(struct.pack('<d', 1.0))      # error
            f.write(struct.pack('<Q', 0))        # track length (0)
    print(f"  Created {path}")

def main():
    scenes = ["aeroplane", "face", "still3", "cycle"]
    
    # Camera parameters (from your earlier output)
    scene_params = {
        "aeroplane": {"width": 1092, "height": 728, "fx": 1310.40, "fy": 1310.40, "cx": 546.00, "cy": 364.00},
        "face": {"width": 640, "height": 480, "fx": 768.00, "fy": 768.00, "cx": 320.00, "cy": 240.00},
        "still3": {"width": 1152, "height": 768, "fx": 1382.40, "fy": 1382.40, "cx": 576.00, "cy": 384.00},
        "cycle": {"width": 1128, "height": 737, "fx": 1353.60, "fy": 1353.60, "cx": 564.00, "cy": 368.50},
    }
    
    for scene in scenes:
        sparse_dir = Path(f"data_ready/{scene}/sparse/0")
        sparse_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[{scene}]")
        
        # Get image names
        images_dir = Path(f"data_ready/{scene}/images")
        image_names = [f.name for f in images_dir.iterdir() if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]
        num_images = len(image_names)
        print(f"  Found {num_images} images")
        
        # Create binaries
        params = scene_params[scene]
        create_cameras_bin(sparse_dir / "cameras.bin", 
                          params["width"], params["height"],
                          params["fx"], params["fy"], 
                          params["cx"], params["cy"])
        create_images_bin(sparse_dir / "images.bin", num_images, image_names)
        create_points_bin(sparse_dir / "points3D.bin", num_points=1000)
        
        print(f"  ✓ Ready for training")

if __name__ == "__main__":
    main()