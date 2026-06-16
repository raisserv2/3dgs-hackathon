"""
Convert COLMAP camera model to PINHOLE for 3DGS compatibility.
Handles hloc's camera.bin format correctly.
"""

import struct
from pathlib import Path

def read_cameras_bin_safe(path):
    """Safely read cameras.bin and detect model type"""
    cameras = []
    with open(path, "rb") as fid:
        num_cameras = struct.unpack("<Q", fid.read(8))[0]
        print(f"  Found {num_cameras} cameras")
        
        for i in range(num_cameras):
            camera_id = struct.unpack("<I", fid.read(4))[0]
            model_id = struct.unpack("<I", fid.read(4))[0]
            width = struct.unpack("<Q", fid.read(8))[0]
            height = struct.unpack("<Q", fid.read(8))[0]
            num_params = struct.unpack("<Q", fid.read(8))[0]
            
            # Read params as doubles
            params = []
            for _ in range(num_params):
                params.append(struct.unpack("<d", fid.read(8))[0])
            
            cameras.append({
                'id': camera_id,
                'model_id': model_id,
                'width': width,
                'height': height,
                'params': params
            })
    return cameras

def write_cameras_pinhole(path, cameras):
    """Write as PINHOLE model (id=1) with 4 params: fx, fy, cx, cy"""
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", len(cameras)))
        
        for cam in cameras:
            params = cam['params']
            w, h = cam['width'], cam['height']
            
            # Convert based on original model
            model_id = cam['model_id']
            
            if model_id == 0:  # SIMPLE_PINHOLE: f, cx, cy
                f, cx, cy = params[0], params[1], params[2]
                fx = fy = f
            elif model_id == 1:  # PINHOLE: fx, fy, cx, cy
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            elif model_id == 2 or model_id == 3:  # SIMPLE_RADIAL or RADIAL
                f, cx, cy = params[0], params[1], params[2]
                fx = fy = f
            elif model_id >= 4:  # OPENCV, etc. — take first 4 params
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            else:
                # Fallback: guess
                fx = fy = params[0] if len(params) > 0 else w / 2
                cx = params[1] if len(params) > 1 else w / 2
                cy = params[2] if len(params) > 2 else h / 2
            
            # Write as PINHOLE (model_id=1)
            fid.write(struct.pack("<I", cam['id']))   # camera_id
            fid.write(struct.pack("<I", 1))           # model_id = PINHOLE
            fid.write(struct.pack("<Q", w))           # width
            fid.write(struct.pack("<Q", h))           # height
            fid.write(struct.pack("<Q", 4))           # num_params = 4
            fid.write(struct.pack("<dddd", fx, fy, cx, cy))
    
    print(f"  Written {len(cameras)} PINHOLE cameras")

def main():
    # Add all scenes that have hloc reconstructions
    scenes = ["aeroplane", "face", "still3", "cycle"]
    
    for scene in scenes:
        sparse_dir = Path(f"data_ready/{scene}/sparse/0")
        cameras_bin = sparse_dir / "cameras.bin"
        
        if not cameras_bin.exists():
            print(f"[{scene}] cameras.bin not found, skipping")
            continue
        
        print(f"\n[{scene}] Converting camera model...")
        
        # Read original
        cameras = read_cameras_bin_safe(cameras_bin)
        print(f"  Original model_id: {cameras[0]['model_id']} (0=SIMPLE_PINHOLE, 1=PINHOLE, 2=SIMPLE_RADIAL, 4=OPENCV)")
        print(f"  Original params: {cameras[0]['params']}")
        
        # Backup
        backup = sparse_dir / "cameras.bin.original"
        if not backup.exists():
            cameras_bin.rename(backup)
            print(f"  Backed up to {backup}")
        else:
            cameras_bin.unlink()
        
        # Write as PINHOLE
        write_cameras_pinhole(cameras_bin, cameras)
        print(f"  ✓ Converted to PINHOLE")

if __name__ == "__main__":
    main()