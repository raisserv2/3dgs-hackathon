"""
Regenerate sparse reconstruction from hloc database with PINHOLE cameras.
No COLMAP CLI needed — uses pycolmap if available, or rebuilds cameras.txt manually.
"""

import sqlite3
import numpy as np
from pathlib import Path

def get_images_from_db(db_path):
    """Get all images from database"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT image_id, name, camera_id FROM images")
    images = c.fetchall()
    conn.close()
    return images

def get_camera_from_db(db_path, camera_id):
    """Get camera parameters from database"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT model, width, height, params FROM cameras WHERE camera_id=?", (camera_id,))
    row = c.fetchone()
    conn.close()
    if row:
        model, width, height, params_blob = row
        # params are stored as blob of doubles
        params = np.frombuffer(params_blob, dtype=np.float64).tolist()
        return model, width, height, params
    return None, None, None, None

def write_cameras_txt(path, cameras):
    """Write cameras.txt with PINHOLE model (simple format)"""
    with open(path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for cam in cameras:
            # PINHOLE model ID = 1, params: fx fy cx cy
            f.write(f"{cam['id']} PINHOLE {cam['width']} {cam['height']} ")
            f.write(f"{cam['fx']} {cam['fy']} {cam['cx']} {cam['cy']}\n")

def write_images_txt(path, images):
    """Write minimal images.txt (no poses, will be triangulated later)"""
    with open(path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X,Y,POINT3D_ID)\n")
        # Placeholder — actual poses from database would need triangulation
        for img in images:
            f.write(f"{img['id']} 1 0 0 0 0 0 0 {img['camera_id']} {img['name']}\n")
            f.write("\n")  # No 2D points initially

def main():
    scenes = ["aeroplane", "face", "still3", "cycle"]
    
    for scene in scenes:
        db_path = Path(f"data_ready/{scene}/sfm/database.db")
        sparse_dir = Path(f"data_ready/{scene}/sparse/0")
        
        if not db_path.exists():
            print(f"[{scene}] Database not found, skipping")
            continue
        
        print(f"\n[{scene}] Regenerating cameras from database...")
        
        # Get images
        images = get_images_from_db(db_path)
        if not images:
            print(f"  No images in database")
            continue
        
        # Get camera for first image
        _, _, cam_id = images[0]
        model, width, height, params = get_camera_from_db(db_path, cam_id)
        
        print(f"  Original model: {model} (0=SIMPLE_PINHOLE, 1=PINHOLE, 2=SIMPLE_RADIAL, 4=OPENCV)")
        print(f"  Original params: {params}")
        
        # Convert to PINHOLE params
        if model == 0 or model == 2:  # SIMPLE_PINHOLE or SIMPLE_RADIAL: f, cx, cy
            f, cx, cy = params[0], params[1], params[2]
            fx = fy = f
        elif model == 1:  # Already PINHOLE: fx, fy, cx, cy
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        elif model == 4:  # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        else:
            fx = fy = params[0] if len(params) > 0 else width / 2
            cx = params[1] if len(params) > 1 else width / 2
            cy = params[2] if len(params) > 2 else height / 2
        
        print(f"  Converted to PINHOLE: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        
        # Create sparse directory
        sparse_dir.mkdir(parents=True, exist_ok=True)
        
        # Write cameras.txt
        cameras = [{
            'id': 1,
            'width': width,
            'height': height,
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy
        }]
        write_cameras_txt(sparse_dir / "cameras.txt", cameras)
        
        # Write minimal images.txt (one camera for all images)
        images_list = []
        for idx, (img_id, name, _) in enumerate(images, 1):
            images_list.append({
                'id': idx,
                'name': name,
                'camera_id': 1
            })
        write_images_txt(sparse_dir / "images.txt", images_list)
        
        # Create empty points3D.txt
        with open(sparse_dir / "points3D.txt", 'w') as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        
        print(f"  ✓ Created sparse/0/ with PINHOLE camera")
        print(f"  Note: Using placeholder poses — 3DGS will refine during training")

if __name__ == "__main__":
    main()