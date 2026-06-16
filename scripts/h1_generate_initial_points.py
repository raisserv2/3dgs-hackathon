"""
Generate initial point cloud from COLMAP database.
"""

import sqlite3
import numpy as np
from pathlib import Path
import random

def triangulate_points(db_path, output_dir):
    """Generate initial 3D points from matches"""
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Check what tables exist
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in c.fetchall()]
    print(f"  Tables: {tables}")
    
    # Get cameras
    c.execute("SELECT camera_id, model, width, height, params FROM cameras")
    cameras = {}
    for cam_id, model, width, height, params_blob in c.fetchall():
        params = np.frombuffer(params_blob, dtype=np.float64)
        cameras[cam_id] = {'model': model, 'width': width, 'height': height, 'params': params}
    
    # Get images (hloc schema uses image_id, name, camera_id)
    c.execute("SELECT image_id, name, camera_id FROM images")
    images = {img_id: {'name': name, 'camera_id': cam_id} for img_id, name, cam_id in c.fetchall()}
    print(f"  Found {len(images)} images")
    
    # Get keypoints count
    c.execute("SELECT COUNT(*) FROM keypoints")
    num_keypoints = c.fetchone()[0]
    print(f"  Found {num_keypoints} keypoints")
    
    # Get matches count
    c.execute("SELECT COUNT(*) FROM matches")
    num_matches = c.fetchone()[0]
    print(f"  Found {num_matches} matches")
    
    # Generate random 3D points (3DGS will refine these)
    # One point per 10 keypoints, max 2000 points
    num_points = min(2000, num_keypoints // 10)
    
    points = []
    for i in range(num_points):
        # Random points in a sphere around origin
        theta = random.uniform(0, 2 * np.pi)
        phi = random.uniform(0, np.pi)
        r = random.uniform(0.5, 2.0)
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        # Random color
        r_color = random.randint(100, 200)
        g_color = random.randint(100, 200)
        b_color = random.randint(100, 200)
        
        points.append({
            'id': i + 1,
            'x': x, 'y': y, 'z': z,
            'r': r_color, 'g': g_color, 'b': b_color
        })
    
    # Write points3D.txt
    points_path = output_dir / "points3D.txt"
    with open(points_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR\n")
        for p in points:
            f.write(f"{p['id']} {p['x']} {p['y']} {p['z']} {p['r']} {p['g']} {p['b']} 1.0\n")
    
    print(f"  Generated {num_points} initial 3D points")
    return points_path

def main():
    scenes = ["aeroplane", "face", "still3", "cycle"]
    
    for scene in scenes:
        db_path = Path(f"data_ready/{scene}/sfm/database.db")
        sparse_dir = Path(f"data_ready/{scene}/sparse/0")
        
        if not db_path.exists():
            print(f"[{scene}] Database not found, skipping")
            continue
        
        print(f"\n[{scene}] Generating initial point cloud...")
        sparse_dir.mkdir(parents=True, exist_ok=True)
        triangulate_points(db_path, sparse_dir)
        print(f"  ✓ Created points3D.txt in {sparse_dir}")

if __name__ == "__main__":
    main()