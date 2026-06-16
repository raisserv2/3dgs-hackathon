import os
import pycolmap

# ----- CONFIGURE YOUR PATHS HERE -----
# Assuming this script is run from the root of your downloaded dataset
GROUP_B_ROOT = "data_given"  
OUT_ROOT = "./group_b_poses"
os.makedirs(OUT_ROOT, exist_ok=True)

scene_names = ["aeroplane", "cycle", "face", "still3"]

print("Scenes to process:", scene_names)

for scene in scene_names:
    scene_image_dir = os.path.join(GROUP_B_ROOT, scene)
    scene_out_dir   = os.path.join(OUT_ROOT, scene)
    os.makedirs(scene_out_dir, exist_ok=True)
    
    print(f"\n=== Processing {scene} ===")
    
    # 1. Configure feature extraction (optimized for low-res images)
    extract_opts = pycolmap.FeatureExtractionOptions()
    extract_opts.max_num_features = 8192
    extract_opts.first_octave = 0
    # Use a simple, general camera model
    camera_mode = pycolmap.CameraMode.SINGLE
    
    # 2. Configure matching (exhaustive with cross-check)
    match_opts = pycolmap.SiftMatchingOptions()
    match_opts.max_ratio = 0.85
    match_opts.cross_check = True
    
    try:
        # 3. Run GLOMAP
        reconstruction = pycolmap.run_glomap(
            scene_image_dir,
            scene_out_dir,
            quality='medium',
            camera_mode=camera_mode,
            extraction_options=extract_opts,
            matching_options=match_opts,
            verbose=True
        )
        n_cameras = len(reconstruction.images)
        n_points  = len(reconstruction.points3D)
        print(f"  ✓ SUCCESS: {n_cameras} cameras registered, {n_points} sparse points")
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")