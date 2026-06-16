import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "Hierarchical-Localization"))

from hloc import extract_features, match_features, reconstruction

def run_hloc(scene):
    img_dir = Path(f"data_processed/{scene}/images_hr")
    sfm_dir = Path(f"data_processed/{scene}/sfm")
    sfm_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Extract SuperPoint features
    features = sfm_dir / "features.h5"
    extract_features.main(extract_features.confs["superpoint_aachen"], img_dir, feature_path=features)
    
    # 2. Match features with SuperGlue
    pairs = sfm_dir / "pairs.txt"
    with open(pairs, "w") as f:
        img_names = sorted([p.name for p in img_dir.glob("*.jpg")])
        for i, a in enumerate(img_names):
            for b in img_names[i+1:]:
                f.write(f"{a} {b}\n")
    matches = sfm_dir / "matches.h5"
    match_features.main(match_features.confs["superglue"], pairs, features=features, matches=matches)
    
    # 3. Run COLMAP reconstruction
    model = reconstruction.main(sfm_dir, img_dir, pairs, features, matches)
    
    # 4. Export to sparse/0 (3DGS expects this structure)
    out_dir = Path(f"data_processed/{scene}/sparse/0")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.write(str(out_dir))
    print(f"Reconstruction for {scene} done: {len(model.images)} images, {model.num_points3D} points.")

if __name__ == "__main__":
    scenes = ["aeroplane", "cycle", "face", "still3"]
    for scene in scenes:
        run_hloc(scene)