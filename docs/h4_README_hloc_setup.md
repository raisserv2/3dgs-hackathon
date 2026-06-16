# hloc + COLMAP Pose Estimation Setup

This setup replaces SIFT extraction/matching with hloc learned features and still uses COLMAP incremental mapping to estimate camera poses.

## 1) Install dependencies

Use the conda environment for the A100 setup:

```bash
conda activate 3dgs
python -m pip install -r requirements_hloc.txt
```

If you want a single file for the HPC server, use:

```bash
python -m pip install -r requirements_hpc.txt
```

That file pins the same A100-oriented stack used by the script defaults.

If `hloc` install fails directly from pip on your machine, install from source:

```bash
pip install git+https://github.com/cvg/Hierarchical-Localization.git
```

## 2) Dataset layout expected by script

Default dataset root is:

- `data-given-3dgs/`

Each scene can be either:

- `data-given-3dgs/<scene>/images/*`
- or directly `data-given-3dgs/<scene>/*` (script auto-detects)

## 3) Run full pipeline

```bash
python run_glomap_b.py
```

The current default front end is `aliked-n16` + `aliked+lightglue`, with multi-pass
resize settings tuned for low-resolution scenes.

Optional custom paths/scenes:

```bash
python run_glomap_b.py --data_root data-given-3dgs --out_root group_b_hloc_colmap --scenes aeroplane,cycle,face,still3
```

## 4) Outputs

For each scene, outputs are written under:

- `group_b_hloc_colmap/<scene>/`

Important files:

- `features-superpoint.h5` (hloc features)
- `matches-superglue.h5` (hloc matches)
- `pairs-sfm.txt` (image pairs)
- `sfm/` (COLMAP sparse model)
- `poses.csv` (camera poses per registered image)

`poses.csv` columns:

- `image_name, qw, qx, qy, qz, tx, ty, tz, camera_id`

## 5) Notes for low-resolution images

- Learned features (SuperPoint) are generally more robust than SIFT on low-res images.
- Matching uses SuperGlue by default.
- Mapping is set with `SIMPLE_RADIAL` and single camera assumptions in the script fallback path.
