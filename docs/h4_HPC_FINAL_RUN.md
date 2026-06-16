# HPC Final Run Guide

## Layout Assumed

- Home code area: `/home/da24b009`
- Large data/output area: `/storage/da24b009`

Recommended subfolders:

- `/home/da24b009/mast3r` for the MASt3R repo checkout
- `/home/da24b009/GS-CPR` for the GSCPR repo checkout
- `/storage/da24b009/data-given-3dgs` for input images
- `/storage/da24b009/checkpoints` for the MASt3R checkpoint
- `/storage/da24b009/mast3r_sparse_final` for MASt3R outputs
- `/storage/da24b009/gscpr_outputs` for GSCPR refined outputs

## Environment

Use the existing `3dgs` conda environment if it already has:

- `torch` and `torchvision` built for CUDA 12.4
- `pycolmap`
- `hloc`
- `opencv-python`
- `kornia`
- `h5py`
- `scipy`
- `matplotlib`
- `plotly`
- `tqdm`
- `gdown`
- `numpy`
- `Pillow`
- `PyYAML`
- `requests`

For MASt3R, install the repo-local requirements too:

```bash
python -m pip install -r /home/da24b009/mast3r/requirements.txt
python -m pip install -r /home/da24b009/mast3r/dust3r/requirements.txt
```

That covers the MASt3R runtime extras such as `roma`, `einops`, `trimesh`, `huggingface-hub`, `scikit-learn`, `gradio`, and `pyglet`.

## Stage 1: MASt3R-SfM

Run the final MASt3R pipeline from the home copy of the script:

```bash
python /home/da24b009/hackathon_final/run_mast3r_groupb_final.py \
  --data_root /storage/da24b009/data-given-3dgs \
  --out_root /storage/da24b009/mast3r_sparse_final \
  --mast3r_dir /home/da24b009/mast3r \
  --checkpoint /storage/da24b009/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
  --scenes buddha,firehydrant,toy
```

What it writes per scene:

- `/storage/da24b009/mast3r_sparse_final/<scene>/pointcloud.ply`
- `/storage/da24b009/mast3r_sparse_final/<scene>/sparse/0/cameras.txt`
- `/storage/da24b009/mast3r_sparse_final/<scene>/sparse/0/images.txt`
- `/storage/da24b009/mast3r_sparse_final/<scene>/sparse/0/points3D.txt`
- `/storage/da24b009/mast3r_sparse_final/<scene>/gscpr_manifest.json`

## Stage 2: GSCPR refinement

The repository here does not contain the GSCPR source code, so the downstream stage is wired as a wrapper, not a guessed implementation.

Run it after cloning the GSCPR repo in `/home/da24b009/GS-CPR` and after you know the repo's exact entrypoint:

```bash
python /home/da24b009/hackathon_final/run_gscpr_refinement.py \
  --manifest /storage/da24b009/mast3r_sparse_final/<scene>/gscpr_manifest.json \
  --command "<your GSCPR command here>"
```

The wrapper exports these environment variables for the GSCPR process:

- `MAST3R_MANIFEST`
- `MAST3R_SPARSE_DIR`
- `MAST3R_OUTPUT_DIR`
- `MAST3R_IMAGES_DIR`
- `MAST3R_POINTCLOUD`

## If you want to run the full chain in one shot

You can pass `--gscpr_cmd` directly to the MASt3R launcher once you know the GSCPR command:

```bash
python /home/da24b009/hackathon_final/run_mast3r_groupb_final.py \
  --data_root /storage/da24b009/data-given-3dgs \
  --out_root /storage/da24b009/mast3r_sparse_final \
  --mast3r_dir /home/da24b009/mast3r \
  --checkpoint /storage/da24b009/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
  --scenes buddha,firehydrant,toy \
  --gscpr_cmd "<your GSCPR command here>"
```

## Practical dependency answer

If the `3dgs` env already runs the current MASt3R pipeline, you are probably only missing the MASt3R-specific extras from the repo requirements. For the MASt3R stage, the likely additions are:

- `roma`
- `einops`
- `trimesh`
- `huggingface-hub[torch]>=0.22`
- `scikit-learn`
- optionally `gradio`, `tensorboard`, `pyglet<2` if you use demo/TSDF features

The GSCPR stage may need its own repo-specific requirements once the exact GSCPR codebase is present.
