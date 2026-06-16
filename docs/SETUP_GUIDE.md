# Environment Setup & Quick Start Guide

Complete guide to set up your development environment for running the hackathon pipelines.

## System Requirements

### Minimum
- **CPU:** Intel i7 or AMD Ryzen 7 (8 cores recommended)
- **GPU:** NVIDIA GPU with 16GB VRAM (RTX 3080 or equivalent)
- **RAM:** 32GB system memory
- **Storage:** 500GB free space (for data, models, outputs)
- **OS:** Windows 10/11, Linux (Ubuntu 20.04+), macOS 12+

### Recommended
- **GPU:** NVIDIA RTX 4090 or NVIDIA A100 (24GB+)
- **RAM:** 64GB+
- **Storage:** 2TB+ SSD
- **Multi-GPU:** For H4 pipeline with parallel SR & training

---

## Installation Steps

### 1. Install NVIDIA Drivers & CUDA

#### Windows
```bash
# Download NVIDIA driver
# https://www.nvidia.com/Download/driverDetails.aspx

# Download CUDA Toolkit 12.4
# https://developer.nvidia.com/cuda-12-4-1-download-archive

# Install both (follow default installation)
# Verify installation:
nvidia-smi
# Should show GPU info and CUDA version 12.4
```

#### Linux (Ubuntu)
```bash
# Update package manager
sudo apt update
sudo apt upgrade

# Install CUDA Toolkit
sudo apt install -y nvidia-cuda-toolkit

# Verify
nvidia-smi
```

### 2. Install Conda

Download from: https://www.anaconda.com/download

Or use Miniconda (lighter):
```bash
# Windows PowerShell
# Download: https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html
# Run installer and follow prompts

# Linux/macOS
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Verify
conda --version
```

### 3. Clone This Repository

```bash
# Navigate to your workspace
cd /path/to/workspace

# Clone from GitHub (after pushing)
git clone https://github.com/YOUR_USERNAME/hackathon-3dgs.git
cd hackathon_codebase

# Or copy from local iterations
# (instructions provided in main README)
```

### 4. Create Conda Environment

```bash
# Create environment with Python 3.10 and PyTorch
conda create -n 3dgs python=3.10 pytorch pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Activate environment
conda activate 3dgs

# Verify PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

### 5. Install Base Dependencies

```bash
# From hackathon_codebase root
pip install -r configs/requirements_hloc.txt
pip install -r configs/requirements_hpc.txt
```

If using pip list, common packages needed:
```bash
pip install -q \
    opencv-python \
    opencv-contrib-python \
    numpy \
    scipy \
    matplotlib \
    plotly \
    Pillow \
    PyYAML \
    tqdm \
    h5py \
    kornia \
    pycolmap \
    roma \
    einops \
    scikit-learn \
    requests \
    gdown
```

### 6. Clone External Repositories

```bash
# From hackathon_codebase root
git clone https://github.com/graphdeco-inria/gaussian-splatting.git
git clone https://github.com/cvg/Hierarchical-Localization.git
git clone https://github.com/xinntao/Real-ESRGAN.git
git clone https://github.com/naver/mast3r.git

# Optional advanced methods
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
```

### 7. Install External Repository Dependencies

```bash
# Gaussian Splatting
cd gaussian-splatting
pip install -e .
cd ..

# Hierarchical-Localization
cd Hierarchical-Localization
pip install -e .
cd ..

# Real-ESRGAN
cd Real-ESRGAN
pip install basicsr
cd ..

# MAst3R
cd mast3r
pip install -r requirements.txt
pip install -r dust3r/requirements.txt
cd ..
```

### 8. Download Pre-trained Models

```bash
# Real-ESRGAN model
python << 'EOF'
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
# Models auto-download on first use to ~/.cache/realesrgan

# MAst3R model
from mast3r.model_zoo import download_model_if_not_exists
download_model_if_not_exists('mast3r_large.pth')

# Verify downloads
import os
cache_dir = os.path.expanduser('~/.cache')
print(f"Cache directory: {cache_dir}")
EOF
```

### 9. Verify Installation

```bash
# Test each major component
python << 'EOF'
import sys
print("=" * 60)
print("DEPENDENCY VERIFICATION")
print("=" * 60)

# PyTorch
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except Exception as e:
    print(f"✗ PyTorch: {e}")

# OpenCV
try:
    import cv2
    print(f"✓ OpenCV {cv2.__version__}")
except Exception as e:
    print(f"✗ OpenCV: {e}")

# HLoc
try:
    from hloc import extract_features
    print(f"✓ Hierarchical-Localization")
except Exception as e:
    print(f"✗ HLoc: {e}")

# Real-ESRGAN
try:
    from realesrgan import RealESRGANer
    print(f"✓ Real-ESRGAN")
except Exception as e:
    print(f"✗ Real-ESRGAN: {e}")

# MAst3R
try:
    from mast3r.model_zoo import load_model
    print(f"✓ MAst3R")
except Exception as e:
    print(f"✗ MAst3R: {e}")

# PyColmap
try:
    import pycolmap
    print(f"✓ PyColmap")
except Exception as e:
    print(f"✗ PyColmap: {e}")

print("=" * 60)
EOF
```

---

## Prepare Data

### Directory Structure

```bash
# Create data directories
mkdir -p data_input/{aeroplane,bike,buddha,cycle,face,firehydrant,still3,toy}/images

# Your structure should look like:
data_input/
├── aeroplane/images/     (place JPG/PNG files here)
├── bike/images/
├── buddha/images/
├── cycle/images/
├── face/images/
├── firehydrant/images/
├── still3/images/
└── toy/images/
```

### Expected File Names
- Standard naming: `DSC_0001.jpg`, `image_001.png`, etc.
- Works with most image formats: JPG, PNG, TIFF
- All images in one scene must have consistent resolution (will be resized otherwise)

---

## Quick Start - Running Your First Pipeline

### Option 1: Basic Pipeline (Recommended for Testing)

```bash
# Activate environment
conda activate 3dgs

# Navigate to codebase
cd hackathon_codebase

# Test with single scene
python pipeline/step1_upsample.py --scenes bike --method bicubic

# This creates upsampled images quickly (Bicubic is fast but lower quality)
# Output: data_sr/bike/images/
```

### Option 2: Full Quality Pipeline

```bash
# 1. Upsampling (4× Real-ESRGAN, ~30 min per scene)
python pipeline/step1_upsample.py --method realesrgan

# 2. COLMAP SfM (~20 min per scene)
python pipeline/step2_colmap.py

# 3. 3DGS Training (~1 hour per scene on RTX 3090)
python pipeline/step3_train_3dgs.py --iterations 30000

# 4. Render Submission (~10 min per scene)
python pipeline/step4_render_submit.py

# Output ready in: submission/<scene>/
```

### Option 3: Kaggle Notebook Pipeline

```bash
# Jupyter must be installed
pip install jupyter

# Start Jupyter
jupyter notebook

# Open in browser: http://localhost:8888
# Navigate to: notebooks/h2_kaggle_s2gs_scene_pipeline.ipynb
# Edit scene name in first cell
# Run → Run All Cells
```

### Option 4: MAst3R (H4 - Advanced)

```bash
# For single scene testing:
python scripts/h4_run_mast3r_groupb_final.py \
    --data-root data_input \
    --out-root mast3r_sparse \
    --scenes bike

# With GSCPR refinement:
python scripts/h4_run_gscpr_refinement.py \
    --manifest mast3r_sparse/gscpr_manifest.json \
    --command "/path/to/gscpr_binary"
```

---

## Debugging & Performance

### Check GPU Usage

```bash
# Terminal 1: Monitor GPU
watch -n 1 nvidia-smi

# Or Python script
python << 'EOF'
import torch
import psutil

print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
print(f"GPU Cached: {torch.cuda.memory_reserved() / 1e9:.1f} GB")
print(f"CPU Usage: {psutil.cpu_percent()}%")
print(f"RAM Used: {psutil.virtual_memory().percent}%")
EOF
```

### Common Issues & Solutions

#### Issue: CUDA out of memory
```bash
# Solution 1: Reduce batch size/tile size
python pipeline/step1_upsample.py --tile_size 256

# Solution 2: Use CPU for that step (slower)
python pipeline/step1_upsample.py --no_gpu

# Solution 3: Process fewer scenes
python pipeline/step3_train_3dgs.py --scenes bike buddha
```

#### Issue: COLMAP not found
```bash
# Windows: Manually specify COLMAP path
python pipeline/step2_colmap.py --colmap_path "C:/path/to/colmap.exe"

# Or ensure it's in system PATH:
# Control Panel → System → Environment Variables → Add to PATH
```

#### Issue: Models not downloading
```bash
# Manually download and place in cache
# Real-ESRGAN: ~/.cache/realesrgan/
# MAst3R: ~/.cache/torch/checkpoints/ or setup per their docs
# Verify: check folder contents after manual download
```

### Performance Tuning

```bash
# Adjust these in pipeline scripts for performance:

# For faster training (lower quality)
ITERATIONS = 7000
LR = 0.01

# For better quality (slower)
ITERATIONS = 50000
EVAL_INTERVAL = 1000

# For memory efficiency
BATCH_SIZE = 1  # Smaller batches
TILE_SIZE = 256  # Smaller processing tiles
```

---

## Multi-GPU Setup (For H4 Pipeline)

If you have 2+ GPUs:

```bash
# Set visible devices
export CUDA_VISIBLE_DEVICES=0,1

# In Python scripts, this is set automatically in H2/H4 notebooks
# GPU 0: 3DGS training
# GPU 1: Real-ESRGAN upsampling
```

---

## Conda Environment Export/Import

### Export Current Environment

```bash
# After full setup, save environment
conda env export > 3dgs_environment.yml

# Add to git
git add 3dgs_environment.yml
git commit -m "docs: add environment specification"
```

### Others Can Reproduce Environment

```bash
# Clone repo and recreate environment
git clone https://github.com/YOUR_USERNAME/hackathon-3dgs.git
cd hackathon_codebase
conda env create -f 3dgs_environment.yml
conda activate 3dgs
```

---

## Running on HPC (Cluster)

See `docs/h4_HPC_FINAL_RUN.md` for detailed HPC setup.

Quick template:
```bash
#!/bin/bash
#SBATCH --job-name=3dgs_train
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=12:00:00

module load cuda/12.4
source activate 3dgs

python pipeline/step3_train_3dgs.py --iterations 30000
```

---

## Troubleshooting & FAQ

### Q: Where are intermediate outputs?
**A:** Check these directories:
- `data_sr/` - Upsampled images (Step 1)
- `data_ready/` - COLMAP sparse reconstructions (Step 2)
- `output/` - Trained 3DGS models (Step 3)
- `submission/` - Rendered final images (Step 4)

### Q: How do I resume from a specific step?
**A:** Each step uses outputs from the previous step. Just run the next step:
```bash
# Resume from step 3 (skip upsampling/COLMAP)
python pipeline/step3_train_3dgs.py
```

### Q: Can I use CPU only?
**A:** Yes, but very slow. Add flags:
```bash
python pipeline/step1_upsample.py --no_gpu
python pipeline/step2_colmap.py --no_gpu
```

### Q: How do I handle failure mid-pipeline?
**A:**
1. Check what failed
2. Fix issue (more memory, install missing lib, etc.)
3. Delete outputs from failed step
4. Rerun that step and continue

### Q: Are there test scripts?
**A:** Yes! Use test files:
```bash
python scripts/h1_diagnostic.py
python scripts/h2_checkcsv.py
python scripts/h4_check_missing_v2.py
```

---

## Next Steps

1. ✓ Install environment (you are here)
2. Run a quick test: `python pipeline/step1_upsample.py --scenes bike --method bicubic`
3. Prepare your data in `data_input/`
4. Run full pipeline: `./run_pipeline.sh`
5. Check outputs in `submission/`
6. Upload to competition/share results

---

## Support Resources

- **PyTorch:** https://pytorch.org
- **Gaussian-Splatting:** https://github.com/graphdeco-inria/gaussian-splatting
- **HLoc:** https://github.com/cvg/Hierarchical-Localization
- **MAst3R:** https://github.com/naver/mast3r
- **COLMAP:** https://colmap.github.io

---

**Last Updated:** 2026-06-16
