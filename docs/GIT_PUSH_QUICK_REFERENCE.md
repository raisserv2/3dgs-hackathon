# Git Push Quick Reference Card

**TL;DR - Copy & Paste This to Push to GitHub**

---

## One-Time Setup

```bash
# Navigate to project
cd "c:\main\IITM\SEM_4\Modern Computer Vision\HACKATHON_ITERATIONS\hackathon_codebase"

# Initialize git
git init

# Configure your identity (one-time)
git config user.name "Your Full Name"
git config user.email "your.email@example.com"

# Add GitHub repository (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/hackathon-3dgs.git

# Verify
git remote -v
```

---

## First Push

```bash
# Stage all files
git add .

# Create initial commit
git commit -m "Initial commit: Consolidated hackathon iterations H1-H4

- H1: Core Real-ESRGAN + COLMAP + 3DGS pipeline
- H2: Kaggle S2Gaussian and alternative approaches  
- H3: Extended SfM experiments (MAst3R, mip-splatting)
- H4: HPC-optimized with GLOMAP and GSCPR

Includes pipeline scripts, utilities, notebooks, comprehensive docs."

# Set main branch and push
git branch -M main
git push -u origin main
```

When prompted for password/token:
- **Windows:** GitHub will open browser to authenticate
- **If prompted for username/password:** Use your GitHub username and a Personal Access Token (not password)
  - Generate token at: https://github.com/settings/tokens
  - Scopes needed: `repo` (full control)

---

## Verify Success

```bash
# Check current status
git status
# Should show: "On branch main, nothing to commit, working tree clean"

# View commit history
git log --oneline
```

Then visit: `https://github.com/YOUR_USERNAME/hackathon-3dgs` to verify!

---

## Future Pushes (After Initial Setup)

```bash
# Check what changed
git status

# Stage changes
git add .

# Commit
git commit -m "Your descriptive message here"

# Push
git push
```

---

## Useful Commands

```bash
# View changes before committing
git diff

# View history
git log --oneline -10

# Undo last commit (keep files)
git reset --soft HEAD~1

# Undo last commit (discard files)
git reset --hard HEAD~1

# Create a new branch
git checkout -b feature-name
git push -u origin feature-name
```

---

## Files Included in This Push

### Core Pipeline Scripts (5 files)
- `step1_upsample.py` - 4× Super-resolution
- `step2_colmap.py` - COLMAP SfM
- `step2b_hloc.py` - Hierarchical Localization
- `step3_train_3dgs.py` - 3D Gaussian Splatting
- `step4_render_submit.py` - Rendering

### Utility Scripts (20+ files)
- H1: `h1_create_binaries.py`, `h1_diagnostic.py`, etc.
- H2: `h2_build_submissions.py`, `h2_render_kaggle_output.py`, etc.
- H4: `h4_run_glomap_b.py`, `h4_run_mast3r_groupb_final.py`, etc.

### Jupyter Notebooks (3 files)
- `h2_kaggle_s2gs_scene_pipeline.ipynb`
- `h2_kaggle_sequential.ipynb`
- `h2_kaggle1.ipynb`

### Utilities Module
- `utils/llff_split.py` - Train/test splitting

### Configuration
- `configs/requirements_hloc.txt` - HLoc dependencies
- `configs/requirements_hpc.txt` - HPC environment

### Documentation
- `README.md` - Comprehensive project guide (this is key!)
- `docs/GITHUB_SETUP.md` - Detailed git instructions
- `docs/SETUP_GUIDE.md` - Development environment setup
- `docs/h4_HPC_FINAL_RUN.md` - HPC execution guide
- `docs/h4_README_hloc_setup.md` - HLoc setup
- `.gitignore` - Files to exclude (data, outputs, etc.)

---

## Total Content Summary

| Category | Count | Details |
|----------|-------|---------|
| Pipeline Scripts | 5 | Main numbered steps 1-4 |
| Utility Scripts | 20+ | Across H1/H2/H3/H4 variants |
| Jupyter Notebooks | 3 | End-to-end pipeline examples |
| Utility Modules | 2 | llff_split.py + __init__.py |
| Config Files | 2 | requirements_hloc/hpc.txt |
| Documentation | 5 | README + 4 setup guides |
| Git Config | 1 | .gitignore |
| **Total Files** | **~40** | **Ready to push!** |

---

## What's NOT Included (Correctly Excluded)

These are excluded via `.gitignore` (as intended):
- ❌ `data_input/`, `data_sr/`, `data_ready/`, `output/` (large data)
- ❌ `submission/` (generated outputs)
- ❌ `gaussian-splatting/`, `Hierarchical-Localization/` (external repos - clone separately)
- ❌ `runs/`, `logs/`, `weights/` (training artifacts)
- ❌ `__pycache__/`, `.venv/` (Python cache)

This keeps your repository lean (~1-2 MB) while external repos are cloned separately.

---

## Repository Structure on GitHub

After pushing, your GitHub repo will look like:
```
hackathon-3dgs/
├── README.md                    ← GitHub displays this
├── .gitignore
├── pipeline/
│   ├── step1_upsample.py
│   ├── step2_colmap.py
│   ├── step2b_hloc.py
│   ├── step3_train_3dgs.py
│   └── step4_render_submit.py
├── scripts/
│   ├── h1_*.py (8 files)
│   ├── h2_*.py (8 files)
│   ├── h4_*.py (8 files)
│   └── h1_run_pipeline.bat
├── notebooks/
│   ├── h2_kaggle_s2gs_scene_pipeline.ipynb
│   ├── h2_kaggle_sequential.ipynb
│   └── h2_kaggle1.ipynb
├── utils/
│   ├── llff_split.py
│   └── __init__.py
├── configs/
│   ├── requirements_hloc.txt
│   └── requirements_hpc.txt
└── docs/
    ├── GITHUB_SETUP.md
    ├── SETUP_GUIDE.md
    ├── h4_HPC_FINAL_RUN.md
    └── h4_README_hloc_setup.md
```

---

## After Pushing: Next Steps

1. **Share your repository URL:**
   ```
   https://github.com/YOUR_USERNAME/hackathon-3dgs
   ```

2. **Add collaborators (if needed):**
   - GitHub → Settings → Collaborators → Add people

3. **Enable GitHub Pages (optional):**
   - GitHub → Settings → Pages → Enable
   - Choose branch/folder for docs
   - Share docs at: `https://YOUR_USERNAME.github.io/hackathon-3dgs/`

4. **Create Releases (optional):**
   ```bash
   git tag -a v1.0 -m "Release: Consolidated H1-H4"
   git push origin v1.0
   ```

5. **Add GitHub Actions CI/CD (optional):**
   - Create `.github/workflows/` folder
   - Add tests/linting workflows

---

## Authentication Troubleshooting

### If Using HTTPS & Git Asks for Password

**You MUST use a Personal Access Token (not your GitHub password)**

1. Create token: https://github.com/settings/tokens
   - Click "Generate new token (classic)"
   - Give it a name
   - Select `repo` scope
   - Copy the token (shown only once!)

2. When Git asks for password, paste the token instead

3. To save token (avoid re-entering):
   ```bash
   # Windows: Git Credential Manager will save automatically
   # Or manually:
   git config --global credential.helper manager
   ```

### If Using SSH (Better Long-term)

1. Generate key:
   ```bash
   ssh-keygen -t ed25519 -C "your.email@example.com"
   ```

2. Add to GitHub:
   - https://github.com/settings/ssh/new
   - Paste contents of `~/.ssh/id_ed25519.pub`

3. Change remote to SSH:
   ```bash
   git remote set-url origin git@github.com:YOUR_USERNAME/hackathon-3dgs.git
   ```

---

## Minimal Command Sequence (Copy-Paste)

```bash
cd "c:\main\IITM\SEM_4\Modern Computer Vision\HACKATHON_ITERATIONS\hackathon_codebase"
git init
git config user.name "Your Name"
git config user.email "your@email.com"
git remote add origin https://github.com/YOUR_USERNAME/hackathon-3dgs.git
git add .
git commit -m "Initial commit: Consolidated hackathon H1-H4"
git branch -M main
git push -u origin main
```

---

## Need Help?

| Issue | Solution |
|-------|----------|
| "fatal: could not read Username" | Use Personal Access Token instead of password |
| "Permission denied (publickey)" | Use HTTPS instead of SSH, or setup SSH keys |
| "repository not found" | Check spelling of username/repo, or create repo first |
| "remote origin already exists" | `git remote remove origin` then `git remote add origin ...` |
| "divergent branches" | `git pull --rebase origin main` before pushing |

More help: See `docs/GITHUB_SETUP.md` for detailed instructions.

---

**Ready to push? Follow the commands at the top of this file!**

**Questions? Read:** `docs/GITHUB_SETUP.md` (comprehensive guide) or `README.md` (project overview)
