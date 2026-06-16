# GitHub Setup & Push Commands

Complete step-by-step guide to initialize git and push the consolidated hackathon codebase to GitHub.

## Prerequisites

1. **Git installed:** https://git-scm.com/download
2. **GitHub account:** https://github.com/signup
3. **SSH or HTTPS authentication configured**

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Enter repository name: `hackathon-3dgs` (or your preferred name)
3. Add description: "Consolidated 3D Gaussian Splatting hackathon work (H1-H4)"
4. Choose **Public** or **Private** (recommend Private initially)
5. **DO NOT** initialize with README/gitignore (we have our own)
6. Click **Create repository**

After creation, you'll see a URL like:
```
https://github.com/YOUR_USERNAME/hackathon-3dgs.git
```
or
```
git@github.com:YOUR_USERNAME/hackathon-3dgs.git
```

## Step 2: Navigate to Project Directory

```bash
# Open PowerShell or Command Prompt
cd "c:\main\IITM\SEM_4\Modern Computer Vision\HACKATHON_ITERATIONS\hackathon_codebase"
```

## Step 3: Initialize Git Repository (HTTPS Method)

### Option A: HTTPS (Easiest for beginners)

```bash
# Initialize git
git init

# Configure user (one-time)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# For global config (applies to all repos on this machine):
# git config --global user.name "Your Name"
# git config --global user.email "your.email@example.com"

# Add GitHub as remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/hackathon-3dgs.git

# Verify remote
git remote -v
# Output should show:
# origin  https://github.com/YOUR_USERNAME/hackathon-3dgs.git (fetch)
# origin  https://github.com/YOUR_USERNAME/hackathon-3dgs.git (push)
```

### Option B: SSH (Recommended for security)

```bash
# Initialize git
git init

# Configure user (one-time)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Add GitHub as remote using SSH (replace YOUR_USERNAME)
git remote add origin git@github.com:YOUR_USERNAME/hackathon-3dgs.git

# Verify remote
git remote -v
# Output should show:
# origin  git@github.com:YOUR_USERNAME/hackathon-3dgs.git (fetch)
# origin  git@github.com:YOUR_USERNAME/hackathon-3dgs.git (push)
```

**Note:** For SSH, ensure your SSH key is added to GitHub:
- Generate: `ssh-keygen -t ed25519 -C "your.email@example.com"`
- Add to GitHub: Settings → SSH and GPG keys

## Step 4: Stage Files

```bash
# View status
git status

# Stage all files
git add .

# Verify staged files
git status

# Expected output: all files in green (staged)
```

## Step 5: Create Initial Commit

```bash
# Commit with descriptive message
git commit -m "Initial commit: Consolidated hackathon iterations H1-H4

- H1: Core Real-ESRGAN + COLMAP + 3DGS pipeline
- H2: Kaggle S2Gaussian and alternative rendering approaches
- H3: Extended SfM experiments (MAst3R, mip-splatting, Depth-Anything)
- H4: HPC-optimized with GLOMAP and GSCPR refinement

Includes pipeline scripts, utilities, notebooks, and comprehensive documentation."
```

## Step 6: Set Default Branch & Push

### For New Repository

```bash
# Rename branch to main (GitHub default)
git branch -M main

# Push to GitHub (first time)
git push -u origin main
# -u sets origin/main as default upstream for future pushes

# Or use long form:
git push --set-upstream origin main
```

When prompted for authentication:
- **HTTPS:** Enter your GitHub username and personal access token (not password)
  - Generate token: GitHub → Settings → Developer settings → Personal access tokens
  - Scopes needed: `repo` (full control of private repositories)
  
- **SSH:** Should work automatically if keys are configured

### For Existing Repository

If remote already exists:

```bash
# Check current remote
git remote -v

# Change remote (if needed)
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/hackathon-3dgs.git

# Push
git push -u origin main
```

## Step 7: Verify Push Success

```bash
# Check git log
git log --oneline
# Should show your commit

# Verify remote
git status
# Expected output:
# On branch main
# Your branch is up to date with 'origin/main'.
# nothing to commit, working tree clean
```

Visit your GitHub repository URL to verify files are uploaded!

---

## Complete Command Sequence (Quick Copy-Paste)

For convenience, here's the complete sequence:

```bash
# Navigate to codebase
cd "c:\main\IITM\SEM_4\Modern Computer Vision\HACKATHON_ITERATIONS\hackathon_codebase"

# Initialize git
git init

# Configure user (modify YOUR_NAME and YOUR_EMAIL)
git config user.name "YOUR_NAME"
git config user.email "YOUR_EMAIL@example.com"

# Add remote (modify YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/hackathon-3dgs.git

# Stage all files
git add .

# Commit
git commit -m "Initial commit: Consolidated hackathon iterations H1-H4"

# Set main branch and push
git branch -M main
git push -u origin main
```

---

## Subsequent Pushes (After Initial Setup)

Once the repository is initialized, future updates are simpler:

```bash
# Check status
git status

# Stage changes
git add .

# Commit
git commit -m "Descriptive commit message"

# Push
git push
```

## Common Git Commands

```bash
# View history
git log --oneline -10
git log --graph --oneline --all

# View changes
git diff
git status

# Undo changes
git checkout -- <filename>              # Undo unstaged changes
git reset HEAD <filename>               # Unstage file
git revert <commit-hash>                # Undo commit

# Create new branch
git checkout -b feature/new-approach
git push -u origin feature/new-approach

# View all branches
git branch -a

# Merge branch
git checkout main
git merge feature/new-approach
git push origin main

# Delete branch
git branch -d feature/new-approach
git push origin --delete feature/new-approach
```

## Troubleshooting

### Issue: "fatal: could not read Username"
**Solution:** Use personal access token instead of password for HTTPS

```bash
# Generate at: GitHub → Settings → Developer settings → Personal access tokens → Generate new token
# Scopes: repo (full control)
# Use token as password when prompted
```

### Issue: "Permission denied (publickey)"
**Solution (SSH):** Add SSH key to GitHub or use HTTPS instead

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# Add to GitHub: Settings → SSH and GPG keys → New SSH key
# Paste contents of ~/.ssh/id_ed25519.pub
```

### Issue: "remote origin already exists"
**Solution:** Change remote

```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/hackathon-3dgs.git
```

### Issue: "Everything up-to-date" but files not on GitHub
**Solution:** Verify push was successful

```bash
git log --all
git remote -v
git branch -vv
```

### Issue: Large files causing push to fail
**Solution:** The codebase excludes data folders, so shouldn't be an issue. If needed:

```bash
# Install Git LFS for large files
git lfs install

# Track large files
git lfs track "*.pth"
git lfs track "*.zip"

# Then commit and push
git add .gitattributes
git commit -m "Configure Git LFS"
git push
```

---

## Organizing Repository for Collaboration

### Create Branches for Features

```bash
# Main branch (stable)
# ├── development
#     ├── feature/mast3r-integration
#     ├── feature/gscpr-refinement
#     └── docs/setup-guide

# Create development branch
git checkout -b development
git push -u origin development

# Create feature branches
git checkout -b feature/improve-mast3r
git push -u origin feature/improve-mast3r
```

### Create Releases

```bash
# Tag a release version
git tag -a v1.0 -m "Release: Consolidated H1-H4"
git push origin v1.0

# View tags
git tag
```

---

## Documentation on GitHub

After pushing, add to repository:

### 1. Update README on GitHub
- File exists at root level (README.md)
- GitHub automatically displays it on repository page

### 2. Add Wiki (Optional)
- GitHub → Repo → Wiki tab
- Create pages like: Setup, Pipeline Guide, Troubleshooting

### 3. Add Discussions (Optional)
- GitHub → Repo → Discussions tab
- Enabled for community Q&A

---

## Final Verification Checklist

- [ ] Repository created on GitHub
- [ ] Git initialized locally
- [ ] Remote origin added
- [ ] All files staged (`git add .`)
- [ ] Initial commit created
- [ ] Branch renamed to main
- [ ] Pushed to origin (`git push -u origin main`)
- [ ] Verified on GitHub website
- [ ] README displays correctly
- [ ] File structure visible

---

## Next Steps

After successful push:

1. **Share Repository Link** with team: `https://github.com/YOUR_USERNAME/hackathon-3dgs`

2. **Add Collaborators** (if needed):
   - Settings → Collaborators → Add people
   - Choose access level (Pull request / Triage / Write / Admin)

3. **Enable Discussions** for Q&A:
   - Settings → Features → Enable Discussions

4. **Setup CI/CD** (Optional, for testing):
   - Add GitHub Actions workflows
   - Auto-test on push

5. **Archive for Citation** (Optional):
   - Zenodo integration for DOI
   - GitHub → Settings → Enable Zenodo integration

---

## Questions?

- **GitHub Help:** https://docs.github.com/
- **Git Documentation:** https://git-scm.com/doc
- **SSH Setup:** https://docs.github.com/en/authentication/connecting-to-github-with-ssh
