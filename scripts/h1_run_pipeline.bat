@echo off
REM ============================================================
REM  Master Pipeline: 3DGS Super-Resolution Baseline
REM  Run from: C:\main\IITM\SEM_4\Modern Computer Vision\Hackathon
REM ============================================================
REM
REM  USAGE:
REM    run_pipeline.bat                  Full pipeline (Real-ESRGAN SR)
REM    run_pipeline.bat bicubic          Fast test with bicubic SR
REM    run_pipeline.bat fast             7k iterations, bicubic (debug run)
REM
REM  PREREQUISITES:
REM    - conda activate 3dgs  (your 3DGS env)
REM    - COLMAP in PATH (or edit COLMAP_PATH below)
REM    - gaussian-splatting/ compiled and working
REM ============================================================

SET METHOD=realesrgan
SET ITERS=30000
SET FAST=0

IF "%1"=="bicubic" SET METHOD=bicubic
IF "%1"=="fast" (
    SET METHOD=bicubic
    SET ITERS=7000
    SET FAST=1
)

echo.
echo ====================================================
echo  3DGS Super-Resolution Pipeline
echo  SR Method : %METHOD%
echo  Iterations: %ITERS%
echo ====================================================
echo.

REM ── Step 1: Upsample all images 4x ──────────────────────
echo [STEP 1] Super-Resolution Upsampling...
python step1_upsample.py --method %METHOD%
IF %ERRORLEVEL% NEQ 0 (echo [ERROR] Step 1 failed & exit /b 1)

REM ── Step 2: COLMAP for missing scenes ───────────────────
echo.
echo [STEP 2] COLMAP + Scene Preparation...
python step2_colmap.py --colmap_path "colmap-x64-windows-cuda\bin\colmap.exe"
IF %ERRORLEVEL% NEQ 0 (echo [ERROR] Step 2 failed & exit /b 1)

REM ── Step 3: Train 3DGS ──────────────────────────────────
echo.
echo [STEP 3] Training 3D Gaussian Splatting...
python step3_train_3dgs.py --iterations %ITERS%
IF %ERRORLEVEL% NEQ 0 (echo [ERROR] Step 3 failed & exit /b 1)

REM ── Step 4: Render + Submit ──────────────────────────────
echo.
echo [STEP 4] Rendering Test Views + Packaging Submission...
python step4_render_submit.py
IF %ERRORLEVEL% NEQ 0 (echo [ERROR] Step 4 failed & exit /b 1)

echo.
echo ====================================================
echo  DONE! Upload submission/submission.csv to Kaggle.
echo ====================================================
