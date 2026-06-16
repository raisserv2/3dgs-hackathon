#!/usr/bin/env python3
"""
Generic GSCPR refinement wrapper.

This script does not implement GSCPR itself. It consumes the MASt3R output
manifest written by run_mast3r_groupb_final.py and launches an external GSCPR
command with a stable set of environment variables and path arguments.

Use this when the GSCPR repo is present on the HPC server but its exact CLI
is kept outside this workspace.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch downstream GSCPR refinement from a MASt3R manifest")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Path to gscpr_manifest.json created by the MASt3R stage.")
    parser.add_argument("--command", type=str, required=True,
                        help="External GSCPR command to execute. The sparse dir is appended as the last argument.")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    sparse_dir = Path(manifest["sparse_dir"])
    out_dir = Path(manifest["out_dir"])

    env = os.environ.copy()
    env["MAST3R_MANIFEST"] = str(manifest_path)
    env["MAST3R_SPARSE_DIR"] = str(sparse_dir)
    env["MAST3R_OUTPUT_DIR"] = str(out_dir)
    env["MAST3R_IMAGES_DIR"] = str(manifest["images_dir"])
    env["MAST3R_POINTCLOUD"] = str(manifest["pointcloud_ply"])

    command = f'{args.command} "{sparse_dir}"'
    print(f"Manifest: {manifest_path}")
    print(f"Running: {command}")

    if args.dry_run:
        return

    subprocess.run(command, shell=True, check=True, env=env)


if __name__ == "__main__":
    main()