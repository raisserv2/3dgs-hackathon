import argparse
import copy
import csv
from pathlib import Path

import pycolmap

try:
    from hloc import extract_features, match_features, pairs_from_exhaustive, reconstruction
except ImportError as exc:
    raise SystemExit(
        "hloc is not installed. Install dependencies first: pip install -r requirements_hloc.txt"
    ) from exc


DEFAULT_ROOT = Path("data-given-3dgs")
DEFAULT_OUT_ROOT = Path("group_b_hloc_colmap")
DEFAULT_SCENES = ["aeroplane", "cycle", "face", "still3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate COLMAP camera poses using hloc features/matching.")
    parser.add_argument("--data_root", type=Path, default=DEFAULT_ROOT, help="Dataset root containing scene folders.")
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT, help="Output root for hloc/COLMAP artifacts.")
    parser.add_argument(
        "--scenes",
        type=str,
        default=",".join(DEFAULT_SCENES),
        help="Comma-separated scene names, e.g. aeroplane,cycle,face,still3",
    )
    parser.add_argument(
        "--feature_conf",
        type=str,
        default="aliked-n32",
        help="hloc feature configuration to use, e.g. aliked-n32 or superpoint_aachen",
    )
    parser.add_argument(
        "--matcher_conf",
        type=str,
        default="aliked+lightglue",
        help="hloc matcher configuration to use, e.g. aliked+lightglue or superglue",
    )
    parser.add_argument(
        "--resize_maxes",
        type=str,
        default="3072",
        help="Comma-separated image resize_max values to try in order for the highest-quality A100-safe run.",
    )
    parser.add_argument(
        "--enable_fallback",
        action="store_true",
        help="Add a SuperPoint/SuperGlue fallback pass after the ALIKED passes.",
    )
    parser.add_argument(
        "--hard_scenes",
        type=str,
        default="aeroplane,face",
        help="Comma-separated scenes that should get extra aggressive tuning passes.",
    )
    return parser.parse_args()


def resolve_image_dir(scene_dir: Path) -> Path:
    images_subdir = scene_dir / "images"
    return images_subdir if images_subdir.exists() else scene_dir


def collect_image_list(image_dir: Path) -> list[str]:
    valid_ext = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    image_paths = [
        p.relative_to(image_dir).as_posix()
        for p in sorted(image_dir.rglob("*"))
        if p.is_file() and p.suffix in valid_ext
    ]
    return image_paths


def estimate_pass_memory_gb(feature_conf: str, resize_max: int, n_images: int) -> str:
    scale = (resize_max / 1024.0) ** 2
    if feature_conf.startswith("aliked"):
        base = 4.5
        per_image = 0.04
    else:
        base = 3.0
        per_image = 0.03

    feature_peak = base * scale
    matching_peak = (base + 2.5) * scale
    working_set = per_image * n_images * scale
    total_low = feature_peak + working_set
    total_high = matching_peak + working_set + 2.0
    return f"~{total_low:.1f}-{total_high:.1f} GB peak GPU memory"


def resolve_hloc_config(
    requested: str,
    available: dict,
    fallbacks: list[str],
    config_kind: str,
) -> str:
    if requested in available:
        return requested

    for candidate in fallbacks:
        if candidate in available:
            print(
                f"  ! Requested {config_kind} config '{requested}' is not available. "
                f"Falling back to '{candidate}'."
            )
            return candidate

    available_names = ", ".join(sorted(available.keys()))
    raise KeyError(
        f"Unknown hloc {config_kind} config: {requested}. "
        f"Available {config_kind} configs: {available_names}"
    )


def run_reconstruction_with_fallbacks(
    sfm_dir: Path,
    image_dir: Path,
    sfm_pairs: Path,
    features_path: Path,
    matches_path: Path,
    image_list: list[str],
) -> None:
    mapper_options = {
        "min_model_size": 3,
        "ba_refine_focal_length": True,
        "ba_refine_principal_point": True,
        "ba_refine_extra_params": True,
    }
    attempts = [
        {
            "camera_mode": pycolmap.CameraMode.SINGLE,
            "mapper_options": mapper_options,
        },
        {
            "mapper_options": mapper_options,
        },
        {},
    ]

    last_error = None
    for extra_kwargs in attempts:
        try:
            reconstruction.main(
                sfm_dir,
                image_dir,
                sfm_pairs,
                features_path,
                matches_path,
                image_list=image_list,
                **extra_kwargs,
            )
            return
        except TypeError as err:
            last_error = err
            continue

    if last_error is not None:
        raise last_error


def load_reconstruction(sfm_dir: Path) -> pycolmap.Reconstruction:
    for candidate in (sfm_dir / "0", sfm_dir):
        if (candidate / "images.bin").exists() or (candidate / "images.txt").exists():
            return pycolmap.Reconstruction(str(candidate))
    raise FileNotFoundError(f"No COLMAP model found under: {sfm_dir}")


def export_poses_csv(recon: pycolmap.Reconstruction, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "qw", "qx", "qy", "qz", "tx", "ty", "tz", "camera_id"])
        for img in sorted(recon.images.values(), key=lambda x: x.name):
            if hasattr(img, "qvec"):
                qvec = img.qvec
            else:
                cam_from_world = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
                rotation = (
                    cam_from_world.rotation() if callable(cam_from_world.rotation) else cam_from_world.rotation
                )
                qvec = rotation.quat() if callable(rotation.quat) else rotation.quat

            if hasattr(img, "tvec"):
                tvec = img.tvec
            else:
                cam_from_world = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
                tvec = (
                    cam_from_world.translation()
                    if callable(cam_from_world.translation)
                    else cam_from_world.translation
                )

            writer.writerow([img.name, *qvec, *tvec, img.camera_id])


def main() -> None:
    args = parse_args()
    scene_names = [s.strip() for s in args.scenes.split(",") if s.strip()]
    hard_scene_names = {s.strip() for s in args.hard_scenes.split(",") if s.strip()}
    resize_maxes = [int(value.strip()) for value in args.resize_maxes.split(",") if value.strip()]

    args.out_root.mkdir(parents=True, exist_ok=True)
    print("Scenes to process:", scene_names)

    feature_conf_name = resolve_hloc_config(
        requested=args.feature_conf,
        available=extract_features.confs,
        fallbacks=["aliked-n32", "aliked-n16", "superpoint_max", "superpoint_aachen"],
        config_kind="feature",
    )
    matcher_conf_name = resolve_hloc_config(
        requested=args.matcher_conf,
        available=match_features.confs,
        fallbacks=["aliked+lightglue", "superpoint+lightglue", "superglue"],
        config_kind="matcher",
    )

    print(f"Using feature config: {feature_conf_name}")
    print(f"Using matcher config: {matcher_conf_name}")

    # Multi-pass settings: keep the best reconstruction for each scene.
    # On the A100 we bias toward the strongest learned front end and try the
    # largest practical image scales first.
    pass_settings = [
        {
            "name": f"{feature_conf_name}_r{resize_max}",
            "feature_conf": feature_conf_name,
            "matcher_conf": matcher_conf_name,
            "resize_max": resize_max,
        }
        for resize_max in resize_maxes
    ]

    if args.enable_fallback:
        pass_settings.append(
            {
                "name": "superpoint_aachen_r1024",
                "feature_conf": "superpoint_aachen",
                "matcher_conf": "superglue",
                "resize_max": 1024,
            }
        )

    hard_scene_pass_settings = [
        {
            "name": f"superpoint_max_r{resize_max}",
            "feature_conf": "superpoint_max",
            "matcher_conf": matcher_conf_name,
            "resize_max": resize_max,
        }
        for resize_max in (1024, 1536, 2048)
        if "superpoint_max" in extract_features.confs
    ]

    if "superpoint+lightglue" in match_features.confs:
        for resize_max in (1024, 1536, 2048):
            hard_scene_pass_settings.append(
                {
                    "name": f"superpoint_max_lightglue_r{resize_max}",
                    "feature_conf": "superpoint_max",
                    "matcher_conf": "superpoint+lightglue",
                    "resize_max": resize_max,
                }
            )

    for scene in scene_names:
        scene_dir = args.data_root / scene
        image_dir = resolve_image_dir(scene_dir)
        scene_out_dir = args.out_root / scene
        scene_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Processing {scene} ===")
        if not image_dir.exists():
            print(f"  ✗ Image directory not found: {image_dir}")
            continue

        image_list = collect_image_list(image_dir)
        if not image_list:
            print(f"  ✗ No images found in: {image_dir}")
            continue

        print(f"  Scene image count: {len(image_list)}")

        scene_pass_settings = list(pass_settings)
        if scene in hard_scene_names:
            print("  Hard-scene tuning enabled: adding extra fallback passes")
            scene_pass_settings.extend(hard_scene_pass_settings)

        best = None

        for idx, setting in enumerate(scene_pass_settings, start=1):
            pass_name = setting["name"]
            sfm_pairs = scene_out_dir / f"pairs-sfm-{pass_name}.txt"
            features_path = scene_out_dir / f"features-{pass_name}.h5"
            matches_path = scene_out_dir / f"matches-{pass_name}.h5"
            sfm_dir = scene_out_dir / f"sfm-{pass_name}"

            feature_conf = copy.deepcopy(extract_features.confs[setting["feature_conf"]])
            feature_conf["preprocessing"]["resize_max"] = setting["resize_max"]

            matcher_conf = copy.deepcopy(match_features.confs[setting["matcher_conf"]])

            try:
                mem_estimate = estimate_pass_memory_gb(setting["feature_conf"], setting["resize_max"], len(image_list))
                print(
                    f"  Pass {idx}: {pass_name} | feature={setting['feature_conf']} | "
                    f"matcher={setting['matcher_conf']} | resize={setting['resize_max']} | {mem_estimate}"
                )
                print(f"    1) Extracting hloc features ({setting['feature_conf']})...")
                extract_features.main(feature_conf, image_dir, image_list=image_list, feature_path=features_path)

                print("    2) Building image pairs (exhaustive)...")
                pairs_from_exhaustive.main(sfm_pairs, image_list=image_list)

                print(f"    3) Matching pairs with hloc ({setting['matcher_conf']})...")
                match_features.main(matcher_conf, sfm_pairs, features=features_path, matches=matches_path)

                print("    4) Running COLMAP incremental mapping via hloc...")
                run_reconstruction_with_fallbacks(
                    sfm_dir=sfm_dir,
                    image_dir=image_dir,
                    sfm_pairs=sfm_pairs,
                    features_path=features_path,
                    matches_path=matches_path,
                    image_list=image_list,
                )

                recon = load_reconstruction(sfm_dir)
                n_registered = len(recon.images)
                n_points = len(recon.points3D)
                print(f"    ✓ Pass result: registered={n_registered}/{len(image_list)}, points3D={n_points}")

                if best is None or n_registered > best["n_registered"] or (
                    n_registered == best["n_registered"] and n_points > best["n_points"]
                ):
                    best = {
                        "pass_name": pass_name,
                        "recon": recon,
                        "n_registered": n_registered,
                        "n_points": n_points,
                    }
            except Exception as e:
                print(f"    ✗ Pass failed: {e}")

        if best is None:
            print("  ✗ FAILED: all reconstruction passes failed for this scene.")
            continue

        recon = best["recon"]
        sparse_model_dir = scene_out_dir / "sparse" / "0"
        sparse_model_dir.mkdir(parents=True, exist_ok=True)
        recon.write(str(sparse_model_dir))

        n_registered = best["n_registered"]
        n_points = best["n_points"]
        n_total = len(image_list)
        n_missing = n_total - n_registered
        coverage = n_registered / n_total if n_total else 0.0

        poses_csv = scene_out_dir / "poses.csv"
        export_poses_csv(recon, poses_csv)

        print(
            f"  ✓ BEST ({best['pass_name']}): registered={n_registered}/{n_total}, "
            f"points3D={n_points}, missing={n_missing}"
        )
        print(f"  ✓ COLMAP sparse model saved to: {sparse_model_dir}")
        print(f"  ✓ Camera poses saved to: {poses_csv}")
        if coverage < 0.7:
            print("  ! Low coverage detected. Consider removing blurred frames or running scene-specific tuning.")


if __name__ == "__main__":
    main()