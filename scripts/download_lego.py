#!/usr/bin/env python3
"""Download the NeRF-synthetic Lego scene from the official bmild/nerf data.

Source: https://github.com/bmild/nerf README → Data section
Drive folder: https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4

Usage:
    python scripts/download_lego.py             # download + verify
    python scripts/download_lego.py --verify    # verify only (skip download)
    python scripts/download_lego.py --dest /my/custom/path  # custom destination
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Official Drive folder from bmild/nerf README:
DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4"
)
# File name we are looking for inside the folder listing:
TARGET_ZIP_NAME = "nerf_synthetic.zip"

# Approximate size of nerf_synthetic.zip (for user information)
APPROX_ZIP_SIZE_MB = 700
# Approximate unzipped lego/ directory size
APPROX_LEGO_SIZE_MB = 180


def _repo_root() -> Path:
    """Walk up from this script to find the repo root (contains pyproject.toml)."""
    d = Path(__file__).resolve().parent
    while d != d.parent:
        if (d / "pyproject.toml").exists():
            return d
        d = d.parent
    raise RuntimeError("Could not find repo root")


# ---------------------------------------------------------------------------
# Verify logic (importable for testing)
# ---------------------------------------------------------------------------

def verify_lego_dataset(
    lego_dir: Path,
    *,
    expected_image_size: tuple[int, int] = (800, 800),
) -> list[dict[str, Any]]:
    """Verify the Lego dataset at *lego_dir*.

    Args:
        lego_dir: Root of the lego scene directory.
        expected_image_size: (width, height) to check for the sample image.
            Defaults to (800, 800) for the real dataset.

    Returns a list of check results, each a dict with keys:
        - check: str  -- human-readable description
        - expected: str
        - status: "PASS" | "FAIL"
        - detail: str (optional, on failure)
    """
    results: list[dict[str, Any]] = []

    def _add(check: str, expected: str, passed: bool, detail: str = "") -> None:
        results.append({
            "check": check,
            "expected": expected,
            "status": "PASS" if passed else "FAIL",
            **({"detail": detail} if detail else {}),
        })

    # 1-3: JSON files exist
    split_jsons = {
        "transforms_train.json": None,
        "transforms_val.json": None,
        "transforms_test.json": None,
    }
    for name in split_jsons:
        path = lego_dir / name
        exists = path.exists()
        _add(f"{name} exists", "exists", exists,
             "" if exists else f"not found at {path}")
        if exists:
            with open(path, "r", encoding="utf-8") as f:
                split_jsons[name] = json.load(f)

    # 4-6: Frame counts
    expected_counts = {
        "transforms_train.json": 100,
        "transforms_val.json": 100,
        "transforms_test.json": 200,
    }
    for name, expected in expected_counts.items():
        data = split_jsons.get(name)
        if data is None:
            _add(f"{name} frame count", str(expected), False,
                 "JSON not loaded")
            continue
        frames = data.get("frames", [])
        actual = len(frames)
        _add(f"{name} frame count", str(expected), actual == expected,
             f"got {actual}")

    # 7: Every file_path resolves to a PNG
    all_resolve = True
    missing_files: list[str] = []
    for name, data in split_jsons.items():
        if data is None:
            all_resolve = False
            continue
        for frame in data.get("frames", []):
            fp = frame.get("file_path", "")
            # file_path is like "./train/r_000" — append .png
            img_path = lego_dir / (fp + ".png") if not fp.endswith(".png") else lego_dir / fp
            if not img_path.exists():
                all_resolve = False
                if len(missing_files) < 5:
                    missing_files.append(str(img_path))
    detail = ""
    if missing_files:
        detail = f"missing (first 5): {missing_files}"
    _add("All file_path PNGs resolve", "all exist", all_resolve, detail)

    # 8: camera_angle_x present in all JSONs
    cax_ok = True
    for name, data in split_jsons.items():
        if data is None or "camera_angle_x" not in data:
            cax_ok = False
    _add("camera_angle_x in all JSONs", "present", cax_ok)

    # 9: Sample image loads as RGBA at expected size
    exp_w, exp_h = expected_image_size
    size_label = f"{exp_w}x{exp_h}"
    sample_ok = False
    sample_detail = ""
    try:
        from PIL import Image
        # Use the first train frame
        train_data = split_jsons.get("transforms_train.json")
        if train_data and train_data.get("frames"):
            fp = train_data["frames"][0].get("file_path", "")
            img_path = lego_dir / (fp + ".png") if not fp.endswith(".png") else lego_dir / fp
            img = Image.open(img_path)
            w, h = img.size
            mode = img.mode
            if w == exp_w and h == exp_h and mode == "RGBA":
                sample_ok = True
            else:
                sample_detail = f"got {w}x{h} {mode}"
        else:
            sample_detail = "no train frames to test"
    except ImportError:
        sample_detail = "Pillow not installed"
    except Exception as e:
        sample_detail = str(e)
    _add(f"Sample image RGBA {size_label}", f"{size_label} RGBA", sample_ok, sample_detail)

    return results


def print_verify_table(results: list[dict[str, Any]]) -> None:
    """Print a formatted pass/fail table."""
    max_check = max(len(r["check"]) for r in results)
    max_exp = max(len(r["expected"]) for r in results)
    header = f"{'Check':<{max_check}}  {'Expected':<{max_exp}}  Status  Detail"
    print(header)
    print("-" * len(header))
    for r in results:
        detail = r.get("detail", "")
        print(f"{r['check']:<{max_check}}  {r['expected']:<{max_exp}}  {r['status']:<6}  {detail}")


# ---------------------------------------------------------------------------
# Download logic
# ---------------------------------------------------------------------------

def _get_file_id_from_folder() -> str | None:
    """Query the official Drive folder to find the file ID for nerf_synthetic.zip.

    Uses gdown.download_folder(..., skip_download=True) to list files
    without downloading them.  Returns the ID or None on failure.
    """
    try:
        import gdown  # type: ignore[import-untyped]
    except ImportError:
        print("ERROR: gdown is not installed.  pip install gdown")
        return None

    try:
        files = gdown.download_folder(
            DRIVE_FOLDER_URL, skip_download=True
        )
    except Exception as e:
        print(f"ERROR: Could not list Drive folder: {e}")
        return None

    if files is None:
        print("ERROR: gdown returned None for folder listing.")
        return None

    for f in files:
        # gdown returns objects with .path and .id attributes
        fname = getattr(f, "path", "") or ""
        fid = getattr(f, "id", "") or ""
        if fname == TARGET_ZIP_NAME or fname.endswith("/" + TARGET_ZIP_NAME):
            return fid

    print(f"ERROR: '{TARGET_ZIP_NAME}' not found in folder listing.")
    return None


def download_lego(dest: Path) -> bool:
    """Download and extract the Lego scene into *dest*.

    Returns True on success, False on failure.
    """
    marker = dest / "transforms_train.json"
    if marker.exists():
        print(f"Lego data already present at {dest} -- skipping download.")
        return True

    print("=" * 60)
    print("NeRF-synthetic Lego dataset download")
    print("=" * 60)
    print(f"  Destination : {dest}")
    print(f"  Zip size    : ~{APPROX_ZIP_SIZE_MB} MB (download)")
    print(f"  Lego on disk: ~{APPROX_LEGO_SIZE_MB} MB (after extraction)")
    print(f"  Peak disk   : ~{APPROX_ZIP_SIZE_MB + APPROX_LEGO_SIZE_MB} MB (during extraction)")
    print()

    # Step 1: get file ID from folder listing
    print("[1/4] Querying Drive folder for nerf_synthetic.zip file ID ...")
    file_id = _get_file_id_from_folder()
    if file_id is None:
        _print_manual_steps(dest)
        return False

    print(f"       Found file ID: {file_id}")

    # Step 2: download zip
    try:
        import gdown  # type: ignore[import-untyped]
    except ImportError:
        print("ERROR: gdown not installed.")
        _print_manual_steps(dest)
        return False

    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest.parent / TARGET_ZIP_NAME

    print(f"[2/4] Downloading {TARGET_ZIP_NAME} (~{APPROX_ZIP_SIZE_MB} MB) ...")
    try:
        result = gdown.download(id=file_id, output=str(zip_path), quiet=False)
    except Exception as e:
        print(f"ERROR: Download failed: {e}")
        _print_manual_steps(dest)
        return False

    if result is None or not zip_path.exists():
        print("ERROR: Download failed (gdown returned None or file missing).")
        _print_manual_steps(dest)
        return False

    # Step 3: extract only lego/
    print("[3/4] Extracting lego/ from zip ...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            lego_members = [m for m in zf.namelist() if m.startswith("lego/")]
            if not lego_members:
                # Some zips nest under nerf_synthetic/lego/
                lego_members = [m for m in zf.namelist()
                                if "/lego/" in m or m.startswith("nerf_synthetic/lego/")]
            if not lego_members:
                print("ERROR: Could not find lego/ directory in zip.")
                print("       Zip contents (first 20):")
                for name in zf.namelist()[:20]:
                    print(f"         {name}")
                _print_manual_steps(dest)
                return False

            # Determine the prefix to strip (e.g. "" or "nerf_synthetic/")
            first = lego_members[0]
            if first.startswith("lego/"):
                prefix = ""
            else:
                prefix = first.split("lego/")[0]  # e.g. "nerf_synthetic/"

            for member in lego_members:
                # Strip prefix to get relative path under lego/
                rel = member[len(prefix):]  # e.g. "lego/train/r_000.png"
                if not rel:
                    continue
                target = dest.parent / rel
                if member.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())

        print(f"       Extracted {len(lego_members)} entries -> {dest}")
    except Exception as e:
        print(f"ERROR: Extraction failed: {e}")
        _print_manual_steps(dest)
        return False

    # Step 4: delete zip
    print(f"[4/4] Deleting {TARGET_ZIP_NAME} to save disk space ...")
    try:
        zip_path.unlink()
        print("       Deleted.")
    except Exception as e:
        print(f"       WARNING: Could not delete zip: {e}")

    return True


def _print_manual_steps(dest: Path) -> None:
    """Print manual download instructions for the user."""
    print()
    print("=" * 60)
    print("MANUAL DOWNLOAD STEPS")
    print("=" * 60)
    print()
    print("1. Open this URL in your browser:")
    print(f"   {DRIVE_FOLDER_URL}")
    print()
    print(f"2. Find '{TARGET_ZIP_NAME}' and download it.")
    print(f"   (Right-click -> Download, or select and click the download icon)")
    print()
    print(f"3. Extract ONLY the 'lego/' folder from the zip into:")
    print(f"   {dest}")
    print()
    print("   On Windows (PowerShell):")
    print(f"     Expand-Archive -Path nerf_synthetic.zip -DestinationPath temp_extract")
    print(f"     Move-Item temp_extract\\lego {dest}")
    print(f"     Remove-Item -Recurse temp_extract")
    print()
    print("   On Linux/macOS:")
    print(f"     unzip nerf_synthetic.zip 'lego/*' -d {dest.parent}")
    print()
    print("4. The final folder tree must look like:")
    print(f"   {dest}/")
    print("   +-- transforms_train.json")
    print("   +-- transforms_val.json")
    print("   +-- transforms_test.json")
    print("   +-- train/")
    print("   |   +-- r_0.png")
    print("   |   +-- r_1.png")
    print("   |   +-- ... (100 images)")
    print("   +-- val/")
    print("   |   +-- ... (100 images)")
    print("   +-- test/")
    print("       +-- ... (200 images)")
    print()
    print("5. Then run:  python scripts/download_lego.py --verify")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download / verify the NeRF-synthetic Lego dataset."
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Only verify existing data (skip download).",
    )
    parser.add_argument(
        "--dest", type=str, default=None,
        help="Destination directory for lego/ data. "
             "Default: <repo_root>/datasets/nerf_synthetic/lego",
    )
    args = parser.parse_args()

    if args.dest:
        dest = Path(args.dest)
    else:
        dest = _repo_root() / "datasets" / "nerf_synthetic" / "lego"

    if not args.verify:
        ok = download_lego(dest)
        if not ok:
            sys.exit(1)

    # Always verify after download (or when --verify is passed)
    print()
    print("Verifying dataset ...")
    results = verify_lego_dataset(dest)
    print()
    print_verify_table(results)
    print()

    failures = [r for r in results if r["status"] == "FAIL"]
    if failures:
        print(f"RESULT: {len(failures)} check(s) FAILED.")
        sys.exit(1)
    else:
        print("RESULT: All checks PASSED.")


if __name__ == "__main__":
    main()
