"""
download_dataset.py — CrowdGuard AI
=====================================
Helper script to download the UCSD Anomaly Dataset from Kaggle.

Prerequisites:
  1. Create a Kaggle account at https://www.kaggle.com
  2. Go to Account → API → Create New Token
  3. Place the downloaded kaggle.json at:
     C:\\Users\\<your-user>\\.kaggle\\kaggle.json

Run:
    python scripts/download_dataset.py

If Kaggle API is not available, instructions for manual download are printed.
"""

import os
import sys
import zipfile
from pathlib import Path

_SCRIPTS = Path(__file__).parent
_ROOT = _SCRIPTS.parent
DATA_DIR = _ROOT / "data" / "raw" / "ucsd"


def check_kaggle_available():
    try:
        import kaggle  # noqa
        return True
    except (ImportError, OSError):
        return False


def check_kaggle_token():
    token_path = Path.home() / ".kaggle" / "kaggle.json"
    return token_path.exists()


def download_via_kaggle():
    import kaggle
    print("Downloading UCSD Anomaly Dataset from Kaggle...")
    os.makedirs(DATA_DIR, exist_ok=True)
    download_dir = _ROOT / "data" / "raw"

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        "orvile/ucsd-anomaly-dataset",
        path=str(download_dir),
        unzip=False,
    )

    zip_path = download_dir / "ucsd-anomaly-dataset.zip"
    if zip_path.exists():
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATA_DIR)
        print(f"Extracted to: {DATA_DIR}")
        zip_path.unlink()
        print("Download complete.")
    else:
        print("ZIP file not found after download. Check Kaggle API.")


def print_manual_instructions():
    print("""
============================================================
  MANUAL DATASET DOWNLOAD INSTRUCTIONS
============================================================

The Kaggle API token is not configured or kaggle package
is not available. Please download the dataset manually:

1. Visit: https://www.kaggle.com/datasets/orvile/ucsd-anomaly-dataset
2. Click the 'Download' button (requires free Kaggle account)
3. Save the downloaded ZIP file
4. Extract it to the following directory:

   model_training/data/raw/ucsd/

After extraction, the structure should look like:

   data/raw/ucsd/
     UCSDped1/
       Train/
         Train001/     <- .tif frame images
         Train001_gt   <- frame-level labels
         ...
       Test/
         ...
     UCSDped2/
       Train/...
       Test/...

5. Then run: python scripts/inspect_dataset.py

============================================================
""")


def verify_structure():
    """Quick check that dataset was extracted correctly."""
    expected_dirs = [
        DATA_DIR / "UCSDped1" / "Train",
        DATA_DIR / "UCSDped1" / "Test",
        DATA_DIR / "UCSDped2" / "Train",
        DATA_DIR / "UCSDped2" / "Test",
    ]
    found = [d for d in expected_dirs if d.exists()]
    return len(found) >= 2  # at least 2 of 4 expected dirs


def main():
    print("\nCrowdGuard AI — Dataset Downloader")
    print("=" * 40)

    if DATA_DIR.exists() and verify_structure():
        print(f"Dataset already exists at: {DATA_DIR}")
        print("Run: python scripts/inspect_dataset.py")
        return

    if check_kaggle_available() and check_kaggle_token():
        download_via_kaggle()
        if verify_structure():
            print("\nDataset structure verified. Next step:")
            print("  python scripts/inspect_dataset.py")
        else:
            print("\nWarning: Expected directory structure not found after extraction.")
            print("Please verify the extracted contents in:", DATA_DIR)
    else:
        if not check_kaggle_available():
            print("Note: kaggle package not available (install with: pip install kaggle)")
        if not check_kaggle_token():
            print("Note: kaggle.json API token not found at ~/.kaggle/kaggle.json")
        print_manual_instructions()


if __name__ == "__main__":
    main()
