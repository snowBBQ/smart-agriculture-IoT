import os
import sys
import requests
from tqdm import tqdm

# ==========================================
# DOWNLOAD CONFIGURATION (CONFIGURE AS NEEDED)
# ==========================================

# Target file name and path where weights will be saved
OUTPUT_FILE = "best.onnx"

# Option 1: Google Drive File ID
# Example link: https://drive.google.com/file/d/1A2b3C4d5E6f7G8h9I/view?usp=sharing
# File ID from the link above is "1A2b3C4d5E6f7G8h9I"
GDRIVE_FILE_ID = "1A2b3C4d5E6f7G8h9I"  # Replace with your File ID

# Option 2: Direct URL (GitHub Releases, AWS S3, etc.)
DIRECT_URL = "https://github.com/YOUR_USERNAME/YOUR_REPO/releases/download/v1.0.0/best.onnx"

# Default source selection: "gdrive" or "release"
DOWNLOAD_SOURCE = "gdrive"


def download_from_gdrive(file_id: str, output_path: str):
    """Download model weights from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        print("❌ The 'gdown' library is not installed.")
        print("   Install it using: pip install gdown")
        sys.exit(1)

    print(f"📥 Starting download of {output_path} from Google Drive (ID: {file_id})...")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)
    
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ File successfully saved to '{output_path}' ({file_size_mb:.2f} MB)")
    else:
        print("❌ Error downloading file from Google Drive. Please check public share permissions.")


def download_file_direct(url: str, output_path: str):
    """Download model weights via direct URL (GitHub Releases / S3) with a tqdm progress bar."""
    print(f"📥 Starting download of {output_path} from URL: {url}")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Connection error: {e}")
        sys.exit(1)

    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 8  # 8 KB chunks

    with open(output_path, 'wb') as file, tqdm(
        desc=output_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

    print(f"✅ File successfully saved to '{output_path}'")


def main():
    # Check if the file already exists locally
    if os.path.exists(OUTPUT_FILE):
        file_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
        print(f"ℹ️  File '{OUTPUT_FILE}' already exists ({file_size:.2f} MB). Skipping download.")
        return

    # Trigger download based on chosen source
    if DOWNLOAD_SOURCE == "gdrive":
        if GDRIVE_FILE_ID == "1A2b3C4d5E6f7G8h9I":
            print("⚠️  Warning: Please specify a valid GDRIVE_FILE_ID at the top of the script!")
            return
        download_from_gdrive(GDRIVE_FILE_ID, OUTPUT_FILE)

    elif DOWNLOAD_SOURCE == "release":
        if "YOUR_USERNAME" in DIRECT_URL:
            print("⚠️  Warning: Please specify a valid DIRECT_URL at the top of the script!")
            return
        download_file_direct(DIRECT_URL, OUTPUT_FILE)
    else:
        print(f"❌ Unknown DOWNLOAD_SOURCE='{DOWNLOAD_SOURCE}'. Use 'gdrive' or 'release'.")


if __name__ == "__main__":
    main()