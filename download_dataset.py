#!/usr/bin/env python3
import argparse
import os
import sys
import zipfile

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

DATASET_URL = "https://public-bucket-with-cards.s3.us-east-1.amazonaws.com/opencv-pytorch-segmentation-project.zip"


def download_dataset(url, output_file):
    if os.path.exists(output_file):
        print("Dataset archive already exists:", output_file)
        return output_file

    print("Downloading dataset from:")
    print(url)
    try:
        urlretrieve(url, output_file)
    except Exception as exc:
        raise RuntimeError("Download failed: {}".format(exc))

    print("Download complete:", output_file)
    return output_file


def extract_dataset(zip_path, extract_to):
    if not os.path.exists(zip_path):
        raise FileNotFoundError("ZIP file not found: {}".format(zip_path))

    print("Extracting dataset to:", extract_to)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_to)
    print("Extraction complete.")


def validate_dataset_root(root_path):
    required_dirs = [
        os.path.join(root_path, "imgs", "imgs"),
        os.path.join(root_path, "masks", "masks"),
    ]
    missing = [p for p in required_dirs if not os.path.isdir(p)]
    if missing:
        return False, missing
    return True, []


def main():
    parser = argparse.ArgumentParser(description="Download and extract the OpenCV PyTorch segmentation dataset.")
    parser.add_argument("--data-dir", default="data", help="Base directory to store the dataset archive and extracted files")
    parser.add_argument("--url", default=DATASET_URL, help="URL of the dataset ZIP archive")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading if the archive already exists")
    parser.add_argument("--skip-extract", action="store_true", help="Skip extraction if the dataset is already extracted")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    archive_name = os.path.basename(args.url)
    archive_path = os.path.join(data_dir, archive_name)

    if not args.skip_download:
        download_dataset(args.url, archive_path)
    else:
        if not os.path.exists(archive_path):
            raise FileNotFoundError("Archive not found, cannot skip download: {}".format(archive_path))

    extracted_root = os.path.join(data_dir, "opencv-pytorch-segmentation-project")
    if not args.skip_extract:
        if os.path.isdir(extracted_root):
            print("Dataset already extracted:", extracted_root)
        else:
            extract_dataset(archive_path, data_dir)
    else:
        if not os.path.isdir(extracted_root):
            raise FileNotFoundError("Dataset not extracted, cannot skip extraction: {}".format(extracted_root))

    valid, missing = validate_dataset_root(extracted_root)
    if not valid:
        print("Warning: expected dataset folders are missing:")
        for path in missing:
            print("  ", path)
        sys.exit(1)

    print("Dataset ready at:", extracted_root)
    print("Expected image folder:", os.path.join(extracted_root, "imgs", "imgs"))
    print("Expected mask folder:", os.path.join(extracted_root, "masks", "masks"))


if __name__ == "__main__":
    main()
