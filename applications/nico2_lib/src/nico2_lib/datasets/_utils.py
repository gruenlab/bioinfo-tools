from pathlib import Path
import shutil
from typing import Optional
import zipfile
import requests


def github_url_to_raw(url: str) -> str:
    """
    Converts a GitHub file URL to its raw content URL.

    Examples:
    https://github.com/user/repo/blob/main/path/to/file.txt
        -> https://raw.githubusercontent.com/user/repo/main/path/to/file.txt
    """
    if not url.startswith("https://github.com/"):
        raise ValueError("URL must start with 'https://github.com/'")

    parts = url[len("https://github.com/") :].split("/")

    if len(parts) < 5 or parts[2] != "blob":
        raise ValueError("URL does not appear to be a valid GitHub file URL")

    user, repo, _, branch = parts[:4]
    path = "/".join(parts[4:])

    raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
    return raw_url


def download_file(url: str, dest: Path) -> None:
    """Download a file from a URL to a destination path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                shutil.copyfileobj(r.raw, f)
    except Exception as e:
        raise RuntimeError(f"Failed to download file from {url}") from e


def extract_zip(zip_path: Path, extract_to: Path, delete_zip: bool = True) -> None:
    """Extract a ZIP archive to a folder. Optionally delete the ZIP after extraction."""
    extract_to.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_to)
    except Exception as e:
        raise RuntimeError(f"Failed to extract ZIP archive: {zip_path}") from e
    finally:
        if delete_zip and zip_path.exists():
            zip_path.unlink()


def download_and_extract(
    url: str, extract_to: Path, zip_name: str = "download.zip"
) -> None:
    """Helper function: download a ZIP file from a URL and extract it."""
    zip_path = extract_to / zip_name
    download_file(url, zip_path)
    extract_zip(zip_path, extract_to)


def ensure_dataset_dir(name: str, dir: Optional[str] = None):
    """Ensure the dataset folder exists and return dataset path and .h5ad path."""
    data_dir = Path(dir) if dir else Path.cwd()
    dataset_path = data_dir / name
    dataset_path.mkdir(exist_ok=True, parents=True)
    anndata_path = dataset_path / f"{name}.h5ad"
    return dataset_path, anndata_path


def cleanup(raw_data_path: Path) -> None:
    """Remove the raw data folder if it exists."""
    if raw_data_path.exists():
        shutil.rmtree(raw_data_path)
