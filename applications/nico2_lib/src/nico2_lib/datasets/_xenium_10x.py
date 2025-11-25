"""
Module for downloading and loading 10x Genomics Xenium datasets.

This module provides functionality to:
1. Download Xenium datasets from 10x Genomics.
2. Extract ZIP archives.
3. Load the datasets as `SpatialData` objects.
4. Convert them to `AnnData` format and cache them locally.
"""

from dataclasses import dataclass
import shutil
import zipfile
import requests
from pathlib import Path
from typing import Dict, Literal, Optional
from anndata import read_h5ad
from anndata.typing import AnnData

xenium_key = Literal[
    "Xenium_V1_hLiver_nondiseased_section_FFPE", "Xenium_V1_hLiver_cancer_section_FFPE"
]


@dataclass
class DatasetEntry:
    """Metadata for a 10x Xenium dataset.

    Attributes:
        url (str): Direct download URL for the dataset ZIP archive.
        dataset_info (Optional[str]): Optional description or metadata for the dataset.
    """

    url: str
    dataset_info: Optional[str] = None


DATASETS: Dict[xenium_key, DatasetEntry] = {
    "Xenium_V1_hLiver_nondiseased_section_FFPE": DatasetEntry(
        "https://cf.10xgenomics.com/samples/xenium/1.9.0/Xenium_V1_hLiver_nondiseased_section_FFPE/Xenium_V1_hLiver_nondiseased_section_FFPE_outs.zip"
    ),
    "Xenium_V1_hLiver_cancer_section_FFPE": DatasetEntry(
        "https://cf.10xgenomics.com/samples/xenium/1.9.0/Xenium_V1_hLiver_cancer_section_FFPE/Xenium_V1_hLiver_cancer_section_FFPE_outs.zip"
    ),
}


def download_from_10x(dir: Path, url: str) -> None:
    """Download and extract a Xenium dataset from 10x Genomics.

    The dataset ZIP archive is downloaded to `dir` and extracted in place.
    The ZIP file is deleted after extraction.

    Args:
        dir (Path): Directory where the dataset will be stored.
        url (str): Direct URL to the 10x Genomics ZIP file.

    Raises:
        RuntimeError: If downloading or extracting the ZIP file fails.
    """
    dir.mkdir(parents=True, exist_ok=True)
    zip_path = dir / "download.zip"

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)
    except Exception as e:
        raise RuntimeError(f"Failed to download dataset from {url}") from e

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dir)
    except Exception as e:
        raise RuntimeError(f"Failed to extract ZIP archive: {zip_path}") from e
    finally:
        if zip_path.exists():
            zip_path.unlink()


def xenium_10x_loader(name: xenium_key) -> AnnData:
    """Load a 10x Xenium dataset as AnnData.

    Downloads the dataset if it is not already present in the current working directory.
    Converts the dataset to `AnnData` format and caches it as an `.h5ad` file.

    Args:
        name (xenium_key): The key identifying the dataset to load.

    Returns:
        AnnData: The loaded Xenium dataset in AnnData format.
    """
    dataset_path = Path.cwd() / name
    anndata_name = f"{name}.h5ad"
    anndata_path = dataset_path / anndata_name

    if not dataset_path.is_dir():
        url = DATASETS[name].url
        download_from_10x(dataset_path, url)

    if not anndata_path.is_file():
        from spatialdata_io import xenium
        from spatialdata_io.experimental import to_legacy_anndata

        data = xenium(dataset_path)
        data = to_legacy_anndata(data)
        data.write_h5ad(anndata_path)
    else:
        data = read_h5ad(anndata_path)

    return data
