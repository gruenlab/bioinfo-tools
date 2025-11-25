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
from anndata import read_h5ad # type: ignore
from anndata.typing import AnnData # type: ignore

xenium_key = Literal[
    "Xenium_V1_hLiver_nondiseased_section_FFPE",
    "Xenium_V1_hLiver_cancer_section_FFPE",
    "Xenium_V1_Human_Lung_Cancer_FFPE",
    "Xenium_Prime_Human_Lung_Cancer_FFPE",
    "Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE",
    "Xenium_FFPE_Human_Breast_Cancer_Rep1",
    "Xenium_FFPE_Human_Breast_Cancer_Rep2",
    "Xenium_V1_FFPE_Preview_Human_Breast_Cancer_Sample_2"
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
        "https://cf.10xgenomics.com/samples/xenium/1.9.0/Xenium_V1_hLiver_nondiseased_section_FFPE/Xenium_V1_hLiver_nondiseased_section_FFPE_outs.zip",
        "Xenium In Situ Gene Expression data for adult human liver sections using the Xenium Human Multi-Tissue and Cancer Panel."
    ),
    "Xenium_V1_hLiver_cancer_section_FFPE": DatasetEntry(
        "https://cf.10xgenomics.com/samples/xenium/1.9.0/Xenium_V1_hLiver_cancer_section_FFPE/Xenium_V1_hLiver_cancer_section_FFPE_outs.zip",
        "Xenium In Situ Gene Expression data for adult human liver sections using the Xenium Human Multi-Tissue and Cancer Panel."
    ),
    "Xenium_V1_Human_Lung_Cancer_FFPE": DatasetEntry(
        "https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_V1_Human_Lung_Cancer_FFPE/Xenium_V1_Human_Lung_Cancer_FFPE_outs.zip",
        "Experiment 1: Xenium In Situ Gene Expression (Xenium v1) data for adult human lung adenocarcinoma tissue (FFPE) using the Xenium Human Lung Gene Expression Panel with nuclear expansion."
    ),
    "Xenium_Prime_Human_Lung_Cancer_FFPE": DatasetEntry(
        "https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/3.0.0/Xenium_Prime_Human_Lung_Cancer_FFPE/Xenium_Prime_Human_Lung_Cancer_FFPE_outs.zip",
        "Experiment 2: Xenium Prime 5K In Situ Gene Expression with Cell Segmentation data for human lung adenocarcinoma tissue (FFPE) using the Xenium Prime 5K Human Pan Tissue and Pathways Panel."
    ),
    "Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE": DatasetEntry(
        "https://cf.10xgenomics.com/samples/xenium/1.7.0/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE_outs.zip",
        "Preview of Xenium In Situ Gene Expression data for adult human skin sections, using a development version of the Xenium Human Skin Gene Expression Panel."
    ),
    "Xenium_FFPE_Human_Breast_Cancer_Rep1": DatasetEntry(
        "https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_outs.zip",
        "In Situ Sample 1, Replicate 1: This dataset is associated with the article High resolution mapping of the tumor microenvironment using integrated single-cell, spatial and in situ analysis published in Nature Communications. See the publication for full details on methods and results."
    ),
    "Xenium_FFPE_Human_Breast_Cancer_Rep2": DatasetEntry(
        "https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep2/Xenium_FFPE_Human_Breast_Cancer_Rep2_outs.zip",
        "In Situ Sample 1, Replicate 2: This dataset is associated with the article High resolution mapping of the tumor microenvironment using integrated single-cell, spatial and in situ analysis published in Nature Communications. See the publication for full details on methods and results."
    ),
    "Xenium_V1_FFPE_Preview_Human_Breast_Cancer_Sample_2": DatasetEntry(
        "https://cf.10xgenomics.com/samples/xenium/1.4.0/Xenium_V1_FFPE_Preview_Human_Breast_Cancer_Sample_2/Xenium_V1_FFPE_Preview_Human_Breast_Cancer_Sample_2_outs.zip",
        "In Situ Sample 2: This dataset is associated with the article High resolution mapping of the tumor microenvironment using integrated single-cell, spatial and in situ analysis published in Nature Communications. See the publication for full details on methods and results."
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


def xenium_10x_loader(name: xenium_key, dir: Optional[str] = None) -> AnnData:
    """Load a 10x Xenium dataset as AnnData.
    Args:
        name (xenium_key): The key identifying the dataset to load.
        dir (Optional[str]): Base directory where the dataset folder will be created.
                             Defaults to the current working directory.

    Returns:
        AnnData: The loaded Xenium dataset in AnnData format.
    """
    data_dir = Path(dir) if dir else Path.cwd()
    dataset_path = data_dir / name
    anndata_name = f"{name}.h5ad"
    anndata_path = dataset_path / anndata_name
    raw_data_path = dataset_path / "raw_data"
    if anndata_path.is_file():
        return read_h5ad(anndata_path)
    if not raw_data_path.is_dir():
        url = DATASETS[name].url
        download_from_10x(raw_data_path, url)
    from spatialdata_io import xenium # type: ignore
    from spatialdata_io.experimental import to_legacy_anndata # type: ignore

    data = xenium(raw_data_path)
    data = to_legacy_anndata(data)
    data.write_h5ad(anndata_path)
    if raw_data_path.exists():
        shutil.rmtree(raw_data_path)
    return data
