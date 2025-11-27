import shutil
from typing import Optional
import pandas as pd
import anndata as ad  # type: ignore
from anndata.typing import AnnData  # type: ignore
import scanpy as sc
from nico2_lib.datasets._utils import (
    github_url_to_raw,
    ensure_dataset_dir,
    download_and_extract,
    cleanup,
)


def small_mouse_intestine_sc(dir: Optional[str] = None) -> AnnData:
    dataset_path, anndata_path = ensure_dataset_dir("mouse_small_intestine_sc", dir)
    raw_data_path = dataset_path / "raw_data"

    if anndata_path.is_file():
        return ad.read_h5ad(anndata_path)

    url = github_url_to_raw(
        "https://github.com/ankitbioinfo/nico_tutorial/blob/main/inputRef.zip"
    )
    download_and_extract(url, raw_data_path)

    input_h5ad = raw_data_path / "inputRef" / "input_ref.h5ad"
    shutil.move(str(input_h5ad), anndata_path)
    adata = ad.read_h5ad(anndata_path)
    cleanup(raw_data_path)
    return adata


def small_mouse_intestine_merfish(dir: Optional[str] = None) -> AnnData:
    dataset_path, anndata_path = ensure_dataset_dir(
        "mouse_small_intestine_merfish", dir
    )
    raw_data_path = dataset_path / "raw_data"

    if anndata_path.is_file():
        return ad.read_h5ad(anndata_path)

    url = github_url_to_raw(
        "https://github.com/ankitbioinfo/nico_tutorial/blob/main/inputQuery.zip"
    )
    download_and_extract(url, raw_data_path)
    adata = sc.read(raw_data_path / "inputQuery" / "gene_by_cell.csv").transpose()
    coordinates = pd.read_csv(
        raw_data_path / "inputQuery" / "tissue_positions_list.csv"
    ).to_numpy() # type: ignore
    adata.obsm["spatial"] = coordinates[:, 1:3].astype(float)
    adata.write_h5ad(anndata_path)
    cleanup(raw_data_path)
    return adata
