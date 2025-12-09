from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional
import anndata as ad  # type: ignore
from anndata.typing import AnnData  # type: ignore
import pandas as pd
import scipy.io

from nico2_lib.datasets._utils import (
    ensure_dataset_dir,
    download_and_extract,
    cleanup,
    download_file,
)


def _load_liver_cell_atlas_mtx_folder(folder: str) -> AnnData:
    path_folder = Path(folder)
    mat = scipy.io.mmread(path_folder / "matrix.mtx.gz").T.tocsr()
    barcodes = pd.read_csv(path_folder / "barcodes.tsv.gz", header=None)[0].astype(str)
    features = pd.read_csv(path_folder / "features.tsv.gz", sep="\t", header=None)
    if features.shape[1] >= 2:
        var = pd.DataFrame(
            {
                "gene_id": features.iloc[:, 0].astype(str),
                "gene_name": features.iloc[:, 1].astype(str),
            }
        )
    else:
        var = pd.DataFrame(
            {
                "gene_id": features.iloc[:, 0].astype(str),
                "gene_name": features.iloc[:, 0].astype(str),
            }
        )
    adata = AnnData(mat)
    adata.obs.index = barcodes
    adata.var = var
    adata.var.index = var["gene_name"]
    return adata


def human_liver_cell_atlas(dir: Optional[str] = None) -> ad.AnnData:
    dataset_path, anndata_path = ensure_dataset_dir("human_liver_cell_atlas", dir)
    raw_data_path = dataset_path / "rawData_human"
    annotation_path = dataset_path / "annot_humanAll.csv"

    if anndata_path.is_file():
        return ad.read_h5ad(anndata_path)

    if not raw_data_path.is_dir():
        download_and_extract(
            "https://www.livercellatlas.org/data_files/toDownload/rawData_human.zip",
            dataset_path,
        )

    count_folder = raw_data_path / "countTable_human"
    adata = _load_liver_cell_atlas_mtx_folder(count_folder)

    download_file(
        "https://www.livercellatlas.org/data_files/toDownload/annot_humanAll.csv",
        annotation_path,
    )
    annot = pd.read_csv(annotation_path).set_index("cell")
    annot.index = annot.index.astype(str)
    common = adata.obs_names.intersection(annot.index)
    adata = adata[common].copy()
    adata.obs.index.name = "cell"
    adata.obs = adata.obs.join(annot.loc[common], how="left")

    adata.write_h5ad(anndata_path)
    cleanup(raw_data_path)

    return adata


brain_cell_atlas_key = Literal["Zeng-Aging-Mouse-10Xv3"]


@dataclass
class AllenBrainCellAtlasEntry:
    expression_matrices_url: str
    metadata_url: str


ABC_DATASETS: Dict[brain_cell_atlas_key, AllenBrainCellAtlasEntry] = {
    "Zeng-Aging-Mouse-10Xv3": AllenBrainCellAtlasEntry(
        "https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/expression_matrices/Zeng-Aging-Mouse-10Xv3/20241130/Zeng-Aging-Mouse-10Xv3-raw.h5ad",
        "https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/index.html#metadata/Zeng-Aging-Mouse-10Xv3/20250131/",
    )
}


def allen_brain_cell_atlas(
    id: brain_cell_atlas_key, dir: Optional[str] = None
) -> ad.AnnData:
    _, anndata_path = ensure_dataset_dir(id, dir)
    if anndata_path.is_file():
        return ad.read_h5ad(anndata_path)
    download_file(ABC_DATASETS[id].expression_matrices_url, anndata_path)
    return ad.read_h5ad(anndata_path)
