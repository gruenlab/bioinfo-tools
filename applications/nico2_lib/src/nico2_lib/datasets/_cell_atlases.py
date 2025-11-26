from pathlib import Path
from typing import Optional
import shutil
import zipfile
import anndata as ad
import pandas as pd
from scanpy import read_10x_mtx
from anndata.typing import AnnData

from nico2_lib.datasets._utils import download_from_url




def _load_liver_cell_atlas_mtx_folder(folder: str) -> AnnData:
    import pandas as pd
    import scipy.io
    from anndata import AnnData
    folder = Path(folder)

    mat = scipy.io.mmread(folder / "matrix.mtx.gz").T.tocsr()
    barcodes = pd.read_csv(folder / "barcodes.tsv.gz", header=None)[0].astype(str)
    features = pd.read_csv(folder / "features.tsv.gz", sep="\t", header=None)
    if features.shape[1] >= 2:
        var = pd.DataFrame({
            "gene_id": features.iloc[:, 0].astype(str),
            "gene_name": features.iloc[:, 1].astype(str),
        })
    else:
        var = pd.DataFrame({
            "gene_id": features.iloc[:, 0].astype(str),
            "gene_name": features.iloc[:, 0].astype(str),
        })

    adata = AnnData(mat)
    adata.obs.index = barcodes
    adata.var = var
    adata.var.index = var["gene_name"]

    return adata

def human_liver_cell_atlas(dir: Optional[str] = None) -> AnnData:
    """
    Load the Human Liver Cell Atlas dataset and return an AnnData object.

    Adds:
        - annot_humanAll.csv (cell-level annotations)
        merged into adata.obs via the 'cell' column.
    """

    data_dir = Path(dir) if dir else Path.cwd()
    name = "human_liver_cell_atlas"
    dataset_path = data_dir / name
    dataset_path.mkdir(exist_ok=True, parents=True)

    anndata_path = dataset_path / f"{name}.h5ad"
    raw_zip_path = dataset_path / "download.zip"
    raw_data_path = dataset_path / "rawData_human"

    url = "https://www.livercellatlas.org/data_files/toDownload/rawData_human.zip"
    annotation_url = "https://www.livercellatlas.org/data_files/toDownload/annot_humanAll.csv"
    annotation_path = dataset_path / "annot_humanAll.csv"

    if anndata_path.is_file():
        return ad.read_h5ad(anndata_path)

    if not raw_data_path.is_dir():
        download_from_url(url, raw_zip_path)
        with zipfile.ZipFile(raw_zip_path, "r") as z:
            z.extractall(dataset_path)
        raw_zip_path.unlink(missing_ok=True)

    count_folder = raw_data_path / "countTable_human"
    adata = _load_liver_cell_atlas_mtx_folder(count_folder)

    download_from_url(annotation_url, annotation_path)
    annot = pd.read_csv(annotation_path)
    annot["cell"] = annot["cell"].astype(str)
    annot = annot.set_index("cell")
    common = adata.obs_names.intersection(annot.index)
    adata = adata[common].copy()
    adata.obs.index.name = "cell"
    adata.obs = adata.obs.join(annot.loc[common], how="left")
    adata.write_h5ad(anndata_path)

    if raw_data_path.exists():
        shutil.rmtree(raw_data_path)

    return adata