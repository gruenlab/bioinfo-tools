from pathlib import Path
from typing import Optional
import shutil
import zipfile
import anndata as ad
import pandas as pd
from scanpy import read_10x_mtx
from anndata.typing import AnnData

from nico2_lib.datasets._utils import download_from_url


def load_samplecomp(table_path: Path) -> pd.DataFrame:
    df = pd.read_csv(table_path, sep="\t", index_col=0)
    df.index = df.index.astype(str)
    return df

def human_liver_cell_atlas(dir: Optional[str] = None) -> AnnData:
    """
    Load the Human Liver Cell Atlas dataset and return an AnnData object.

    The function:
        - checks if <dataset>/<name>.h5ad already exists
        - if not, downloads rawData_human.zip as download.zip
        - unzips into raw_data
        - loads countTable_* folders (RNA + ADT)
        - loads sampleComp_* files into obs
        - creates and saves an AnnData object
    """


    data_dir = Path(dir) if dir else Path.cwd()
    name = "human_liver_cell_atlas"
    dataset_path = data_dir / name
    dataset_path.mkdir(exist_ok=True, parents=True)

    anndata_path = dataset_path / f"{name}.h5ad"
    raw_zip_path = dataset_path / "download.zip"
    raw_data_path = dataset_path / "rawData_human"

    url = "https://www.livercellatlas.org/data_files/toDownload/rawData_human.zip"

    if anndata_path.is_file():
        return ad.read_h5ad(anndata_path)

    if not raw_data_path.is_dir():
        download_from_url(url, raw_zip_path)
        with zipfile.ZipFile(raw_zip_path, "r") as z:
            z.extractall(dataset_path)
        raw_zip_path.unlink(missing_ok=True)

    count_folders = sorted(
        f for f in raw_data_path.iterdir()
        if f.is_dir() and (f / "matrix.mtx.gz").exists()
    )

    adatas = [read_10x_mtx(f) for f in count_folders]

    data = ad.concat(adatas, join="outer", axis=0)  # concat cells, align features by name

    for t in raw_data_path.glob("sampleComp_*.txt"):
        df = load_samplecomp(t)
        intersect = data.obs_names.intersection(df.index)
        if len(intersect) > 0:
            colname = t.stem  # e.g. "sampleComp_humanAll"
            tmp = df.loc[intersect]
            for c in tmp.columns:
                data.obs[f"{colname}_{c}"] = tmp[c]

    data.write_h5ad(anndata_path)

    if raw_data_path.exists():
        shutil.rmtree(raw_data_path)

    return data