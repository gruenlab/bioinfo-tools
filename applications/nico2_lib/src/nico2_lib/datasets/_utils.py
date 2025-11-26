from pathlib import Path


def download_from_url(url: str, destination: Path) -> None:
    """
    Download a file from a URL to the given destination path.

    Args:
        url (str): URL to download from.
        destination (Path): File path where the downloaded file will be saved.
    """
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

