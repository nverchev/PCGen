"""Download of the archives holding the datasets."""

import logging
import pathlib
import zipfile

import requests
import tqdm

__all__ = ['download_extract_zip']

CHUNK_SIZE = 1 << 20


def download_extract_zip(target_folder: pathlib.Path, url: str) -> None:
    """Fetch an archive and extract it next to the folder it holds.

    The archive is kept, so an interrupted extraction does not download it
    again, and a folder that already exists is left alone.

    Args:
        target_folder: folder the archive extracts to.
        url: address of the archive.

    Side Effects:
        Writes the archive and the extracted folder under the parent folder.
    """
    if target_folder.exists():
        logging.info(f'Found {target_folder}, skipping the download.')
        return

    target_folder.parent.mkdir(parents=True, exist_ok=True)
    zip_path = target_folder.with_suffix('.zip')
    if not zip_path.exists():
        _download(zip_path, url)

    logging.info(f'Extracting {zip_path} to {target_folder.parent}.')
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target_folder.parent)

    return


def _download(zip_path: pathlib.Path, url: str) -> None:
    logging.info(f'Downloading {url}.')
    partial_path = zip_path.with_suffix('.part')
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        progress = tqdm.tqdm(total=total or None, unit='B', unit_scale=True, desc=zip_path.name)
        with partial_path.open('wb') as file, progress:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                file.write(chunk)
                progress.update(len(chunk))

    partial_path.rename(zip_path)
    return
