import os
import requests

from pandas import DataFrame, read_csv
from zipfile import ZipFile
from tqdm.notebook import tqdm
from urllib.parse import urlparse

from ...MetaObject import MetaObject


_EXPECTED_FILES = ["links.csv", "movies.csv", "ratings.csv", "tags.csv"]

# forcer types specifiques
_EXPECTED_TYPES = {"links.csv": {"imdbId": str, "tmdbId": str}}


def _instantiate(config):
    dataframes = {}
    for filename in _EXPECTED_FILES:
        # dataframe
        key = filename.split(".")[0]
        csv_path = os.path.join(config.install_path, filename)
        dtype = _EXPECTED_TYPES.get(filename, None)
        dataframes[key] = read_csv(csv_path, dtype=dtype)

        # path to dataframe
        path_key = "_".join([key, "path"])
        dataframes[path_key] = csv_path

    return MetaObject.from_dict(dataframes)

def _download(config):
    try:
        r = requests.get(config.url, stream=True)

        content_size = int(r.headers.get('content-length'))
        content_disposition = r.headers.get('content-disposition')

        if content_disposition is None:
            parsed_url = urlparse(r.url)
            _, filename = os.path.split(parsed_url.path)
        else:
            filename = content_disposition.split("=", 1)[-1]

        filename = filename.replace('"', "")
        filename = os.path.join(config.install_path, filename)

        if config.force_download or not os.path.exists(filename):
            with open(filename, "wb") as f:
                with tqdm(total=content_size) as progress:
                    for data in r.iter_content(chunk_size=16*1024):
                        f.write(data)
                        progress.update(len(data))
    except Exception as e:
        print(e)
        return None
    else:
        return filename

def _unzip_at_root(config, zip_filename):
    try:
        with ZipFile(zip_filename, 'r') as zip:
            for zip_info in zip.infolist():
                _, filename = os.path.split(zip_info.filename)
                if not filename:
                    continue

                dst_name = os.path.join(config.install_path, filename)
                bytes = zip.read(zip_info)

                with open(dst_name, "wb") as dst:
                    dst.write(bytes)
                
    except Exception as e:
        print(e)
        return None
    else:
        return True

def _exists(config):
    if not os.path.exists(config.install_path):
        return False

    for f in _EXPECTED_FILES:
        file_path = os.path.join(config.install_path, f)
        if not os.path.exists(file_path):
            return False

    return True

def load(config):
    """
    Utilitaire encapsulant installation et preprocessing du dataset MovieLens.

    config:
        Instance de Config

    Retour:
        MetaObject encapsulant le dataset ou None si probleme.
    """
    exists = _exists(config)

    if not config.force_install and exists:
        return _instantiate(config)

    if config.force_install or not exists:
        os.makedirs(config.install_path, exist_ok=True)

        print(f"Downloading {config.url}")
        zip_filename = _download(config)
        if zip_filename is None:
            print("Failed")
            return None

        print(f"Unziping {zip_filename}")
        if not _unzip_at_root(config, zip_filename):
            print("Failed")
            return None

        if config.delete_download:
            os.remove(zip_filename)

    return _instantiate(config)
