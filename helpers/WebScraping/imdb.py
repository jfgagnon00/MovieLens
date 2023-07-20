import requests

from tqdm.notebook import tqdm

def imdb_request(imdbId_series, config, callback):
    """
    Utilitaire pour generer une liste de requete a IMDB.

    imdbId_series:
        Pandas Series contenant les id IMDB pour les requetes

    config:
        instance de Config

    callback:
        callback appele a chaque requete. Signature:
        callback(requests.Response, Series.index, imdbId)
    """
    count = imdbId_series.shape[0]
    for index, id in tqdm(imdbId_series.items(), total=count):
        url = "".join([config.imdb_url, str(id)])
        with requests.get(url, headers=config.imdb_headers.__dict__) as r:
            callback(r, index, id)
