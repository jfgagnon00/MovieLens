import requests

from tqdm.notebook import tqdm
from ..Concurrent import parallel_for


def _request(imdbId, config):
    url = "".join([config.imdb_url, str(imdbId)])
    return requests.get(url, headers=config.imdb_headers.__dict__)

def _task_iter(config, callback, index_imdbId):
    index, imdbId = index_imdbId
    with _request(imdbId, config) as response:
        request_result = callback(response, index, imdbId)

    return request_result

def _task_completed(progress, task_results, final_results):
    final_results.append(task_results)
    progress.update(1)

def imdb_requests_parallel(imdbId_series, config, callback, executor=None):
    """
    Utilitaire pour generer une liste de requete a IMDB en parralele.

    imdbId_series:
        Pandas Series contenant les id IMDB pour les requetes

    config:
        instance de Config

    callback:
        callback appele a chaque requete. Signature:
        callback(requests.Response, Series.index, imdbId)
    """
    count = imdbId_series.shape[0]
    with tqdm(total=count) as progress:
        results = []
        parallel_for(imdbId_series.items(),
                    _task_iter,
                    config,
                    callback,
                    task_completed=lambda tr: _task_completed(progress, tr, results),
                    executor=executor)
    return results

def imdb_requests(imdbId_series, config, callback):
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
        with _request(id, config) as r:
            callback(r, index, id)
