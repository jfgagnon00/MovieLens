from .Concurrent import create_thread_pool_executor, parallel_for
from .MetaObject import MetaObject
from .Profile import Profile


def get_configs(config_overrides, executor=None):
    """
    Utilitaire pour obtenir la configuration complete du pipeline.

    config_overrides:
        Nom du fichier .json qui contient les overrides des configs.

    executor:
        Pour les etapes qui requierent du multiprocessing, l'executor
        a utiliser. Si None, Concurrent.create_thread_pool_executor()
        sera utilise.

    retour:
        MetaObject contenant toute la configuration avec les overrides
        appliques.
    """
    if executor is None:
        executor = create_thread_pool_executor(max_workers=None)

    try:
        config_overrides = MetaObject.from_json(config_overrides)
    except:
        config_overrides = None
        pass

    from .dataset.MovieLens.Config import Config as MovieLensConfig

    # generer les configs par defaut
    pv_config = MovieLensConfig()

    # appliquer les overrides
    if not pv_config is None:
        MetaObject.override_from_object(pv_config,
                                        config_overrides.dataset)
        return MetaObject.from_kwargs(dataset=pv_config,
                                      executor=executor)