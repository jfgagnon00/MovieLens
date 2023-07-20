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
    from .WebScraping.Config import Config as WebScrapingConfig

    # generer les configs par defaut
    dataset_config = MovieLensConfig()
    web_scraping_config = WebScrapingConfig()

    # appliquer les overrides
    if not config_overrides is None:
        MetaObject.override_from_object(dataset_config,
                                        config_overrides.dataset)
        
        MetaObject.override_from_object(web_scraping_config,
                                        config_overrides.web_scraping)

    return MetaObject.from_kwargs(dataset=dataset_config,
                                  web_scraping=web_scraping_config,
                                  executor=executor)