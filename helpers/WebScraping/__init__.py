from .imdb import imdb_request

def get_nested_property(json, keys_iterable):
    """
    Acceder une propriete d'un object json

    json:
        object json sur lequel on demande la propriete

    keys_iterable:
        iterable sur les noms de proprietes a acceder
        ex) ["a", "b", "c"] -> json.a.b.c

    Retour:
        object si propriete est presente, None autrement
    """
    for k in keys_iterable:
        if not k in json:
            return None
        json = json[k]
        
    return json