class Config():
    """
    Parametres configurant le web scaraping
    """
    def __init__(self):
        self.mvl_url = ""
        self.tmdb_url = ""

        # web scrapping a des problemes avec python
        # il faut 'hacker' les request http
        # details a: https://tinyurl.com/54babbdh
        self.imdb_headers = ""
        self.imdb_url = ""
