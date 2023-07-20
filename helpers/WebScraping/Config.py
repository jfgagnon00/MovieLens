class Config():
    """
    Parametres configurant le web scaraping
    """
    def __init__(self):
        self.mvl_url = "http://movielens.org/movies/"
        self.tmdb_url = "https://www.themoviedb.org/movie/"

        # web scrapping a des problemes avec python
        # il faut 'hacker' les request http
        # details a: https://tinyurl.com/54babbdh
        self.imdb_url = "http://www.imdb.com/title/tt"
        self.imdb_headers = ""
