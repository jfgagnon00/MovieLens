class Config():
    """
    Parametres configurant l'installation/preprocessing
    de MovieLens
    """
    def __init__(self):
        self.url = ""
        self.install_path = ""

        self.force_download = False
        self.delete_download = True
        self.force_install = False

        self.genre_splitter = "|"

        self.title_regex = ""
        self.scrapping_mvl_url = "http://movielens.org/movies/"
        self.scrapping_tmdb_url = "https://www.themoviedb.org/movie/"

        # web scrapping a des problemes avec python
        # il faut 'hacker' les request http
        # details a: https://tinyurl.com/54babbdh
        self.scrapping_imdb_url = "http://www.imdb.com/title/tt"
        self.scrapping_imdb_headers = ""
