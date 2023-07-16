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

        self.title_regex = ""
        self.scrapping_mvl_url = "https://movielens.org/movies/"
        self.scrapping_imdb_url = "https://www.imdb.com/title/tt"
        self.scrapping_tmdb_url = "https://www.themoviedb.org/movie/"