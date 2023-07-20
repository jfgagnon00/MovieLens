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
