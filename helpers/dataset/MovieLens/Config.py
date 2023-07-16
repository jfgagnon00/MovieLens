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

        # config a le pattern
        # sera compiler lors du load des configs
        self.title_regex = ""
