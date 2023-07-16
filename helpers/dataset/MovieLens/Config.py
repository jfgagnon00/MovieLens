class Config():
    """
    Parametres configurant l'installation/preprocessing
    de MovieLens
    """
    def __init__(self):
        self.url = ""
        self.install_path = ""

        self.force_download = False
        self.force_install = False
