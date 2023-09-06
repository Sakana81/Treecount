class Tree:

    def addDiameters(self, diameters: list):
        self.diameters = diameters
        self.diameter_med = sum(diameters) / len(diameters)

    def addColor(self, rgb: list):
        self.color = rgb

    def addHeight(self, height: float):
        self.height = height

    def addSpecies(self, species: str):
        self.species = species

    def getSpecies(self, function_params: list):
        pass

    def __init__(self, x: float, y: float, z_top: float, z_bottom: float, scaler: float):
        self.diameters = list()
        self.diameter_med = float()
        self.height = (z_top - z_bottom) * scaler
        self.coordinates = [x, y]
        self.species = str()
        self.mass = float()
        self.color = list()
