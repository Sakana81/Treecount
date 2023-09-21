from species import Species

class Tree:

    def addDiameters(self, diameters: list):
        self.diameters = diameters
        self.diameter_med = sum(diameters) / len(diameters)

    def addColor(self, rgb: list):
        self.color = rgb

    def addHeight(self, height: float):
        self.height = height

    def addSpecies(self, species: Species):
        self.species = species.name

    def getSpecies(self):
        return self.species

    def __init__(self, x: float, y: float, z_top: float, z_bottom: float, scaler: float):
        self.diameters = list()
        self.diameter_med = float()
        self.height = (z_top - z_bottom) * scaler
        self.coordinates = [x, y]
        self.species = str()
        self.mass = float()
        self.color = list()
