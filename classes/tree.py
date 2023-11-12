from classes.species import Species


class Tree:

    def __init__(self, coordinates_of_peak: list, height: float, diam_log: float, diam_exp: float, diam_poly: float):
        self.coordinates = coordinates_of_peak
        self.height = height
        self.diam_log = diam_log
        self.diam_exp = diam_exp
        self.diam_poly = diam_poly
        self.color = list()
        self.species = str()
        self.mass = float()

    def addColor(self, rgb: list):
        self.color = rgb

    def addHeight(self, height: float):
        self.height = height

    def addSpecies(self, species: Species):
        self.species = species.name

    def getSpecies(self):
        return self.species

    def __str__(self):
        return str(self.height) + ' ' + str(self.species) + ' ' + str(self.diam_log) + ' ' + str(self.diam_exp) + ' ' \
               + str(self.diam_poly) + ' ' + str(self.color) + ' ' + str(self.coordinates)
