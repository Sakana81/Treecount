from image import Image
from pointcloud import PointCloud
from classes.tree import Tree
from classes.species import Species


class CuttingArea:

    def __init__(self, pathToImg: str, channels: list, pathToLas: str, scaler):
        self.pointcloud = PointCloud(pathToLas, scaler)
        self.image = Image(pathToImg, channels)
        self.trees = list()
        self.scaler = float()
        self.species_list = list()

    def setScaler(self, scaler: float):
        self.scaler = scaler

    def addTree(self, tree: Tree):
        self.trees.append(tree)

    def add_species(self, name: str, color: list):
        self.species_list.append(Species(name, color))

    def get_species_colors(self):
        return [species.mean_color for species in self.species_list]
