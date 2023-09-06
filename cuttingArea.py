import numpy as np

from image import Image
from pointcloud import PointCloud
from tree import Tree


class CuttingArea:
    def setScaler(self, scaler: float):
        self.scaler = scaler

    def addTree(self, x: float, y: float, z_top: float, z_bottom: float):
        np.append(self.trees, Tree(x, y, z_top, z_bottom, scaler=self.scaler))

    def __init__(self, pathToImg: str, channels: list, pathToLas: str):
        self.pointcloud = PointCloud(pathToLas)
        self.image = Image(pathToImg, channels)
        self.trees = np.empty(0)
        self.scaler = float()
