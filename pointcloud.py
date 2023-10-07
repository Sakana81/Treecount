import laspy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
import open3d as o3d

from collections import defaultdict


class PointCloud:
    def makeSlice(self, divider, num_slice):
        size = self.las.shape[0]
        steps = [(size * i) // divider for i in range(divider)]
        planes = np.split(self.las, steps)
        self.slice = planes[num_slice]
        self.ground_slice = planes[1]

    def getFloor(self, radius=15):
        local_minima = []

        for i in range(self.slice.shape[0]-1):
            # radial mask with radius, could be beautified via numpy.linalg
            mask = np.sqrt(
                (self.ground_slice[:, 0] - self.ground_slice[i, 0]) ** 2 + (self.ground_slice[:, 1] - self.ground_slice[i, 1]) ** 2) <= radius
            # if current z value equals z_max in current region of interest, append to result list
            if self.ground_slice[i, 2] == np.min(self.ground_slice[mask], axis=0)[2]:
                local_minima.append(tuple(self.ground_slice[i]))
        local_min = np.array(local_minima)
        #local_max = np.delete(local_minima, 2, 1)
        min_vect = np.array([np.min(local_min[:, 0]), np.min(local_min[:, 1]), 0])
        flat_max = local_min - min_vect
        flat_max = flat_max / 10
        self.ground = flat_max
        return flat_max

    def getMax(self, radius=15, eps=7, min_samples=10, divider=5, num_slice=5, scaler=10):

        self.makeSlice(divider, num_slice)
        local_maxima = []

        for i in range(self.slice.shape[0]):
            # radial mask with radius, could be beautified via numpy.linalg
            mask = np.sqrt(
                (self.slice[:, 0] - self.slice[i, 0]) ** 2 + (self.slice[:, 1] - self.slice[i, 1]) ** 2) <= radius
            # if current z value equals z_max in current region of interest, append to result list
            if self.slice[i, 2] == np.max(self.slice[mask], axis=0)[2]:
                local_maxima.append(tuple(self.slice[i]))
        local_max = np.array(local_maxima)
        #local_max = np.delete(local_maxima, 2, 1)
        min_vect = np.array([np.min(local_max[:, 0]), np.min(local_max[:, 1]), 0])
        flat_max = local_max - min_vect
        flat_max = flat_max / 10

        db = DBSCAN(eps=eps, min_samples=min_samples).fit(flat_max)
        pred = db.fit_predict(flat_max)

        clusters = defaultdict(list)

        for i, c in enumerate(db.labels_):
            clusters[c].append(flat_max[i])

        centroids = []
        for k, v in clusters.items():
            centroid_of_cluster = np.mean(v, axis=0)
            centroids.append(centroid_of_cluster)
            # print(sect, centroid_of_cluster)
        centers = np.array((centroids)).astype(int)

        return centers, clusters

    def showPointCloud(self):
        plt.scatter(self.slice[:, 0], self.slice[:, 1])
        plt.show()

    def plot_o3d(self):
        geom = o3d.geometry.PointCloud()
        geom.points = o3d.utility.Vector3dVector(self.las)
        o3d.visualization.draw_geometries([geom])

    def __init__(self, path2las: str):
        las = laspy.read(path2las)
        point_data = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
        self.las = point_data[point_data[:, 2].argsort()]
        self.slice = np.empty(0)
        self.ground_slice = np.empty(0)
        self.ground = np.empty(0)
