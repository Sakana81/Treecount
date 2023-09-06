import os
from collections import defaultdict

import laspy
import numpy as np
import pandas as pd
import itertools

import rasterio

# import tensorflow as tf
# from keras import layers

from sklearn.preprocessing import minmax_scale
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score as ss
import open3d as o3d
import matplotlib.pyplot as plt
import skimage.filters as filters
import skimage.feature as feature
from sklearn import linear_model
from sklearn.cluster import KMeans
# import shapesML
# import tiff
import colorextraction
import coordsextraction

import diameterextraction


def getLasFile(path_to_file):
    try:
        las = laspy.read(path_to_file)
        return las
    except:
        print("no such file in directory")


def prepSlice(cloud, divider):
    size = cloud.shape[0]
    steps = [(size * i) // divider for i in range(divider)]
    planes = np.split(cloud, steps)
    # for plane in planes:
    # np.concatenate(plane)
    # ste = np.arange(divider-num+1,divider+1)
    # res = np.empty(shape=[0,3])
    # res = planes[:, ste]
    return planes


def prepSegment(cloud, divider):
    plane = np.array_split(cloud, divider)

    return plane[1]


def getScaledDataFromLas(las, scale):
    point_data = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
    dots_sorted_by_z = point_data[point_data[:, 2].argsort()]
    df_scaled = minmax_scale(dots_sorted_by_z, feature_range=(0, scale))
    return df_scaled


def plot(plot_data):
    geom = o3d.geometry.PointCloud()
    geom.points = o3d.utility.Vector3dVector(plot_data)
    o3d.visualization.draw_geometries([geom])


def plotscatter(dfx, dfy, pred, name):
    plt.scatter(dfx, dfy, c=pred, cmap='Paired')
    plt.title(name)
    plt.show()


def dbscan(X, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    pred = db.fit_predict(X)
    return pred


def findMaximums(las_sorted_by_z):
    np.delete(las_sorted_by_z, 2, 1)
    for point in las_sorted_by_z:
        if point[0] and point[1]:
            pass


def getNotScaledLas(las):
    result = las.xyz
    X = np.max(result[:, 0]) - np.min(result[:, 0])
    Y = np.max(result[:, 1]) - np.min(result[:, 1])
    print('max x', np.max(result[:, 0]), 'min x', np.min(result[:, 0]), 'delta', X)
    print('max y', np.max(result[:, 1]), 'min y', np.min(result[:, 1]), 'delta', Y)
    # for point in result:
    #   point[0] +=4

    return result


def colorDots():
    pass


def get_scores_and_labels(X):
    #   Находим комбинации
    epsilons = np.linspace(0.05, 1, num=15)
    min_samples = np.arange(10, 250, step=10)
    combinations = list(itertools.product(epsilons, min_samples))

    scores = []
    all_labels = []

    for i, (eps, num_samples) in enumerate(combinations):
        dbscan_cluster_model = DBSCAN(eps=eps, min_samples=num_samples).fit(X)
        labels = dbscan_cluster_model.labels_
        labels_set = set(labels)
        num_clusters = len(labels_set)
        if -1 in labels_set:
            num_clusters -= 1

        if (num_clusters < 100) or (num_clusters > 500):
            scores.append(-10)
            all_labels.append('bad')
            c = (eps, num_samples)
            # print(f'Combination {c} on iteration {i+1} has {num_clusters} clusters. Moving on')
            continue

        scores.append(ss(X, labels))
        all_labels.append(labels)
        print(
            f'!!!!!!! Combination: {combinations[i]} Index: {i}, Score: {scores[-1]}, Labels: {all_labels[-1]}, NumClusters: {num_clusters}')

    best_index = np.argmax(scores)
    best_parameters = combinations[best_index]
    best_labels = all_labels[best_index]
    best_score = scores[best_index]
    return {'best_epsilon': best_parameters[0],
            'best_min_samples': best_parameters[1],
            'best_labels': best_labels,
            'best_score': best_score}


def calculateNumTrees(lasFileDir):
    for i, file in enumerate(os.listdir(lasFileDir)):
        print(file)
        las = getLasFile(os.path.join(lasFileDir, file))
        df = getNotScaledLas(las)
        height_slices = prepSlice(df, 5)

        df_pd = pd.DataFrame(height_slices[5])
        # height_slice = prepSegment(df, 7)
        # df_pd = pd.DataFrame(height_slice)
        plane_xy = pd.DataFrame({0: df_pd[0], 1: df_pd[1]})

        db = dbscan(plane_xy, 0.2535714285, 30)

        # print(len(set(db)))
        plotscatter(plane_xy[0], plane_xy[1], db, file)

        # 'best_epsilon': 0.25357142857142856, 'best_min_samples': 20
        # 0.185714, 30
        # 0.2535714285, 30

        # results = get_scores_and_labels(plane_xy)
        # print(results['best_epsilon'],results['best_min_samples'])

        # plot(df)
        print(len(set(db)))


def separateGround(las):
    arr = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    ground_dots = np.array(arr)
    print(ground_dots)
    dots_sorted_by_z = las[las[:, 2].argsort()]
    a = np.hsplit(dots_sorted_by_z, 3)
    b = np.hstack((a[0], a[1]))

    ransac = linear_model.RANSACRegressor()
    ransac.fit(b, a[2])
    inlier_mask = ransac.inlier_mask_
    d = np.hstack((a[2], inlier_mask))
    c = np.hstack((b, d))
    c = c[c[4] == True]
    print(c)
    return c


def main():
    pathToLasFile = "C:\\Users\pickles\Downloads\PP_Yusva_03052022\PP_Yusva_03052022\pp_1.las"
    # pathToLasFile = "C:\\Users\pickles\Downloads\Etalony_las\Etalony_las\Pinus\P7_1.las"
    peaks, clusters = coordsextraction.calculateNumTrees(pathToLasFile, num_slice=15, slice=15, radius=15)

    peaks = peaks.astype(int)

    impath = 'C:\\Users\pickles\Downloads\pp_nrgb_cut\pp1_nrgb_cut.tif'
    img, colors = colorextraction.getColorsByImg(impath, channels=[2, 3, 7], number_of_colors=15)
    pixels = []
    for peak in peaks:
        pixels.append(img[peak[1]][peak[0]])
    pixels = np.array((pixels))
    fff = KMeans(len(colors)).fit(colors)
    pred = fff.predict(pixels)
    # result = np.c_(pixels,pred)

    clusters = defaultdict(list)

    for i, c in enumerate(pred):
        clusters[c].append(peaks[i])

    for k, v in clusters.items():
        print('......')
        print('cluster #{}'.format(k), v)

    # plt.gca().invert_yaxis()
    plt.imshow(img)
    scatter = plt.scatter(peaks[:, 0], peaks[:, 1], marker='.', color="white")
    # ax = scatter.axes
    # ax.invert_yaxis()
    # ax.invert_xaxis()
    plt.show()


from cuttingArea import CuttingArea


def main2():
    impath = 'C:\\Users\pickles\Downloads\pp_nrgb_cut\pp1_nrgb_cut.tif'
    laspath = 'C:\\Users\pickles\Downloads\PP_Yusva_03052022\PP_Yusva_03052022\pp_1.las'

    probe = CuttingArea(impath, [3,2,7], laspath)
    # probe.pointcloud.makeSlice(15, 15)
    # probe.pointcloud.showPointCloud()
    # probe.pointcloud.plot_o3d()

    centers, clusters = probe.pointcloud.getMax(radius=10, eps=7, min_samples=10, divider=15, num_slice=15)
    for i in centers:
        z_min = probe.pointcloud.getFloor([i[0],i[1]], radius=50)
        print(i[2],z_min,i[2]-z_min)

    plt.scatter(centers[:, 0], centers[:, 1], color='r')
    plt.show()
    probe.image.plot()
    # diameterextraction.getDiams()


if __name__ == '__main__':
    main2()
