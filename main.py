

from cuttingArea import CuttingArea
import diameterextraction
import matplotlib.pyplot as plt

import numpy as np
from sklearn import linear_model
import open3d as o3d

def separateGround(las):
    ground_dots = np.empty(0)
    dots_sorted_by_z = las[las[:, 2].argsort()]
    a = np.hsplit(dots_sorted_by_z, 3)
    b = np.hstack((a[0], a[1]))

    ransac = linear_model.RANSACRegressor()
    ransac.fit(b, a[2])
    inlier_mask = ransac.inlier_mask_
    d = np.hstack((a[2], inlier_mask))
    c = np.hstack((b, d))
    c = c[c[4] == True]
    geom = o3d.geometry.PointCloud()
    geom.points = o3d.utility.Vector3dVector(c)
    o3d.visualization.draw_geometries([geom])

    print(c)
    return c

def main():
    impath = 'C:\\Users\pickles\Downloads\pp_nrgb_cut\pp1_nrgb_cut.tif'
    laspath = 'C:\\Users\pickles\Downloads\PP_Yusva_03052022\PP_Yusva_03052022\pp_1.las'

    probe = CuttingArea(impath, [3,2,7], laspath)
    # probe.pointcloud.makeSlice(15, 15)
    # probe.pointcloud.showPointCloud()
    # probe.pointcloud.plot_o3d()
    #s = separateGround(probe.pointcloud.las)
    probe.image.getPalette(number_colors=5,display=True)
    centers, clusters = probe.pointcloud.getMax(radius=10, eps=7, min_samples=10, divider=15, num_slice=15)

    for i in centers:
        z_min = probe.pointcloud.getFloor([i[0],i[1]], radius=50)
        print(i[2],z_min,i[2]-z_min)



    plt.scatter(centers[:, 0], centers[:, 1], color='r')
    plt.show()
    probe.image.plot()
    # diameterextraction.getDiams()


if __name__ == '__main__':
    #diameterextraction.getDiams()
    main()
