import matplotlib.pyplot as plt
from cuttingArea import CuttingArea

def main():
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
    main()
