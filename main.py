import matplotlib.pyplot as plt
import numpy as np

from cuttingArea import CuttingArea
import diameterextraction
import heightextraction


def main():
    impath = 'C:\\Users\pickles\Downloads\pp_nrgb_cut\pp1_nrgb_cut.tif'
    laspath = 'C:\\Users\pickles\Downloads\PP_Yusva_03052022\PP_Yusva_03052022\pp_1.las'

    probe = CuttingArea(impath, [3, 2, 7], laspath)

    # probe.image.getPalette(number_colors=5,display=True)
    centers, clusters = probe.pointcloud.getMax(radius=10, eps=7, min_samples=10, divider=15, num_slice=15)
    z_min = probe.pointcloud.getFloor(radius=15)
    sea_level_height = min(z_min[:, 2])

    # Я не знаю в каких величинах извлекается высота, на тесте высота всех деревьев была примерно 300 условных единиц.
    # НЕОБХОДИМО ИЗМЕНИТЬ ЭТОТ ПАРАМЕТР
    height_scaler = 20

    heights = map(lambda x: x / height_scaler, heightextraction.getHeight(centers, z_min, sea_level_height))

    # ЭТО ДЛЯ ПРИМЕРА! необходимо подгружать от пользователя в таком же формате: [[диаметр (см), высота(м)],[диаметр(см), высота(м)],...]
    model_trees = np.array([[8.45, 11.1],
                            [7.95, 11.4],
                            [10.5, 11.7],
                            [10.25, 14.7],
                            [11.55, 17.5],
                            [12, 12.4],
                            [14.4, 15.5],
                            [13.5, 12.5],
                            [17.5, 21.3],
                            [17.4, 20.8],
                            [18.65, 21.3],
                            [18.5, 21],
                            [21.35, 20],
                            [22.5, 22.3],
                            [25.1, 21.1],
                            [25.45, 25],
                            [26.2, 24.6],
                            [27.5, 24]])

    # Здесь будет результат, формат можно глянуть в описании diameterextraction.getDiams()
    diameters_and_height = diameterextraction.getDiams(model_trees, heights)
    # print(diameters_and_height)

    plt.scatter(centers[:, 0], centers[:, 1], color='r')
    plt.show()
    # probe.image.plot()


if __name__ == '__main__':
    main()
