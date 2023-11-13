import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from classes.cuttingArea import CuttingArea
from classes.tree import Tree
import diameterextraction
import heightextraction


def main():
    #impath = ''
    #laspath = ''
    impath = ''
    laspath = ''
    las_to_tiff_scaler = 10
    number_of_colors = 5
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

    probe = CuttingArea(impath, [3, 2, 7], laspath, las_to_tiff_scaler)
    probe.image.getPalette(number_colors=number_of_colors, display=False)

    centers, clusters = probe.pointcloud.getMax(radius=10, eps=7, min_samples=10, divider=15, num_slice=15)
    z_min = probe.pointcloud.getFloor(radius=1.5)
    sea_level_height = 0

    for i, color in enumerate(probe.image.dominantColors):
        probe.add_species(f'{i}', color)
    species_classifier = KMeans(len(probe.get_species_colors())).fit(probe.get_species_colors())

    d1 = []
    d2 = []
    d3 = []
    h = []
    params = diameterextraction.get_params(model_trees)
    for center in centers:
        height = heightextraction.get_height(center, z_min, sea_level_height, radius=5)
        h.append(height)
        diams = diameterextraction.fit_tree(height, params)
        d1.append(diams[0])
        d2.append(diams[1])
        d3.append(diams[2])
        tree_color = probe.image.getColorByCoordinates(center[0],center[1])
        species = species_classifier.predict(tree_color)
        tree = Tree(center, height, diams[0], diams[1], diams[2])
        tree.addSpecies(species)
        probe.trees.append(tree)
        print(tree)
    print(len(probe.trees))

    plt.scatter(d1,h, color='r')
    plt.scatter(d2,h, color='g')
    plt.scatter(d3,h, color='b')
    plt.show()
    #probe.image.plot()


if __name__ == '__main__':
    main()
