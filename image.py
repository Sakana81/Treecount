import matplotlib.pyplot as plt
import rasterio
import numpy as np
import cv2


def create_bar(height, width, color):
    bar = np.full((height, width, 3), color)
    red, green, blue = int(255 * color[0]), int(255 * color[1]), int(255 * color[2])
    return bar, (red, green, blue)


class Image:
    """
    img = np.array()
    imgRGB = np.array()
    dominantColors = np.array()
    meanColor = np.array()
    """

    def selectChannels(self, channels):
        def normalize(array):
            array_min, array_max = array.min(), array.max()
            return (array - array_min) / (array_max - array_min)

        # Навести порядок тут с чтением и выбором каналов в изображении
        res = []
        for num_channel in channels:
            layer = normalize(self.img[num_channel-1])
            res.append(layer)
        return np.dstack(res)

    def getPalette(self, number_colors=5, display=False):

        data = np.reshape(self.imgRGB, (self.imgRGB.shape[0] * self.imgRGB.shape[1], self.imgRGB.shape[2]))
        data = data[~np.all(data == 0, axis=1)]

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, self.dominantColors = cv2.kmeans(data, number_colors, None, criteria, 10, flags)
        self.meanColor = self.dominantColors.mean(axis=0)

        if display:

            bars = []
            rgb_values = []

            for index, row in enumerate(self.dominantColors):
                bar, rgb = create_bar(50, 50, row)
                bars.append(bar)
                rgb_values.append(rgb)

            img_bar = np.hstack(bars)

            cv2.imshow('Image', self.imgRGB)
            cv2.imshow('Dominant colors', img_bar)
            cv2.waitKey(0)

    def getColorByCoordinates(self, x: int, y: int):
        return self.imgRGB[x,y]

    def plot(self):
        plt.imshow(self.imgRGB)
        plt.show()

    def __init__(self, path2img: str, channels=None):
        if channels is None:
            channels = [1, 2, 3]

        with rasterio.open(path2img) as source:
            bands = source.read()

        self.img = bands
        self.imgRGB = self.selectChannels(channels)
        self.meanColor = list()
        self.dominantColors = list()
