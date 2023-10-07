
def getHeight(centers, z_min, sea_level_height):
    """

    :param centers: точки вершин деревьев
    :param z_min: точки земли
    :param sea_level_height: миниимум по высоте для этого облака точек
    :return: высоты деревьев
    """
    rad = 5
    heights = []
    for top_point in centers:

        guessed_height = []
        top_elevation = top_point[2]
        for bottom_point in z_min:

            diff_x = top_point[0] - bottom_point[0]
            diff_y = top_point[1] - bottom_point[1]
            if abs(diff_x) < rad and abs(diff_y) < rad:
                guessed_height.append(bottom_point[2])

        if not guessed_height:
            diff_z = top_elevation - sea_level_height
        else:
            diff_z = top_elevation - min(guessed_height)
        heights.append(diff_z)
    return heights
