
def get_height(center, z_min, sea_level_height, radius=5):
    """

    :param center: точка вершины дерева
    :param z_min: точки земли
    :param sea_level_height: миниимум по высоте для этого облака точек
    :return: высота дерева
    """
    guessed_height = []
    top_elevation = center[2]
    for bottom_point in z_min:
        if abs(center[0] - bottom_point[0]) < radius and abs(center[1] - bottom_point[1]) < radius:
            guessed_height.append(bottom_point[2])

    if not guessed_height:
        height = top_elevation - sea_level_height
    else:
        height = top_elevation - min(guessed_height)
    return height
