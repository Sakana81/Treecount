class Species:

    def addColor(self, color: list):
        self.colors.append(color)

    def __init__(self, name: str, colors: list):
        self.name = name
        self.colors = colors

