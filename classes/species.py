class Species:

    def __init__(self, name: str, colors: list):
        self.name = name
        self.colors = colors
        self.diam_extr_func_log_params = list()
        self.diam_extr_func_exp_params = list()
        self.diam_extr_func_poly_params = list()

    def addColor(self, color: list):
        self.colors.append(color)

    def add_func_params(self, log: list, exp: list, poly: list):
        self.diam_extr_func_log_params = log
        self.diam_extr_func_exp_params = exp
        self.diam_extr_func_poly_params = poly
