import numpy as np
from scipy.optimize import curve_fit


def log(x, y):
    return np.polyfit(np.log(x), y, 1)


def exp(x, y):
    return curve_fit(lambda t, a, b: a * np.exp(b * t), x, y)[0]


def poly(x, y):
    return np.polyfit(x, y, 3)


def get_params(trees: np.array):
    x = trees[:, 0]
    y = trees[:, 1]
    return [log(x, y), exp(x, y), poly(x, y)]


def fit_tree(height, funcs_params):
    log_params, exp_params, poly_params = funcs_params
    diam_log = log_params[0] * np.log(height) + log_params[1]
    diam_exp = exp_params[0] * np.exp(exp_params[1] * height)
    diam_poly = poly_params[0] * pow(height, 3) + poly_params[1] * pow(height, 2) + poly_params[2] * height + poly_params[3]
    return [diam_log, diam_exp, diam_poly]
