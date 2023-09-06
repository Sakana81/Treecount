import random

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
"""
tree = {'diams':[10,12], 'height':


trees = [[8.45, 11.1], [7.95, 11.4], [10.5, 11.7], [10.25, 14.7], [11.55, 17.5], [12,12.4],[14.4,15.5],[13.5,12.5],
            [17.5, 21.3], [17.4, 20.8], [18.65, 21.3], [18.5, 21], [21.35, 20], [22.5, 22.3], [25.1, 21.1], [25.45, 25], 
            [26.2, 24.6], [27.5	24]]
Получить модельные деревья
Генератор функций для пород
Построить функцию зависимости для каждой породы
Получить высоту дерева
Провести вычисление диаметров из функции f(диаметр)=A*ln(высота)+B

"""

params = dict()
# heights = np.random.randint(5,50, 40)
heights = np.linspace(5, 50, 40)

treeas = np.array([[8.45, 11.1],
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


def log(x, y):
    return np.polyfit(np.log(x), y, 1)


def exp(x, y):
    f = curve_fit(lambda t, a, b: a * np.exp(b * t), x, y)
    a,b = f[0]
    return a,b
    # return np.polyfit(np.exp(x), y, 1)


def poly(x, y):
    return np.polyfit(x, y, 3)


def fitCurve(x, y, function):
    """
    fparams = np.polyfit(np.log(x),y,1) #fparams = arr[a,b], y=a*ln(x)+b
    fparams = np.polyfit(np.exp(x),y,1)
    """
    fparams = function(x, y)
    params.update({f'{function}': fparams})

    return fparams[0], fparams[1]


def getFunc(trees: np.array, func):
    y = trees[:, 0]  # diams
    x = trees[:, 1]  # heights

    a, b = fitCurve(y, x, func)  # third parameter defines type of function
    # diam = a*ln(height)+b
    return a, b


def getDiams():
    x = treeas[:,0]
    y = treeas[:,1]
    a, b = getFunc(treeas, log)
    #plt.scatter(a,b)
    #plt.show()
    a1, b1 = getFunc(treeas, exp)
    a2 = poly(x,y)
    print('diam','diam_exp', 'diam_poly', 'height')
    d = []
    for height in heights:
        diam = a * np.log(height) + b
        diam_exp = a1 * np.exp(b1 * height)
        diam_poly = a2[0]*pow(height,3) + a2[1]*pow(height,2) + a2[2]*height + a2[3]
        d.append([height,diam,diam_exp, diam_poly])
        print(diam, diam_exp, diam_poly, height)

    d = np.array(d)
    """
    plt.scatter(d[:,0],d[:,1], color='r')
    plt.scatter(d[:,0],d[:,2], color='g')
    plt.scatter(d[:,0],d[:,3], color='y')
    plt.scatter(treeas[:,0],treeas[:,1], color='b')

    plt.show()
    """
    return d