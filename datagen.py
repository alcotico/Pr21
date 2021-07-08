from numpy.random import *


def datagen(n, isint: bool = True):
    """
    Генерирует исходный набор точек данных заданного количества.

    :param n: количество точек данных;
    :param isint: целочисленность значений.
    :return: два набора значений x, y, описывающих аргументы функции и ее значения в этих точках.
    """
    '''зададим пределы генерации значений для x и y, т.е. x∊[xlow, xhigh) и  x∊[yxlow, yhigh)'''
    xlow = 0
    xhigh = 500
    ylow = 0
    yhigh = 200

    x = []
    y = []

    ''' в зависимости от указанного значения параметра @isint генерируются либо целочисленные, либо вещественные
        значения; параллельно происходит проверка их уникальности для искючения ситуации, когда для одного и того же
        аргумента определены разные значения функции'''
    if isint:
        for i in range(n):
            val = randint(xlow, xhigh)  # генерация данных
            # проверка уникальности
            while val in x:
                val = randint(xlow, xhigh)
            x.append(val)  # добавление корректных данных
            val = randint(ylow, yhigh)  # генерация данных
            # проверка уникальности
            while val in y:
                val = randint(ylow, yhigh)
            y.append(val)  # добавление корректных данных
    else:
        for i in range(n):
            val = (xhigh - xlow) * random_sample(n) + xlow  # генерация данных
            # проверка уникальности
            while val in x:
                val = (xhigh - xlow) * random_sample(n) + xlow
            x.append(val)  # добавление корректных данных
            val = (yhigh - ylow) * random_sample(n) + ylow  # генерация данных
            # проверка уникальности
            while val in y:
                val = (yhigh - ylow) * random_sample(n) + ylow
            y.append(val)  # добавление корректных данных

    return x, y
