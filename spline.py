import numpy as np
import matplotlib.pyplot as plt
from datagen import datagen as dg


class Spline:

    def __init__(self, a=0, b=1, c=2, d=6, xcur=0):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.xcur = xcur

    def get_value(self, x):
        return self.a + self.b*(x - self.xcur) + self.c*(x - self.xcur)**2 + self.d*(x - self.xcur)**3


class SplineIpl:

    def __init__(self, x, y):
        if len(x) != len(y):
            raise ValueError("Количество точек и значений функции не совпадают.")
        if len(x) < 3:
            raise ValueError("Слишком мало точек.")
        points = len(x)
        for i in range(points):
            for j in range(i+1, points):
                if x[j] < x[i]:
                    x[j], x[i] = x[i], x[j]
                    y[j], y[i] = y[i], y[j]
        self.x = tuple(x)
        self.y = tuple(y)

        def h(idx: int):
            if idx < 1 or idx >= len(self.x):
                raise ValueError("Некорреткный индекс")
            return self.x[idx] - self.x[idx - 1]

        def nd(idx):
            d1 = (self.y[idx+1] - self.y[idx]) / (self.x[idx+1] - self.x[idx])
            d2 = (self.y[idx] - self.y[idx-1]) / (self.x[idx] - self.x[idx-1])
            return (d1 - d2) / (self.x[idx+1] - self.x[idx-1])

        def tma(matrix, fm):
            # прямой ход
            gammas = [matrix[0, 0]]
            betas = [fm[0]/gammas[0]]
            alphas = [-matrix[0, 1]/gammas[0]]
            n = matrix.shape[0]-1
            for i in range(1, n):
                gammas.append(matrix[i, i]+matrix[i, i-1]*alphas[i-1])
                betas.append((fm[i]-matrix[i, i-1]*betas[i-1])/gammas[i])
                alphas.append(-matrix[i, i+1]/gammas[i])

            gammas.append(matrix[n, n] + matrix[n, n - 1] * alphas[n - 1])
            betas.append((fm[n]-matrix[n, n - 1]*betas[n-1])/gammas[n])
            # обратный ход
            xs = [betas[len(betas)-1]]
            for i in range(n-1, -1, -1):
                xs.insert(0, alphas[i]*xs[0]+betas[i])
            # конец прогонки
            return xs

        N = len(x)
        matrix = np.zeros(shape=(N-2, N-2))
        fm = np.zeros(N-2)
        for i in range(1, N-1):
            matrix[i-1, i-1] = 2
            if i-2 >= 0:
                matrix[i-1, i-2] = h(i) / (h(i) + h(i+1))
            if i <= N-3:
                matrix[i - 1, i] = h(i+1) / (h(i) + h(i + 1))
            fm[i-1] = 6*nd(i)
        c = tma(matrix, fm)
        c.append(0)
        a = [self.y[i] for i in range(1, len(y))]
        d = [c[0]/h(1)]
        for i in range(2, len(c)+1):
            d.append((c[i-1] - c[i-2])/h(i))
        b = [c[0]*h(1)/3 + (self.y[1] - self.y[0])/(self.x[1] - self.x[0])]
        for i in range(2, len(c)+1):
            b.append(h(i) * (c[i-1]/3 + c[i-2]/6) + (self.y[i] - self.y[i-1])/(self.x[i] - self.x[i-1]))

        # тестовый пример для проверки прогонки; на выходе должно быть [0.5256, 0.628, 0.64, 1.2]
        # matrix = np.array([[5, -1, 0, 0],
        #                    [2, 4.6, -1, 0],
        #                    [0, 2, 3.6, -0.8],
        #                    [0, 0, 3, 4.4]])
        # fm = np.array([2, 3.3, 2.6, 7.2])
        # tma(matrix, fm)

        self.res = tuple(Spline(a[i-1], b[i-1], c[i-1], d[i-1], self.x[i]) for i in range(1, len(self.x)))
        print()


########################################################################################################################
# Тест
########################################################################################################################
POINTS = 10
SAMPLING_STEP = 20
SEED = 666

np.random.seed(SEED)
tx, ty = dg(POINTS)
spl = SplineIpl(tx, ty)
