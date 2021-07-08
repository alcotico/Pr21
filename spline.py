import numpy as np
from math import factorial as fct
import matplotlib.pyplot as plt
from datagen import datagen as dg
from time import time


class Spline:
    """
    Класс описывающий сплайн как аналитическую кривую в виде кубического полинома, где a, b, c, d - соответствующие
    коэффициенты данной кривой, а xcur - значение узла данных, относительно которого идет построение.
    """

    def __init__(self, a=0, b=1, c=2, d=6, xcur=0):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.xcur = xcur

    def get_value(self, x):
        """
        Метода класса, позволяющий получить значение функции в заданной точке.
        :param x: точка для рассчета
        :return: значение функции в данной точке.
        """
        return self.a + self.b*(x - self.xcur) + self.c / 2 * (x - self.xcur)**2 + self.d / 6 * (x - self.xcur)**3


class SplineIpl:
    """
    Класс, позволяющий проводить сплайн-интерполяцию по исходным данным.
    На вход подаются два массива х и у, которые обозначают узлы сетки и истинные значения в них, по которым и будет
    проводится интерполяция.

    В данном классе используется приведение матрицы коэффициентов к трех-диагональному виду и осуществляется ее решение
    методом прогонки.
    """
    def __init__(self, x, y):

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

        if len(x) != len(y):
            raise ValueError("Количество точек и значений функции не совпадают.")
        if len(x) < 3:
            raise ValueError("Слишком мало точек.")

        start = time()
        points = len(x)
        for i in range(points):
            for j in range(i+1, points):
                if x[j] < x[i]:
                    x[j], x[i] = x[i], x[j]
                    y[j], y[i] = y[i], y[j]
        self.x = tuple(x)
        self.y = tuple(y)

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
        end = time()
        self.time = end - start
        print()

    def show(self, s):
        """
        Позволяет графически отобразить полученное решение.
        :param s: число промежуточных точек для интерполяции, помимо исходных.
        :return: None.
        """
        segment = list(np.linspace(self.x[0], self.x[len(spl.x) - 1], s + 1))
        last = 1
        for i in range(len(segment)):
            if self.x[last] < segment[i]:
                segment.insert(i, self.x[last])
                if last + 1 == len(self.x)-1:
                    break
                else:
                    last += 1
        fig, ax = plt.subplots(figsize=(15, 15))
        plt.suptitle('СПЛАЙН-ИНТЕРПОЛЯЦИЯ (3-х диаг. матрица)', fontsize=30)
        # ax.axis('equal')
        func = []
        for x in segment:
            if x in self.x:
                func.append(self.y[self.x.index(x)])
            else:
                for lim in range(1, len(self.x)):
                    if x < self.x[lim]:
                        func.append(self.res[lim-1].get_value(x))
                        break
                    else:
                        continue
        size = [25 if elm in self.x else 0 for elm in segment]
        colors = ['green' if elm in self.x else 'red' for elm in segment]
        plt.title('Затраченное время на рассчет интерполяционной кривой {:.3f}ms\n'.format(self.time * 1000) +
                  'mean: {:.3f} | std^2: {:.3f}'.format(np.mean(func), np.var(func)), fontsize=12,
                  loc='left')
        ax.scatter(segment, func, c=colors, s=size)
        ax.plot(segment, func)
        ax.minorticks_on()
        ax.grid(which='major', axis='both', linestyle='--', linewidth=1.5)
        ax.grid(which='minor', axis='both', linestyle='--', )
        plt.show()
        # print(len(segment) == len(func))
        return None


class SplineIpl2:
    """
    Класс, позволяющий проводить сплайн-интерполяцию по исходным данным.
    На вход подаются два массива х и у, которые обозначают узлы сетки и истинные значения в них, по которым и будет
    проводится интерполяция.

    В данном классе используется составление матрицы коэффициентов "в лоб" по заданным условиям непрерывности и
    гладкости с последующим ее решением посредством библиотеки NumPy (linalg.solve).
    """
    def __init__(self, x, y):
        if len(x) != len(y):
            raise ValueError("Количество точек и значений функции не совпадают.")
        if len(x) < 3:
            raise ValueError("Слишком мало точек.")
        start = time()
        points = len(x)
        for i in range(points):
            for j in range(i+1, points):
                if x[j] < x[i]:
                    x[j], x[i] = x[i], x[j]
                    y[j], y[i] = y[i], y[j]
        self.x = tuple(x)
        self.y = tuple(y)

        def h(idx: int):
            return self.x[idx+1] - self.x[idx]

        coefs = np.zeros(shape=(4*(points-1), 4*(points-1)))
        fmems = np.zeros(shape=(4*(points-1)))

        # А_n-1 = Y_n-1
        coefs[0, 4*(points-2)] = 1
        fmems[0] = self.y[points-2]

        # А_n-1 + B_n-1*H_n-1 + C_n-1/2 * (H_n-1)^2 + D_n-1/6 *(H_n-1)^3 = Y_n
        buf = [0 for i in range(4*(points-2))]
        buf.extend([h(points-2)**idx/fct(idx) for idx in range(4)])
        coefs[1] = np.array(buf)
        del buf
        fmems[1] = self.y[points-1]

        # C_0 = 0
        coefs[2, 2] = 1
        # fmems[2] = 0

        # C_n-1 + D_n-1 * H_n-1 = 0
        coefs[3, 4*(points-2)+2] = 1
        coefs[3, 4*(points-2)+3] = h(points-2)
        # fmems[3] = 0

        cur_row = 4
        for i in range(points-2):
            # А_i = Y_i
            coefs[cur_row, 4*i] = 1
            fmems[cur_row] = self.y[i]
            cur_row += 1

            # А_i + B_i*H_i + C_i/2 * (H_i)^2 + D_i/6 * (H_i)^3 = Y_i+1
            coefs[cur_row, 4*i] = 1
            coefs[cur_row, 4*i + 1] = h(i)
            coefs[cur_row, 4*i + 2] = h(i)**2 / 2
            coefs[cur_row, 4*i + 3] = h(i)**3 / 6
            fmems[cur_row] = self.y[i+1]
            cur_row += 1

            # B_i + C_i * H_i + D_i * (H_i)^2 = B_i+1
            coefs[cur_row, 4 * i + 1] = 1
            coefs[cur_row, 4 * (i+1) + 1] = -1
            coefs[cur_row, 4 * i + 2] = h(i)
            coefs[cur_row, 4 * i + 3] = h(i) ** 2 / 2
            # fmems[cur_row] = 0
            cur_row += 1

            # C_i + D_i * H_i = C_i+1
            coefs[cur_row, 4*i + 2] = 1
            coefs[cur_row, 4*(i+1) + 2] = -1
            coefs[cur_row, 4*i + 3] = h(i)
            # fmems[cur_row] = 0
            cur_row += 1

        abcd = np.linalg.solve(coefs, fmems)
        self.splines = []
        for i in range(points-1):
            self.splines.append(Spline(abcd[4*i], abcd[4*i+1], abcd[4*i+2], abcd[4*i+3], self.x[i]))

        self.time = time() - start

    def show(self, s):
        """
        Позволяет графически отобразить полученное решение.

        :param s: число промежуточных точек для интерполяции, помимо исходных.
        :return: None
        """
        segment = list(np.linspace(self.x[0], self.x[len(spl.x) - 1], s + 1))
        last = 1
        for i in range(len(segment)):
            if self.x[last] < segment[i]:
                segment.insert(i, self.x[last])
                if last + 1 == len(self.x)-1:
                    break
                else:
                    last += 1

        fig, ax = plt.subplots(figsize=(15, 15))
        plt.suptitle('СПЛАЙН-ИНТЕРПОЛЯЦИЯ ("в лоб")', fontsize=30)
        # ax.axis('equal')
        func = []
        for x in segment:
            if x in self.x:
                func.append(self.y[self.x.index(x)])
            else:
                for lim in range(1, len(self.x)):
                    if x < self.x[lim]:
                        func.append(self.splines[lim-1].get_value(x))
                        break
                    else:
                        continue
        size = [25 if elm in self.x else 0 for elm in segment]
        colors = ['green' if elm in self.x else 'red' for elm in segment]
        ax.scatter(segment, func, c=colors, s=size)
        ax.plot(segment, func, c='indigo')
        plt.title('Затраченное время на рассчет интерполяционной кривой {:.3f}ms\n'.format(self.time*1000) +
                  'mean: {:.3f} | std^2: {:.3f}'.format(np.mean(func), np.var(func)), fontsize=12,
                  loc='left')
        ax.minorticks_on()
        ax.grid(which='major', axis='both', linestyle='--', linewidth=1.5)
        ax.grid(which='minor', axis='both', linestyle='--',)
        plt.show()
        return None


########################################################################################################################
# Тест
########################################################################################################################
POINTS = 100
SAMPLING = 1000
SEED = 666

np.random.seed(SEED)
tx, ty = dg(POINTS)
spl = SplineIpl(tx, ty)
spl.show(SAMPLING)
spl = SplineIpl2(tx, ty)
spl.show(SAMPLING)
