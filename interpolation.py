import numpy as np
import matplotlib.pyplot as plt


class Intpl1D:
    """
    Класс, предназняченный для интерполяции исходных данных, представленных одномерно, и ее отображения.

    Методы класса:

    lineal(n: int) - возвращает кусочно-линейную интерполяцию данных;
    square(n: int) - возвращает кусочно-квадратичную интерполяцию данных;
    show() - визуализирует интерполированные данные.
    """
    def __init__(self, x, y):
        if len(x) != len(y):
            raise KeyError("Количество значений интерполируемой функции не совпадает с количеством точек")
        else:
            data = []
            for i in range(len(x)):
                data.append((x[i], y[i]))
            self.data = np.asarray(sorted(data, key=lambda f: f[0]))
            self.res = None

    def show(self):
        """
        Визуализирует интерполированные данные.

        :return: нет возвращаемого значения.
        """
        if self.res is not None:
            x = [elm[0] for elm in self.res]
            y = [elm[1] for elm in self.res]
            true_x = [tx[0] for tx in self.data]
            colors = ['green' if elm in true_x else 'red' for elm in x]
            size = [25 if elm in true_x else 2 for elm in x]
            plt.scatter(x, y, s=size, alpha=0.6, c=colors, linewidths=2, edgecolors="face")
            plt.plot(x, y, "g--", alpha=0.4)
            plt.show()
        else:
            raise ValueError('Структура не определена')

    def lineal(self, n: int):
        """
        Проводит кусочно-линейную интерполяцию данных по двум точкам.

        :param n: количество точек дискретизации одного отрезка, включая его граничные точки (значение не может быть меньше 3).

        :return: структуру вида Tuple[Tuple[x0, y0], Tuple[x1, y1], ... ], дискретно описывающую полученную функцию интерполяции.
        """

        ''' @nsegments - количество отрезков интерполяции по кол-ву точек данных '''

        nsegments = len(self.data) - 1

        ''' поскольку данная интерполяция строится минимум по двум точкам, необходимо проверять исходные данные на их
            наличие; также число промежуточных точек на каждом отрезке из соображений здравого смысла не может быть
            меньше одного, из чего следует, что @n >= 3'''

        if nsegments < 1:
            raise ValueError("Количество точек для данной интерполяции слишком мало (необходимо минимум 2)")
        if n < 3:
            n = 3

        ''' для избежания проблемы повторения узлов данных на каждой итерации не будем запоминать первую пару
            аргумент-значение из очередного интервала, предварительно запомнив первое достоверное значение на области
            интерполяции '''

        res = [(self.data[0][0], self.data[0][1])]
        for i in range(nsegments):

            ''' в цикле на каждой итерации определим аргументы функции на текущем отрезке, исходя из количества шагов
                дискретизации @n в переменную @segment'''

            segment = np.linspace(self.data[i][0], self.data[i+1][0], n)

            ''' для сокращения количества обращений к массиву исходных данных запомним их в отдельные переменные'''

            y2 = self.data[i+1][1]
            y1 = self.data[i][1]
            x2 = self.data[i+1][0]
            x1 = self.data[i][0]

            ''' для каждого аргумента текущего отрезка определим значение функции по формуле
                f(x) = f(x_i) + ( f(x_i+1) - f(x_i) ) / (x_i+1 - x_i) * (x - x_i)'''

            for x in segment:
                if x != x1:
                    res.append((x, (y2-y1)/(x2-x1)*(x-x1)+y1))
        self.res = tuple(res)
        return res

    def square(self, n: int):
        """
        Проводит кусочно-квадратичную интерполяцию данных по трем точкам.

        :param n: количество точек дискретизации одного отрезка, включая его граничные точки (значение не может быть меньше 3).

        :return: структуру вида Tuple[Tuple[x0, y0], Tuple[x1, y1], ... ], дискретно описывающую полученную функцию интерполяции.
        """

        ''' @points - число точек исходных данных '''

        points = len(self.data)

        ''' поскольку данная интерполяция строится минимум по трем точкам, необходимо проверять исходные данные на их
            наличие; также число промежуточных точек на каждом отрезке из соображений здравого смысла не может быть
            меньше одного, из чего следует, что @n >= 3'''

        if points < 3:
            raise ValueError("Количество точек для данной интерполяции слишком мало (необходимо минимум 3)")
        if n < 3:
            n = 3

        ''' для избежания проблемы повторения узлов данных на каждой итерации не будем запоминать первую пару
            аргумент-значение из очередного интервала, предварительно запомнив первое достоверное значение на области
            интерполяции '''

        res = [(self.data[0][0], self.data[0][1])]
        a = b = c = 0
        for i in range(points - 2):

            ''' для сокращения количества обращений к массиву исходных данных запомним их в отдельные переменные:
                > xl = x_i  # x_left
                > xm = x_i+1  # x_middle
                > xr = x_i+2  # x_right
                > fl = f(x_i)
                > fm = f(x_i+1)
                > fr = f(x_i+2)
            '''

            xl = self.data[i][0]
            xm = self.data[i+1][0]
            xr = self.data[i+2][0]
            fl = self.data[i][1]
            fm = self.data[i+1][1]
            fr = self.data[i+2][1]

            ''' для сокращения количества вычислений запомним в отдельную переменную @frac значение, используемое
                в нескольких местах '''

            frac = (fm - fl) / (xm - xl)

            ''' по соответствующим формулам расчитаем коэффициента полинома a + bx + cx^2 на участке [x_i, x_i+1] '''
            c = ((fr - fl) / (xr - xl) - frac) / (xr - xm)
            b = frac - c*(xm + xl)
            a = fl - b*xl - c*xl**2

            ''' для каждого аргумента участка [x_i, x_i+1] определим значение функции, подставив значение аргумента
                в формулу полинома '''

            segment = np.linspace(self.data[i][0], self.data[i + 1][0], n)
            for x in segment:
                if x != self.data[i][0]:
                    res.append((x, a + b*x + c*x**2))

        ''' поскольку для построение параболы необходимо три точки, возникает проблема с определенем коэффициентов 
            полинома последнего участка, поэтому берутся коэффициенты предыдущего '''

        segment = np.linspace(self.data[points - 2][0], self.data[points - 1][0], n)
        for x in segment:
            if x != self.data[points - 2][0]:
                res.append((x, a + b*x + c*x**2))
        self.res = tuple(res)
        return res

    def cubical(self, n: int):
        """
        Проводит кусочно-кубическую интерполяцию данных по четырем точкам.

        :param n: количество точек дискретизации одного отрезка, включая его граничные точки (значение не может быть меньше 3).

        :return: структуру вида Tuple[Tuple[x0, y0], Tuple[x1, y1], ... ], дискретно описывающую полученную функцию интерполяции.
        """

        ''' @points - число точек исходных данных '''

        points = len(self.data)

        # поскольку данная интерполяция строится минимум по четырем точкам, необходимо проверять исходные данные на их
        # наличие; также число промежуточных точек на каждом отрезке из соображений здравого смысла не может быть
        # меньше одного, из чего следует, что @n >= 3

        if points < 4:
            raise ValueError("Количество точек для данной интерполяции слишком мало (необходимо минимум 3)")
        if n < 3:
            n = 3

        # для избежания проблемы повторения узлов данных на каждой итерации не будем запоминать первую пару
        # аргумент-значение из очередного интервала, предварительно запомнив первое достоверное значение на области
        # интерполяции

        res = [(self.data[0][0], self.data[0][1])]
        a = b = c = d = 0
        for i in range(points - 3):
            x0 = self.data[i][0]
            x1 = self.data[i+1][0]
            x2 = self.data[i+2][0]
            x3 = self.data[i+3][0]
            f0 = self.data[i][1]
            f1 = self.data[i+1][1]
            f2 = self.data[i+2][1]
            f3 = self.data[i+3][1]
            FRAC1 = (f1 - f0) / (x1 - x0)
            FRAC2 = (f2 - f0) / (x2 - x0)

            d = ((f3 - f0) / ((x3 - x0) * (x3 - x1)) - FRAC2 / (x2 - x1)) / (x3 - x2) + FRAC1 / ((x2 - x1) * (x3 - x1))
            c = (FRAC2 - FRAC1) / (x2 - x1) - d * (x0 + x1 + x2)
            b = FRAC1 - c * (x1 + x0) - d * (x1**2 + x1*x0 + x0**2)
            a = f0 - b*x0 - c*x0**2 - d*x0**3

            # по соответствующим формулам расчитаем коэффициента полинома a + bx + cx^2 + dx^3 на участке
            # [x_i, x_i+1]

            segment = np.linspace(self.data[i][0], self.data[i + 1][0], n)
            for x in segment:
                if x != self.data[i][0]:
                    res.append((x, a + b*x + c*x**2 + d*x**3))

        for j in range(points - 3, points - 1):
            segment = np.linspace(self.data[j][0], self.data[j + 1][0], n)
            for x in segment:
                if x != self.data[j][0]:
                    res.append((x, a + b * x + c * x ** 2 + d * x ** 3))

        self.res = tuple(res)
        return res

    def lagrange(self, n: int):
        """
        Проводит интерполяцию данных многочленом Лагранжа.

        :param n: количество точек дискретизации одного отрезка, включая его граничные точки (значение не может быть меньше 3).

        :return: структуру вида Tuple[Tuple[x0, y0], Tuple[x1, y1], ... ], дискретно описывающую полученную функцию интерполяции.
        """

        points = len(self.data)
        res = [(self.data[0][0], self.data[0][1])]

        def polynomial(arg):
            """
            Вспомогательная функция для определения вида полинома Лагранжа

            :param arg: значение аргумента функции
            :return: значение функции в заданной точке
            """
            composition = 1
            summa = 0
            for i in range(points):
                for j in range(points):
                    if i != j:
                        composition *= (arg - self.data[j][0]) / (self.data[i][0] - self.data[j][0])
                    else:
                        continue
                summa += self.data[i][1]*composition
                composition = 1
            return summa

        func_domain = [self.data[0][0]]  # переменная для области определения функции
        for k in range(points - 1):
            func_domain.extend(list(np.linspace(self.data[k][0], self.data[k + 1][0], n))[1:])
        # на выходе цикла получаем соответственно всю область определения функции

        tx = list(np.squeeze(self.data[:, 0:1]))  # список исходных ординат
        # если аргумент из области определения функции является изначально заданым, берется заданное значение функции,
        # в противном случае - оно высчитывается полученным многочленом Лагранжа
        for x in func_domain:
            if x in tx:
                idx = tx.index(x)
                res.append((x, self.data[idx][1]))
            else:
                res.append((x, polynomial(x)))

        self.res = tuple(res)
        return res

    def cubical_smooth(self, n: int):

        points = len(self.data)
        res = [(self.data[0][0], self.data[0][1])]
        derivs = [0 for i in range(points)]

        for i in range(points - 1):
            y1 = self.data[i+1][1]
            y0 = self.data[i][1]
            x1 = self.data[i+1][0]
            x0 = self.data[i][0]

            d = (derivs[i+1] + derivs[i]) / (x1 - x0)**2 - 2 * (y1 - y0) / (x1 - x0)**3
            c = (derivs[i] - (y1 - y0) / (x1 - x0) + d * (x1**2 + x1*x0 - 2*x0**2)) / (x0 - x1)
            b = (y1 - y0) / (x1 - x0) - c*(x1 + x0) - d*(x1**2 + x1*x0 + x0**2)
            a = y0 - b*x0 - c*x0**2 - d*x0**3

            segment = np.linspace(self.data[i][0], self.data[i + 1][0], n)[1:-1]
            for x in segment:
                res.append((x, a + b*x + c*x**2 + d*x**3))
            res.append((x1, y1))

        self.res = tuple(res)
        return res
