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

        :param n: количество точек дискретизации одного отрезка, включая его граничные точки (значение не может быть
        меньше 3).

        :return: структуру вида Tuple[Tuple[x0, y0], Tuple[x1, y1], ... ], дискретно описывающую полученную функцию
        интерполяции.
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

        :param n: количество точек дискретизации одного отрезка, включая его граничные точки (значение не может быть
        меньше 3).

        :return: структуру вида Tuple[Tuple[x0, y0], Tuple[x1, y1], ... ], дискретно описывающую полученную функцию
        интерполяции.
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
                res.append((x, a + b * x + c * x ** 2))
        self.res = tuple(res)
        return res