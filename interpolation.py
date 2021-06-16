import numpy as np
import matplotlib.pyplot as plt


class Intpl1D:
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
        if self.res is not None:
            x = [elm[0] for elm in self.res]
            y = [elm[1] for elm in self.res]
            true_x = [tx[0] for tx in self.data]
            colors = ['green' if elm in true_x else 'red' for elm in x]
            plt.scatter(x, y, s=8, alpha=0.6, c=colors, linewidths=2, edgecolors="face")
            plt.plot(x, y, "g--", alpha=0.4)
            plt.show()
        else:
            print('Структура не определена')

    def lineal(self, n):
        nsegments = len(self.data) - 1
        res = [(self.data[0][0], self.data[0][1])]
        for i in range(nsegments):
            segment = np.linspace(self.data[i][0], self.data[i+1][0], n)
            Y2 = self.data[i+1][1]
            Y1 = self.data[i][1]
            X2 = self.data[i+1][0]
            X1 = self.data[i][0]
            for x in segment:
                if x != X1:
                    res.append((x, (Y2-Y1)/(X2-X1)*(x-X1)+Y1))
        self.res = res
        return res
