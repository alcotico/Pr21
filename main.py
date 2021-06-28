import interpolation as ipl
from datagen import datagen as dg
from numpy.random import seed
import matplotlib.pyplot as plt

POINTS = 10
SAMPLING_STEP = 20
SEED = 666
TSIZE = 25
SIZE = 2

if __name__ == '__main__':
    seed(SEED)
    tx, ty = dg(POINTS)
    cls = ipl.Intpl1D(tx, ty)

    plt.figure(figsize=(12, 7))

    plt.subplot(2, 2, 1)
    res = cls.lineal(SAMPLING_STEP)
    x = [elm[0] for elm in res]
    y = [elm[1] for elm in res]
    size = [TSIZE if elm in tx else SIZE for elm in x]
    plt.scatter(x, y, c='green', s=size, label='Линейная')
    plt.legend()

    plt.subplot(2, 2, 2)
    res = cls.square(SAMPLING_STEP)
    x = [elm[0] for elm in res]
    y = [elm[1] for elm in res]
    size = [TSIZE if elm in tx else SIZE for elm in x]
    plt.scatter(x, y, c='blue', s=size, label='Квадратичная')
    plt.legend()

    plt.subplot(2, 2, 3)
    res = cls.cubical(SAMPLING_STEP)
    x = [elm[0] for elm in res]
    y = [elm[1] for elm in res]
    size = [TSIZE if elm in tx else SIZE for elm in x]
    plt.scatter(x, y, c='orange', s=size, label='Кубическая')
    plt.legend()

    plt.subplot(2, 2, 4)
    res = cls.lagrange(SAMPLING_STEP)
    x = [elm[0] for elm in res]
    y = [elm[1] for elm in res]
    size = [TSIZE if elm in tx else SIZE for elm in x]
    plt.scatter(x, y, c='red', s=size, label='многочлен Лагранжа')
    plt.legend()

    plt.suptitle('Сравнение методов интерполяции', fontsize=24)
    plt.show()

    plt.figure(figsize=(15, 9))
    res = cls.cubical_smooth(SAMPLING_STEP)
    x = [elm[0] for elm in res]
    y = [elm[1] for elm in res]
    size = [TSIZE if elm in tx else SIZE for elm in x]
    colors = ['green' if elm in tx else 'red' for elm in x]
    plt.scatter(x, y, c=colors, s=size, label='Кубическая (сглаженная)')
    plt.legend()
    plt.show()
