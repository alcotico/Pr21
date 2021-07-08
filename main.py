import interpolation as ipl
from datagen import datagen as dg
from numpy.random import seed
from numpy import asarray, mean, var
import matplotlib.pyplot as plt
from time import time

POINTS = 50
SAMPLING_STEP = 40
SEED = 666
TSIZE = 25
SIZE = 0

if __name__ == '__main__':
    seed(SEED)
    tx, ty = dg(POINTS)
    cls = ipl.Intpl1D(tx, ty)

    ####################################################################################################################
    # сравнение четырех методов интерполяции
    ####################################################################################################################
    plt.figure(figsize=(12, 7))

    plt.subplot(2, 2, 1)
    start = time()
    res = cls.lineal(SAMPLING_STEP)
    ptime = '|time: {:.3f}ms\n'.format((time() - start) * 1000)
    x = [elm[0] for elm in res]
    y = [elm[1] for elm in res]
    label = 'Линейная\n' + ptime + '|mean: {:.3f}\n|std^2: {:.3f}'.format(mean(y), var(y))
    size = [TSIZE if elm in tx else SIZE for elm in x]
    plt.plot(x, y, c='darkgreen')
    plt.scatter(x, y, c='green', s=size, label=label)
    plt.legend()

    plt.subplot(2, 2, 2)
    start = time()
    res = cls.square(SAMPLING_STEP)
    ptime = '|time: {:.3f}ms\n'.format((time() - start) * 1000)
    x = [elm[0] for elm in res]
    y = [elm[1] for elm in res]
    label = 'Квадратичная\n' + ptime + '|mean: {:.3f}\n|std^2: {:.3f}'.format(mean(y), var(y))
    size = [TSIZE if elm in tx else SIZE for elm in x]
    plt.plot(x, y, c='darkblue')
    plt.scatter(x, y, c='blue', s=size, label=label)
    plt.legend()

    plt.subplot(2, 2, 3)
    start = time()
    res = cls.cubical(SAMPLING_STEP)
    ptime = '|time: {:.3f}ms\n'.format((time() - start) * 1000)
    x = [elm[0] for elm in res]
    y = [elm[1] for elm in res]
    label = 'Кубическая\n' + ptime + '|mean: {:.3f}\n|std^2: {:.3f}'.format(mean(y), var(y))
    size = [TSIZE if elm in tx else SIZE for elm in x]
    plt.plot(x, y, c='#993300')
    plt.scatter(x, y, c='orange', s=size, label=label)
    plt.legend()

    plt.subplot(2, 2, 4)
    start = time()
    res = cls.lagrange(SAMPLING_STEP)
    ptime = '|time: {:.3f}ms\n'.format((time() - start) * 1000)
    x = [elm[0] for elm in res]
    y = [elm[1] for elm in res]
    label = 'Многочлен Лагранжа\n' + ptime + '|mean: {:.3f}\n|std^2: {:.3f}'.format(mean(y), var(y))
    size = [TSIZE if elm in tx else SIZE for elm in x]
    plt.plot(x, y, c='darkred')
    plt.scatter(x, y, c='red', s=size, label=label)
    plt.legend()

    plt.suptitle('Сравнение методов интерполяции', fontsize=24)
    plt.show()

    ####################################################################################################################
    # сравнение улучшенных методов кубической интерполяции
    ####################################################################################################################
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
    plt.suptitle('Кубическая интерполяция', fontsize=24)

    ax1.set_title("y' = 0")
    ax1.axis('equal')
    start = time()
    res = cls.cubical_smooth(SAMPLING_STEP)
    ptime = '|time: {:.3f}ms\n'.format((time() - start) * 1000)
    x = [elm[0] for elm in res]
    y = [elm[1] for elm in res]
    label = ptime + '|mean: {:.3f}\n|std^2: {:.3f}'.format(mean(y), var(y))
    size = [TSIZE if elm in tx else SIZE for elm in x]
    colors = ['green' if elm in tx else 'red' for elm in x]
    ax1.plot(x, y, c='red', label=label)
    ax1.scatter(x, y, c=colors, s=size)
    ax1.minorticks_on()
    ax1.grid(which='major', axis='both', linestyle='--', linewidth=1.5)
    ax1.grid(which='minor', axis='both', linestyle='--', )
    ax1.legend()

    ax2.set_title("y' = (a-x)/(y-b)")
    ax2.axis('equal')
    start = time()
    res, crl = cls.cubical_smooth(SAMPLING_STEP, derivs='c')
    ptime = '|time: {:.3f}ms\n'.format((time() - start) * 1000)
    x = [elm[0] for elm in res]
    y = [elm[1] for elm in res]
    label = ptime + '|mean: {:.3f}\n|std^2: {:.3f}'.format(mean(y), var(y))
    size = [TSIZE if elm in tx else SIZE for elm in x]
    colors = ['green' if elm in tx else 'red' for elm in x]
    ax2.plot(x, y, c='red', label=label)
    ax2.scatter(x, y, c=colors, s=size)
    ax2.minorticks_on()
    ax2.grid(which='major', axis='both', linestyle='--', linewidth=1.5)
    ax2.grid(which='minor', axis='both', linestyle='--', )
    ax2.legend()

    for i in crl:
        ax2.add_artist(plt.Circle(i[0], i[1], alpha=.3, edgecolor='indigo', fill=False))

    plt.show()
