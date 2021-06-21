import interpolation as ipl
from datagen import datagen as dg
from numpy.random import seed

POINTS = 10
SAMPLING_STEP = 20
SEED = 666

if __name__ == '__main__':
    seed(SEED)
    x, y = dg(POINTS)
    cls = ipl.Intpl1D(x, y)
    cls.square(SAMPLING_STEP)
    cls.show()
