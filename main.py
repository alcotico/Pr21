import interpolation as ipl

if __name__ == '__main__':
    x = [0, 10, 20]
    y = [2, 12, 36]
    cls = ipl.Intpl1D(x, y)
    cls.lineal(5)
    cls.show()
