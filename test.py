from numpy import sqrt
import matplotlib.pyplot as plt

x = [0, 2, 4]
y = [0, 2, 0]

alpha = 2*(y[1] - y[2] - (x[2]-x[1])*(y[0]-y[1])/(x[1]-x[0]))
b = ((x[2] - x[1])*(y[1]**2 - y[0]**2 + x[1]**2 - x[0]**2)/(x[1]-x[0]) + x[1]**2 - x[2]**2 + y[1]**2 - y[2]**2) / alpha
a = (2*b*(y[0] - y[1]) + y[1]**2 - y[0]**2 + x[1]**2 - x[0]**2) / (x[1] - x[0]) * .5
r = sqrt((x[2] - a)**2 + (y[2] - b)**2)

fig, ax = plt.subplots(figsize=(15, 15))
ax.axis('square')
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.scatter(x, y, s=20, c='green')

# ax.add_artist(plt.Circle((2, 0), 2, alpha=.2, edgecolor='green', fill=True))
ax.add_artist(plt.Circle((a, b), r, alpha=.2, edgecolor='black', fill=False))
plt.show()
