import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure()
ax = Axes3D(fig)

x = np.linspace(-5, 5, 200)
y = 2*x+1
# x = 10
# y = 20

w = np.linspace(-5, 5, 200)
b = np.linspace(-5, 5, 200)
w, b = np.meshgrid(w, b)

# Z = -np.exp(-(w ** 2 + b ** 2))

# Z = np.square(w*x+b-y)
Z = np.square(w)+np.square(b)
plt.xlabel('w',fontsize=12,color='red',rotation=60,verticalalignment='top')
plt.ylabel('b',fontsize=14,color='blue',rotation=30,horizontalalignment='center')
ax.view_init(elev=90., azim=90)
# ax.view_init(elev=0., azim=90)
# ax.view_init(elev=0., azim=0)

ax.plot_surface(w, b, Z, rstride=1, cstride=1, cmap='rainbow')
plt.show()
