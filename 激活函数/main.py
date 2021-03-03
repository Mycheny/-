# -*- coding: utf-8 -*- 
# @File main.py
# @Time 2021/3/3 10:11
# @Author wcy
# @Software: PyCharm
# @Site
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1, 1, 200, dtype=np.float32)


def show(x, y, z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'), alpha=0.5)
    ax.set_xlabel('w')
    ax.set_ylabel('b')
    plt.show()


def func1(w, b):
    global x
    y = w * x + b
    # y = 1 / (1 + np.exp(-y)) # sigmoid激活函数
    return y


def func2(w, b):
    global x
    y = w * x + b
    y = 1/(1+np.exp(-y)) # sigmoid激活函数
    return y


def cost1(y, y_):
    J = np.square(y - y_)
    return J


def cost2(y, y_):
    J = -(y_*np.log(y)-(1-y_)*np.log(1-y))
    return J


if __name__ == '__main__':
    # y=w*x+b
    # J=(y-y_)^2=(w*x+b-y_)^2
    #
    # y=w*x+b
    # J=-(y_*log(y)-(1-y_)*log(1-y))=-(y_*log(w*x+b)-(1-y_)*log(1-(w*x+b)))

    y_1 = func1(2, 0)  # 计算当w为2，b为0时，y的值，以此值为标签
    # y_2 = func2(2, 0)

    w, b = np.meshgrid(np.linspace(-50, 50, 30), np.linspace(-30, 50, 30))
    J1 = np.array([np.mean(cost1(func1(w_, b_), y_1)) for w_, b_ in zip(w.flatten(), b.flatten())]).reshape(w.shape)
    # J2 = np.array([np.mean(cost2(func2(w_, b_), y_2)) for w_, b_ in zip(w.flatten(), b.flatten())]).reshape(w.shape)

    show(w, b, J1)
    # show(w, b, J2)
