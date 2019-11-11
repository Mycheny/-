#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
二维问题的梯度下降法示例
"""
import math
import random

import numpy as np
from scatter3d.scatter3d import scatter3d


def func_2d(x):
    """
    目标函数
    :param x: 自变量，二维向量
    :return: 因变量，标量
    """
    # return - math.exp(-(x[0] ** 2 + x[1] ** 2))
    return - np.exp(-(np.square(x[0]) + np.square(x[1])))


def grad_2d(x):
    """
    目标函数的梯度
    :param x: 自变量，二维向量
    :return: 因变量，二维向量
    """
    deriv0 = 2 * x[0] * math.exp(-(x[0] ** 2 + x[1] ** 2))
    deriv1 = 2 * x[1] * math.exp(-(x[0] ** 2 + x[1] ** 2))
    return np.array([deriv0, deriv1])


def gradient_descent_2d(grad, cur_x=np.array([0.1, 0.1]), learning_rate=0.01, precision=0.0001, max_iters=10000):
    """
    二维问题的梯度下降法
    :param grad: 目标函数的梯度
    :param cur_x: 当前 x 值，通过参数可以提供初始值
    :param learning_rate: 学习率，也相当于设置的步长
    :param precision: 设置收敛精度
    :param max_iters: 最大迭代次数
    :return: 局部最小值 x*
    """
    print(f"{cur_x} 作为初始值开始迭代...")
    for i in range(max_iters):
        grad_cur = grad(cur_x)
        if np.linalg.norm(grad_cur, ord=2) < precision:
            break  # 当梯度趋近为 0 时，视为收敛
        cur_x = cur_x - grad_cur * learning_rate
        if i % 1 == 0: print("第", i, "次迭代：x 值为 ", cur_x)
        # if i > 4840000: print("第", i, "次迭代：x 值为 ", cur_x)

    print("局部最小值 x =", cur_x)
    return cur_x


if __name__ == '__main__':
    gradient_descent_2d(grad_2d, cur_x=np.array([1, -1]), learning_rate=0.2, precision=0.00000001, max_iters=100000000)
