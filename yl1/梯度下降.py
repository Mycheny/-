import math


def sum_of_gradient(x, y, thetas):
    """计算梯度向量，参数分别是x和y轴点坐标数据以及方程参数"""
    m = len(x)
    grad0 = 1.0 / m * sum([(thetas[0] + thetas[1] * x[i] - y[i]) for i in range(m)])
    grad1 = 1.0 / m * sum([(thetas[0] + thetas[1] * x[i] - y[i]) * x[i] for i in range(m)])
    return [grad0, grad1]


def step(thetas, direction, step_size):
    """move step_size in the direction from thetas"""
    return [thetas_i + step_size * direction_i
            for thetas_i, direction_i in zip(thetas, direction)]


def distance(v, w):
    """两点的距离"""
    return math.sqrt(squared_distance(v, w))


def squared_distance(v, w):
    vector_subtract = [v_i - w_i for v_i, w_i in zip(v, w)]
    return sum(vector_subtract_i * vector_subtract_i for vector_subtract_i, vector_subtract_i
               in zip(vector_subtract, vector_subtract))


def gradient_descent(stepSize, x, y, tolerance=0.000000001, max_iter=100000):
    """梯度下降"""
    iter = 0
    # initial theta
    thetas = [0, 0]
    # Iterate Loop
    while True:
        gradient = sum_of_gradient(x, y, thetas)

        next_thetas = step(thetas, gradient, stepSize)

        if distance(next_thetas, thetas) < tolerance:  # stop if we're converging
            break
        thetas = next_thetas  # continue if we're not

        iter += 1  # update iter

        if iter == max_iter:
            print('Max iteractions exceeded!')
            break

    return thetas


x = [1, 2, 3]
y = [5, 9, 13]
stepSize = 0.001
t0, t1 = gradient_descent(-stepSize, x, y)
print(t0, " ", t1)
