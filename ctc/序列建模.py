import numpy as np

np.random.seed(1111)

T, V = 2, 3
m, n = 3, V

x = np.random.random([T, m])  # T x m
w = np.random.random([m, n])  # weights, m x n


def softmax(logits):
    max_value = np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits - max_value)
    exp_sum = np.sum(exp, axis=1, keepdims=True)
    dist = exp / exp_sum
    return dist


def toy_nw(x):
    y = np.matmul(x, w)  # T x n
    y = softmax(y)
    return y


y = toy_nw(x)
print(y)
print(y.sum(1, keepdims=True))


def forward(y, labels):
    T, V = y.shape
    L = len(labels)
    alpha = np.zeros([L, T])

    # init
    alpha[0, 0] = y[0, labels[0]]
    alpha[0, 1] = y[0, labels[1]]

    for t in range(1, T):
        for i in range(L):
            s = labels[i]

            a = alpha[i, t - 1]
            if i - 1 >= 0:
                a += alpha[i - 1, t - 1]
            if i - 2 >= 0 and s != 0 and s != labels[i - 2]:
                a += alpha[i - 2, t - 1]

            alpha[i, t] = a * y[t, s]

    return alpha


labels = [0, 2, 0]  # 0 for blank
# labels = [0, 2, 0]  # 0 for blank
y = np.ones_like(y, dtype=y.dtype)*0.1
y[0, 0] = 0.5
y[0, 1] = 0.2
y[0, 2] = 0.3
y[1, 0] = 0.4
y[1, 1] = 0.3
y[1, 2] = 0.3
print(y)
print(y.sum(1, keepdims=True))
index = np.argmax(y, axis=1)
alpha = forward(y, labels)
print(alpha)

# 最后可以得到似然 p(l|x)=αT(|l′|)+αT(|l′|−1)
p = alpha[-1, -1] + alpha[-2, -1]
print(p)

# def backward(y, labels):
#     T, V = y.shape
#     L = len(labels)
#     beta = np.zeros([T, L])
#
#     # init
#     beta[-1, -1] = y[-1, labels[-1]]
#     beta[-1, -2] = y[-1, labels[-2]]
#
#     for t in range(T - 2, -1, -1):
#         for i in range(L):
#             s = labels[i]
#
#             a = beta[t + 1, i]
#             if i + 1 < L:
#                 a += beta[t + 1, i + 1]
#             if i + 2 < L and s != 0 and s != labels[i + 2]:
#                 a += beta[t + 1, i + 2]
#
#             beta[t, i] = a * y[t, s]
#
#     return beta
#
#
# beta = backward(y, labels)
# print(beta)
