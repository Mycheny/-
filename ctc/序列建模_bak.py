import numpy as np
import tensorflow as tf

np.random.seed(1111)

T, V = 4, 3
m, n = 4, V

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
    alpha = np.zeros([T, L])

    # init
    alpha[0, 0] = y[0, labels[0]]
    alpha[0, 1] = y[0, labels[1]]

    for t in range(1, T):
        for i in range(L):
            s = labels[i]

            a = alpha[t - 1, i]
            if i - 1 >= 0:
                a += alpha[t - 1, i - 1]
            if i - 2 >= 0 and s != 0 and s != labels[i - 2]:
                a += alpha[t - 1, i - 2]

            alpha[t, i] = a * y[t, s]

    return alpha


labels = [0, 1, 0, 1, 0, 2, 0]  # 0 for blank
alpha = forward(y, labels)
print(alpha)


def backward(y, labels):
    T, V = y.shape
    L = len(labels)
    beta = np.zeros([T, L])

    # init
    beta[-1, -1] = y[-1, labels[-1]]
    beta[-1, -2] = y[-1, labels[-2]]

    for t in range(T - 2, -1, -1):
        for i in range(L):
            s = labels[i]

            a = beta[t + 1, i]
            if i + 1 < L:
                a += beta[t + 1, i + 1]
            if i + 2 < L and s != 0 and s != labels[i + 2]:
                a += beta[t + 1, i + 2]

            beta[t, i] = a * y[t, s]

    return beta


beta = backward(y, labels)
print(beta)


def gradient(y, labels):
    T, V = y.shape
    L = len(labels)

    alpha = forward(y, labels)
    beta = backward(y, labels)
    p = alpha[-1, -1] + alpha[-1, -2]

    grad = np.zeros([T, V])
    for t in range(T):
        for s in range(V):
            lab = [i for i, c in enumerate(labels) if c == s]
            for i in lab:
                grad[t, s] += alpha[t, i] * beta[t, i]
            grad[t, s] /= y[t, s] ** 2

    grad /= p
    return grad


grad = gradient(y, labels)
print(grad)


def probability(y, labels):
    T, V = y.shape
    L = len(labels)

    alpha = forward(y, labels)
    beta = backward(y, labels)
    p = alpha[-1, -1] + alpha[-1, -2]

    pro = 0
    for t in range(T):
        for s in range(V):
            lab = [i for i, c in enumerate(labels) if c == s]
            for i in lab:
                pro += alpha[t, i] * beta[t, i] / y[t, s]

    # pro /= p
    return pro


labels = [0, 1, 0, 2, 0, 1, 0]
y = np.ones_like(y, dtype=y.dtype) * 0.009
y[0, 1] = 0.9
y[1, 2] = 0.9
y[2, 1] = 0.9
y[3, 1] = 0.9
# y[4, 0] = 0.9
# y[5, 0] = 0.9
# y[6, 2] = 0.9
# y[7, 3] = 0.9
# y[8, 4] = 0.9
# y[9, 0] = 0.9
# y[10, 0] = 0.9
# y[11, 0] = 0.9

print(y)

pro = probability(y, labels)
print("pro", pro)

"""
pro 3.67069551949757
pro 3.634711874108333
"""


def check_grad(y, labels, w=-1, v=-1, toleration=1e-3):
    grad_1 = gradient(y, labels)[w, v]

    delta = 1e-10
    original = y[w, v]

    y[w, v] = original + delta
    alpha = forward(y, labels)
    log_p1 = np.log(alpha[-1, -1] + alpha[-1, -2])

    y[w, v] = original - delta
    alpha = forward(y, labels)
    log_p2 = np.log(alpha[-1, -1] + alpha[-1, -2])

    y[w, v] = original

    grad_2 = (log_p1 - log_p2) / (2 * delta)
    if np.abs(grad_1 - grad_2) > toleration:
        print('[%d, %d]：%.2e' % (w, v, np.abs(grad_1 - grad_2)))


for toleration in [1e-5, 1e-6]:
    print('%.e' % toleration)
    for w in range(y.shape[0]):
        for v in range(y.shape[1]):
            check_grad(y, labels, w, v, toleration)


def remove_blank(labels, blank=0):
    new_labels = []

    # combine duplicate
    previous = None
    for l in labels:
        if l != previous:
            new_labels.append(l)
            previous = l

    # remove blank
    new_labels = [l for l in new_labels if l != blank]

    return new_labels


def insert_blank(labels, blank=0):
    new_labels = [blank]
    for l in labels:
        new_labels += [l, blank]
    return new_labels


def greedy_decode(y, blank=0):
    raw_rs = np.argmax(y, axis=1)
    rs = remove_blank(raw_rs, blank)
    return raw_rs, rs


# y = softmax(np.random.random([20, 6]))
rr, rs = greedy_decode(y)
print(rr)
print(rs)
