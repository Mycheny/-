import matplotlib.pylab as pyl
import numpy as np


def operation():
    pass


if __name__ == '__main__':
    X_ = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    Y_ = np.array([1, 2.3, 2.7, 4.3, 4.7, 6.3, 7, 8, 9])

    # X_ = np.array([1])
    # Y_ = np.array([1])

    n = len(X_)

    for i in range(3):
        x_ = X_[i*3:(i+1)*3]
        y_ = Y_[i*3:(i+1)*3]
        w = (np.sum(x_) * np.sum(y_) - n * np.sum(x_ * y_)) / (np.square(np.sum(x_)) - n * np.sum(np.square(x_)))
        b = (np.sum(y_) - w * np.sum(x_))/n
        print(w, b)
        y = w * x_ + b

        pyl.plot(x_, y)
        pyl.plot(x_, y_, 'o')
        pyl.show()