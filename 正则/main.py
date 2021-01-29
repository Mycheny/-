# -*- coding: utf-8 -*- 
# @File main.py
# @Time 2021/1/29 10:50
# @Author wcy
# @Software: PyCharm
# @Site
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
tf.random.set_random_seed(1)
np.random.seed(1)


if __name__ == '__main__':
    k = 0.2
    x = np.linspace(-5, 5, 10, dtype=np.float64)
    y_ = x * k + np.random.random((x.shape[0],))*2

    x = np.concatenate((x, np.array([-2])))
    y_ = np.concatenate((y_, np.array([4])))
    # plt.scatter(x, y_, alpha=0.6)
    # plt.show()

    w = tf.Variable(initial_value=1, dtype=tf.float32)
    b = tf.Variable(initial_value=0, dtype=tf.float32)
    y = w * x + b
    loss = tf.reduce_mean(tf.sqrt(tf.square(y - y_)))
    total_loss1 = loss + tf.reduce_sum(tf.abs(w))
    total_loss2 = loss + tf.reduce_sum(tf.sqrt(tf.square(w)))

    op0 = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
    op1 = tf.train.AdamOptimizer(learning_rate=0.01).minimize(total_loss1)
    op2 = tf.train.AdamOptimizer(learning_rate=0.01).minimize(total_loss2)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in range(100):
            # sess.run(op0)
            sess.run(op1)
            # sess.run(op2)
            w_predict, y_predict, t_loss0, t_loss1, t_loss2 = sess.run([w, y, loss, total_loss1, total_loss2])

        plt.scatter(x, y_, alpha=0.2)
        plt.plot(x, y_predict, alpha=0.6)
        plt.show()

