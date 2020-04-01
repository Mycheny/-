# -*- coding: utf-8 -*- 
# @Time 2020/3/16 10:44
# @Author wcy

import numpy as np
import tensorflow as tf

y = np.array([[1, 0],
              [0, 1],
              [0, 1],
              [1, 0],
              [1, 0]])
y = tf.convert_to_tensor(y)
predict = np.array([[1, 0],
                    [1, 0],
                    [1, 0],
                    [1, 0],
                    [0.0, 1]])
predict = tf.convert_to_tensor(predict)

confusion_matrix = tf.confusion_matrix(tf.argmax(predict, 1), tf.argmax(y, 1), num_classes=2)
TP = confusion_matrix[0, 0]
FN = confusion_matrix[0, 1]
FP = confusion_matrix[1, 0]
TN = confusion_matrix[1, 1]

with tf.Session() as sess:  # 开始一个会话
    matrix, tp, fn, fp, tn  = sess.run([confusion_matrix, TP, FN , FP, TN])
    print(matrix)
# 输出[[5 0] [0 0]]
# 如果不加num_classes=2就会输出[[5]]
