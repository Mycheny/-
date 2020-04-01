import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import BatchNormalization

if __name__ == '__main__':
    a = np.reshape(np.random.random(28*28)*10-5, (1, 28, 28, 1))
    x = tf.compat.v1.convert_to_tensor(a, tf.float32)
    # x = tf.compat.v1.nn.batch_normalization(x, axis=-1)
    ReLU6 = tf.keras.layers.ReLU(max_value=6, name="ReLU6")
    res = ReLU6(a)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.initialize_all_variables())
        constant_ops = [op for op in sess.graph.get_operations()]
        for constant_op in constant_ops:
            print(constant_op.name)
        y = sess.run([x, res])

        print(a.mean(axis=-1), a.std(axis=-1))

