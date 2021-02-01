# -*- coding: utf-8 -*- 
# @Time 2020/6/5 13:20
# @Author wcy
import tensorflow as tf


def l2_regularizer():
    # 模拟 tf.contrib.layers.l2_regularizer(scale, scope=None)
    with tf.Session() as sess:
        weight_decay = 0.1
        tmp = tf.constant([0, 1, 2, 3], dtype=tf.float32)
        """
        l2_reg=tf.contrib.layers.l2_regularizer(weight_decay)
        a=tf.get_variable("I_am_a",regularizer=l2_reg,initializer=tmp)
        """
        # **上面代码的等价代码
        a = tf.get_variable("I_am_a", initializer=tmp)
        a2 = tf.reduce_sum(a * a) * weight_decay / 2
        a3 = tf.get_variable(a.name.split(":")[0] + "/Regularizer/l2_regularizer", initializer=a2)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, a2)
        # **

        sess.run(tf.global_variables_initializer())
        keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        for key in keys:
            print("%s : %s" % (key.name, sess.run(key)))


def all_l2():
    tf.reset_default_graph()
    with tf.Session() as sess:
        weight_decay = 0.1  # (1)定义weight_decay
        l2_reg = tf.contrib.layers.l2_regularizer(weight_decay)  # (2)定义l2_regularizer()
        tmp = tf.constant([0, 1, 2, 3], dtype=tf.float32)
        a = tf.get_variable("I_am_a", regularizer=l2_reg, initializer=tmp)  # (3)创建variable，l2_regularizer复制给regularizer参数。
        # 目测REXXX_LOSSES集合
        # regularizer定义会将a加入REGULARIZATION_LOSSES集合
        print("Global Set:")
        keys = tf.get_collection("variables")
        for key in keys:
            print(key.name)
        print("Regular Set:")
        keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        for key in keys:
            print(key.name)
        print("--------------------")
        sess.run(tf.global_variables_initializer())
        print(sess.run(a))
        reg_set = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES)  # (4)则REGULARIAZTION_LOSSES集合会包含所有被weight_decay后的参数和，将其相加
        l2_loss = tf.add_n(reg_set)
        print("loss=%s" % (sess.run(l2_loss)))
        """
    
        此处输出0.7,即:
    
           weight_decay*sigmal(w*2)/2=0.1*(0*0+1*1+2*2+3*3)/2=0.7
    
        其实代码自己写也很方便，用API看着比较正规。
    
        在网络模型中，直接将l2_loss加入loss就好了。(loss变大，执行train自然会decay)
    
        """


if __name__ == '__main__':
    # 我们很容易可以模拟出tf.contrib.layers.l2_regularizer都做了什么，不过会让代码变丑。
    l2_regularizer()
    # 以下比较完整实现L2 正则化。
    all_l2()
