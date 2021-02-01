# -*- coding: utf-8 -*- 
# @Time 2020/4/21 14:34
# @Author wcy

import tensorflow as tf

def control_dependencies():
    """
    参数：操作或者张量（tensor）的列表
    返回：返回一个上下文管理器，控制上下文中所有操作的控制依赖
    函数的功能：上下文中的所有操作（注意：必须是操作operation）会在参数list中所有操作完成以后才开始执行
    """

    # 本例中，with结构（上下文）内的变量y不是operation，所有它不会在参数中的操作执行之前进行，相反，z是一个操作

    x = tf.Variable(1.0)
    y = tf.Variable(0.0)
    #返回一个op，表示给变量x加1的操作
    x_plus_1 = tf.assign_add(x, 1)

    #control_dependencies的意义是，在执行with包含的内容（在这里就是 y = x）前，
    #先执行control_dependencies参数中的内容（在这里就是 x_plus_1）
    with tf.control_dependencies([x_plus_1]):
        y = x
        # z = tf.identity(x)  # tf.identity()就是返回参数本身，这里相当于z=x

    init = tf.initialize_all_variables()

    with tf.Session() as session:
        init.run()
        for i in range(5):
            # 相当于sess.run(y)，按照我们的预期，由于control_dependencies的作用，
            # 所以应该执行print前都会先执行x_plus_1，但是这种情况会出问题
            print(y.eval())
            # print(z.eval())


if __name__ == '__main__':
    control_dependencies()