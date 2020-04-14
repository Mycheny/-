# -*- coding: utf-8 -*- 
# @Time 2020/4/14 16:16
# @Author wcy
import tensorflow as tf
from tensorflow.python.tools import saved_model_utils

if __name__ == '__main__':
    with tf.Graph().as_default() as g1:
        base64_str = tf.placeholder(tf.string, name='input_string')
        input_str = tf.decode_base64(base64_str)
        decoded_image = tf.image.decode_png(input_str, channels=1)
        # Convert from full range of uint8 to range [0,1] of float32.
        decoded_image_as_float = tf.image.convert_image_dtype(decoded_image,
                                                              tf.float32)
        decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
        resize_shape = tf.stack([28, 28])
        resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
        resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                                 resize_shape_as_int)
        # 展开为1维数组
        resized_image_1d = tf.reshape(resized_image, (-1, 28 * 28))
        print(resized_image_1d.shape)
        tf.identity(resized_image_1d, name="DecodeJPGOutput")

    g1def = g1.as_graph_def()

    with tf.Graph().as_default() as g2:
        with tf.Session(graph=g2) as sess:
            input_graph_def = saved_model_utils.get_meta_graph_def(
                "./model", tag_constants.SERVING).graph_def

            tf.saved_model.loader.load(sess, ["serve"], "./model")

            g2def = graph_util.convert_variables_to_constants(
                sess,
                input_graph_def,
                ["myOutput"],
                variable_names_whitelist=None,
                variable_names_blacklist=None)
