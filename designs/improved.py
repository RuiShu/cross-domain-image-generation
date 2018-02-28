from config import args
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorbayes.layers import dense, conv2d, conv2d_transpose, upsample
from extra_layers import leaky_relu, wndense, noise, wnconv2d, wnconv2d_transpose

dropout = tf.layers.dropout

def encoder(x, phase, reuse=None):
    with tf.variable_scope('enc', reuse=reuse):
        with arg_scope([dense], bn=True, phase=phase, activation=leaky_relu), \
             arg_scope([conv2d], bn=True, phase=phase, activation=leaky_relu):

            if x._shape_as_list()[-1] == 1:
                x = tf.image.grayscale_to_rgb(x)

            x = conv2d(x, 64, 3, 2)
            x = conv2d(x, 128, 3, 2)
            x = conv2d(x, 256, 3, 2)
            x = dense(x, 128)

    return x

def generator(x, phase, reuse=None):
    with tf.variable_scope('gen', reuse=reuse):
        with arg_scope([dense], bn=True, phase=phase, activation=leaky_relu), \
             arg_scope([conv2d_transpose], bn=True, phase=phase, activation=leaky_relu):

            x = dense(x, 4 * 4 * 512)
            x = tf.reshape(x, [-1, 4, 4, 512])
            x = conv2d_transpose(x, 256, 5, 2)
            x = conv2d_transpose(x, 128, 5, 2)
            x = wnconv2d_transpose(x, 1, 5, 2, bn=False, activation=tf.nn.tanh, scale=True)

    return x

def discriminator(x, phase, reuse=None, depth=1):
    with tf.variable_scope('disc', reuse=reuse):
        with arg_scope([wnconv2d, wndense], activation=leaky_relu), \
             arg_scope([noise], phase=phase):

            x = dropout(x, rate=0.2, training=phase)
            x = wnconv2d(x, 64, 3, 2)

            x = dropout(x, training=phase)
            x = wnconv2d(x, 128, 3, 2)

            x = dropout(x, training=phase)
            x = wnconv2d(x, 256, 3, 2)

            x = dropout(x, training=phase)
            x = wndense(x, 1024)

            x = dense(x, depth, activation=None, bn=False)
    return x
