from config import args
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorbayes.layers import dense, conv2d, conv2d_transpose, upsample
from extra_layers import leaky_relu, noise

dropout = tf.layers.dropout
lrelu = lambda x: leaky_relu(x, a=0.2)

def encoder(x, phase, reuse=None):
    with tf.variable_scope('enc', reuse=reuse):
        with arg_scope([dense], bn=True, phase=phase, activation=leaky_relu), \
             arg_scope([conv2d], bn=True, phase=phase, activation=leaky_relu):

            if x._shape_as_list()[-1] == 1:
                x = tf.image.grayscale_to_rgb(x)

            if args.dense:
                # Level 0: 32
                l0 = x

                # Level 1: 32 -> 16, 8, 4
                a1 = conv2d(l0, 64, 3, 2)
                a2 = conv2d(l0, 64, 5, 4)
                a3 = conv2d(l0, 64, 9, 8)
                l1 = a1

                # Level 2: 16 -> 8, 4
                b2 = conv2d(l1, 64, 3, 2)
                b3 = conv2d(l1, 64, 5, 4)
                l2 = tf.concat([a2, b2], -1)

                # Level 3: 8 -> 4
                c3 = conv2d(l2, 64, 3, 2)
                l3 = tf.concat([a3, b3, c3], -1)

                # Level 4: Dense
                x = dense(l3, 128)

            else:
                x = conv2d(x, 64, 3, 2)
                x = conv2d(x, 128, 3, 2)
                x = conv2d(x, 256, 3, 2)
                x = dense(x, 128)

    return x

def generator(x, phase, reuse=None):
    with tf.variable_scope('gen', reuse=reuse):
        with arg_scope([dense], bn=True, phase=phase, activation=tf.nn.relu), \
             arg_scope([conv2d_transpose], bn=True, phase=phase, activation=tf.nn.relu):

            if args.dense:
                raise NotImplementedError('yep')
                # Level 0: 1
                l0 = tf.reshape(x, [-1, 1, 1, 128])

                # Level 1: 1 -> 4, 8, 16
                a1 = conv2d_transpose(l0, 64, 1, 1)
                a2 = conv2d_transpose(l0, 64, 1, 1)
                a3 = conv2d_transpose(l0, 64, 1, 1)
                l1 = a1

                # Level 2: 4 -> 8, 16
                b2 = conv2d_transpose(l1, 64, )

                # Level 2: 8 -> 16, 32
                b2 = conv2d_transpose(l1, 64, 3, 2)
                l2 = tf.concat([a2, b2], -1)

                # Level 3: 16 -> 32
                c3 = conv2d_transpose(l2, 64, 3, 2)
                l3 = tf.concat([a3, b3, c3], -1)

            else:
                x = dense(x, 4 * 4 * 512)
                x = tf.reshape(x, [-1, 4, 4, 512])
                x = conv2d_transpose(x, 256, 5, 2)
                x = conv2d_transpose(x, 128, 5, 2)
                x = conv2d_transpose(x, 1, 5, 2, bn=False, activation=tf.nn.tanh)

    return x

def discriminator(x, phase, reuse=None, depth=1):
    with tf.variable_scope('disc', reuse=reuse):
        with arg_scope([conv2d, dense], bn=True, phase=phase, activation=lrelu), \
             arg_scope([noise], phase=phase):

            x = dropout(x, rate=0.2, training=phase)
            x = conv2d(x, 64, 3, 2, bn=False)
            x = dropout(x, training=phase)
            x = conv2d(x, 128, 3, 2)
            x = dropout(x, training=phase)
            x = conv2d(x, 256, 3, 2)
            x = dropout(x, training=phase)
            x = dense(x, 1024)
            x = dense(x, depth, activation=None, bn=False)
    return x

"""
Should I switch LRELU to 0.2? Yes.
Should I set LR to 3e-4 for both? Yes.
"""
