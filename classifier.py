from config import args, get_log_dir, delete_existing
import tensorflow as tf
import tensorbayes as tb
from tensorbayes.layers import placeholder, dense
import os
from tensorbayes.tfutils import reduce_l2_loss, reduce_sum_sq

# Dynamic import and assignment of gen/disc
exec "from designs import {:s}".format(args.design)
exec "encoder = {:s}.encoder".format(args.design)
exec "generator = {:s}.generator".format(args.design)
exec "discriminator = {:s}.discriminator".format(args.design)

def reshape_img(x):
    C = x._shape_as_list()[-1]
    img = tf.reshape(x, [10, 10, 32, 32, C])
    img = tf.reshape(tf.transpose(img, [0, 2, 1, 3, 4]), [1, 320, 320, C])
    img = (img + 1) / 2
    img = tf.clip_by_value(img, 0, 1)
    return img

def accuracy(a, b):
    a = tf.argmax(a, 1)
    b = tf.argmax(b, 1)
    eq = tf.cast(tf.equal(a, b), 'float32')
    return tf.reduce_mean(eq)

def classify(x, phase, reuse=None):
    z = encoder(x, phase, reuse=reuse)
    with tf.variable_scope('enc/final', reuse=reuse):
        y = dense(z, 10, activation=None, bn=False)
    return y

def classifier(T):
    ##################
    # Classification #
    ##################
    src_y = classify(T.src_x, T.phase)
    test_y = classify(T.test_x, T.phase, reuse=True)

    # Loss
    softmax_xent = tf.nn.softmax_cross_entropy_with_logits
    loss_main = tf.reduce_mean(softmax_xent(labels=T.src_y, logits=src_y))

    train_acc = accuracy(T.src_y, src_y)
    test_acc = accuracy(T.test_y, test_y)

    #############
    # Optimizer #
    #############
    var_main = tf.get_collection('trainable_variables', 'enc/')
    T.train_main = tf.train.AdamOptimizer(args.glr).minimize(loss_main, var_list=var_main)

    ##############
    # Summarizer #
    ##############
    tf.summary.scalar('train/loss_main', loss_main)
    tf.summary.scalar('train/acc', train_acc)
    tf.summary.scalar('test/acc', test_acc)
    summary_main = tf.summary.merge(tf.get_collection('summaries', 'train'))
    summary_epoch = tf.summary.merge(tf.get_collection('summaries', 'test'))

    #######
    # Ops #
    #######
    T.ops_print = [loss_main, train_acc, test_acc]
    T.ops_disc = []
    T.ops_main = [summary_main, T.train_main]
    T.ops_epoch = [summary_epoch]
    T.ops_image = []

    delete_existing(get_log_dir())
    T.train_writer = tf.summary.FileWriter(get_log_dir())

    return T
