from config import args, get_log_dir
import numpy as np
from data import Mnist, Svhn
import tensorflow as tf
import tensorbayes as tb
from tensorbayes.layers import placeholder

################
# Set up model #
################
T = tb.utils.TensorDict(dict(
    sess = tf.Session(),
    trg_x = placeholder((None, 32, 32, 1), name='trg_x'),
    trg_d = placeholder((None, 32, 32, 1), name='trg_d'),
    src_x = placeholder((None, 32, 32, 3), name='src_x'),
    src_y = placeholder((None, 10), name='src_y'),
    test_x = placeholder((None, 32, 32, 3), name='test_x'),
    test_y = placeholder((None, 10), name='test_y'),
    phase = placeholder((), tf.bool, name='phase')
))

exec "from {0:s} import {0:s}".format(args.model)
exec "T = {:s}(T)".format(args.model)
T.sess.run(tf.global_variables_initializer())

if args.model != 'classifier':
    path = tf.train.latest_checkpoint('save')
    restorer = tf.train.Saver(tf.get_collection('trainable_variables', 'enc'))
    restorer.restore(T.sess, path)

#############
# Load data #
#############
mnist = Mnist(size=32)
svhn = Svhn(size=32)

#########
# Train #
#########
bs = 100
iterep = 600
n_epoch = 5000 if args.model != 'classifier' else 17
epoch = 0
feed_dict = {T.phase: 1}
saver = tf.train.Saver()

print "Batch size:", bs
print "Iterep:", iterep
print "Total iterations:", n_epoch * iterep
print "Log directory:", get_log_dir()

for i in xrange(n_epoch * iterep):
    #################
    # Discriminator #
    #################
    src_x, src_y = svhn.train.next_batch(100)
    trg_x, _ = mnist.train.next_batch(100)
    trg_d, _ = mnist.train.next_batch(100)
    feed_dict.update({T.trg_x: trg_x, T.trg_d: trg_d,
                      T.src_x: src_x, T.src_y: src_y})

    if len(T.ops_disc) > 0:
        summary, _ = T.sess.run(T.ops_disc, feed_dict)
        T.train_writer.add_summary(summary, i + 1)

    ###################
    # Train generator #
    ###################
    src_x, src_y = svhn.train.next_batch(100)
    trg_x, _ = mnist.train.next_batch(100)
    feed_dict.update({T.trg_x: trg_x,
                      T.src_x: src_x, T.src_y: src_y})

    summary, _ = T.sess.run(T.ops_main, feed_dict)
    T.train_writer.add_summary(summary, i + 1)
    T.train_writer.flush()

    end_epoch, epoch = tb.utils.progbar(i, iterep,
                                        message='epoch: {:d}'.format(epoch),
                                        display=args.run >= 999)

    if end_epoch:
        if len(T.ops_epoch) > 0:
            test_x, test_y = svhn.test.next_batch(1000)
            feed_dict.update({T.test_x: test_x, T.test_y: test_y})
            summary, = T.sess.run(T.ops_epoch, feed_dict)
            T.train_writer.add_summary(summary, i + 1)

        if len(T.ops_print) > 0:
            print T.sess.run(T.ops_print, feed_dict)

        if args.model == 'classifier':
            path = saver.save(T.sess, 'save/{:s}'.format(args.model), global_step=i+1)
            print "Saving to {:s}".format(path)

    if end_epoch and epoch % 10 == 0:
        if len(T.ops_image) > 0:
            summary, = T.sess.run(T.ops_image, feed_dict)
            T.train_writer.add_summary(summary, i + 1)
