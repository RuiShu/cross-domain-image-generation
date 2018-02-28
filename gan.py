from config import args, get_log_dir, delete_existing
import tensorflow as tf
import tensorbayes as tb
from tensorbayes.layers import placeholder
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

def gan(T):
    #####################
    # Target Autoencode #
    #####################
    trg_z = encoder(T.trg_x, T.phase)
    trg_rx = generator(trg_z, T.phase)

    # Loss
    if args.mse:
        loss_trg_ae = tf.reduce_mean(reduce_sum_sq(trg_rx - T.trg_x, axis=-1))
        loss_trg_mul = loss_trg_ae / (32 * 32)
    else:
        loss_trg_ae = tf.reduce_mean(tf.square(trg_rx - T.trg_x))
        loss_trg_mul = loss_trg_ae

    ################
    # Cross-Domain #
    ################
    src_z = encoder(T.src_x, T.phase, reuse=True)
    src_rx = generator(src_z, T.phase, reuse=True)
    src_rz = encoder(src_rx, T.phase, reuse=True)

    # Loss
    if args.mse:
        loss_src_ae = tf.reduce_mean(reduce_sum_sq(src_rz - src_z, axis=-1))
        loss_src_mul = loss_src_ae / 128
    else:
        loss_src_ae = tf.reduce_mean(tf.square(src_rz - src_z))
        loss_src_mul = loss_src_ae

    #############
    # Adversary #
    #############
    src_fake = discriminator(src_rx, T.phase, depth=3)
    trg_fake = discriminator(trg_rx, T.phase, reuse=True, depth=3)
    trg_real = discriminator(T.trg_d, T.phase, reuse=True, depth=3)

    # Loss
    softmax_xent = tf.nn.softmax_cross_entropy_with_logits
    bs = tf.unstack(tf.shape(T.trg_x))[0]
    hot0 = tf.one_hot(tf.fill([bs], 0), 3)
    hot1 = tf.one_hot(tf.fill([bs], 1), 3)
    hot2 = tf.one_hot(tf.fill([bs], 2), 3)

    loss_disc = tf.reduce_mean(
        softmax_xent(labels=hot0, logits=trg_real) +
        softmax_xent(labels=hot1, logits=trg_fake) +
        softmax_xent(labels=hot2, logits=src_fake))

    loss_gen = tf.reduce_mean(
        softmax_xent(labels=hot0, logits=trg_fake) +
        softmax_xent(labels=hot0, logits=src_fake))

    #############
    # Optimizer #
    #############
    loss_main = (loss_gen +
                 args.trg_w * loss_trg_ae +
                 args.src_w * loss_src_ae)

    var_disc = tf.get_collection('trainable_variables', 'disc/')
    var_gen = tf.get_collection('trainable_variables', 'gen/')
    var_enc = [] # tf.get_collection('trainable_variables', 'enc/')
    var_main = var_gen + var_enc

    T.train_main = tf.train.AdamOptimizer(args.glr).minimize(loss_main, var_list=var_main)
    T.train_disc = tf.train.AdamOptimizer(args.dlr).minimize(loss_disc, var_list=var_disc)

    ##############
    # Summarizer #
    ##############
    tf.summary.scalar('main/loss_gen', loss_gen / 2)
    tf.summary.scalar('main/loss_trg_mul', loss_trg_mul)
    tf.summary.scalar('main/loss_src_mul', loss_src_mul)

    tf.summary.scalar('disc/loss_disc', loss_disc / 3)

    tf.summary.image('image_trg/gen', reshape_img(trg_rx))
    tf.summary.image('image_scr/gen', reshape_img(src_rx))
    tf.summary.image('image_trg_real/real', reshape_img(T.trg_x))
    tf.summary.image('image_src_real/real', reshape_img(T.src_x))

    summary_main = tf.summary.merge(tf.get_collection('summaries', 'main'))
    summary_disc = tf.summary.merge(tf.get_collection('summaries', 'disc'))
    summary_image = tf.summary.merge(tf.get_collection('summaries', 'image'))

    #######
    # Ops #
    #######
    T.ops_print = [loss_disc / 3, loss_gen / 2, loss_trg_mul, loss_src_mul]
    T.ops_disc = [summary_disc, T.train_disc]
    T.ops_main = [summary_main, T.train_main]
    T.ops_epoch = []
    T.ops_image = [summary_image]
    delete_existing(get_log_dir())
    T.train_writer = tf.summary.FileWriter(get_log_dir())

    return T
