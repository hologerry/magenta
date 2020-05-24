import tensorflow as tf
import numpy as np

from tensor2tensor.layers import common_layers

dim = 10
hdim = 1024
batch_size = 2
seq_len = 51
# initial_output = tf.zeros((batch_size, 0, 1, hdim), dtype=tf.float32)
# zero_pad = tf.zeros((batch_size, 1, 1, dim), dtype=tf.float32)
# # Hack: foldl complains when the output shape is less specified than the
# # input shape, so we confuse it about the input shape.
# i_s = common_layers.shape_list(initial_output)
# print(i_s)
# initial_output = tf.slice(initial_output, [0, 0, 0, 0], i_s)
# z_s = common_layers.shape_list(zero_pad)
# print(z_s)
# zero_pad = tf.slice(zero_pad, [0, 0, 0, 0], z_s)

# print(initial_output.get_shape())
# print(zero_pad.get_shape())
with tf.Session() as sess:
    # a = tf.random.normal((6, 1), dtype=tf.float32) * 5
    # # print(a.eval(session=sess))
    # a = tf.abs(a) % 4
    # a = tf.cast(a, tf.int32)
    # a = tf.reshape(a, [-1])
    # o = tf.stack([tf.range(tf.size(a)), a], axis=-1)
    # print(o.get_shape())
    # out_mean = tf.reshape(tf.range(24, dtype=tf.float32), (6, 4))
    # # out_std = tf.random.normal((batch_size*seq_len*6, 50), dtype=tf.float32)
    # print("o size", o.get_shape())
    # print(o.eval(session=sess))
    # print("out_mean", out_mean.get_shape())
    # print(out_mean.eval(session=sess))
    # choose_mean = tf.gather_nd(out_mean, o)

    # print("chosen", choose_mean.get_shape())
    # print(choose_mean.eval(session=sess))
    # data = np.reshape(np.arange(30), [5, 6])
    # x = tf.constant(data)
    # print(x.eval(session=sess))
    # idx = tf.constant([[1, 2], [4, 3], [2, 5], [3, 3], [4, 5]])
    # result = tf.gather_nd(x, idx)
    # print(result.eval(session=sess))

    # target = tf.random.normal((batch_size, seq_len, 10), dtype=tf.float32)
    # command = target[..., :4]
    # arg = target[..., 4:]
    # arg = tf.reshape(arg, [-1, 3 * 2])
    # masktemplate = tf.constant([[0., 0., 0., 0., 0., 0.],
    #                             [0., 0., 0., 0., 1., 1.],
    #                             [0., 0., 0., 0., 1., 1.],
    #                             [1., 1., 1., 1., 1., 1.]])
    # print("command", command.get_shape())
    # print("template", masktemplate.get_shape())
    # mask = tf.tensordot(command, masktemplate, [[-1], [-2]])
    # print("mask", mask.get_shape())
    # args_flat = tf.reshape(arg, [-1, 1])
    # print(arg.get_shape())
    target = tf.random.normal((batch_size, 1, 64, 64), dtype=tf.float32)
    output = tf.random.normal((batch_size, 1, 64, 64), dtype=tf.float32)
    weight = common_layers.weights_all(target)
    print(weight.eval())
    loss_num = tf.pow(output - target, 2)
    l1o = tf.reduce_sum(loss_num * weight)
    l2o = tf.reduce_sum(weight)
    print(l1o.eval())
    print(l2o.eval())
