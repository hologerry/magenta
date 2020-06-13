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
    # target = tf.random.normal((batch_size, 1, 64, 64), dtype=tf.float32)
    # output = tf.random.normal((batch_size, 1, 64, 64), dtype=tf.float32)
    # weight = common_layers.weights_all(target)
    # print(weight.eval())
    # loss_num = tf.pow(output - target, 2)
    # l1o = tf.reduce_sum(loss_num * weight)
    # l2o = tf.reduce_sum(weight)
    # print(l1o.eval())
    # print(l2o.eval())

    # predcit = tf.random.uniform((2, 5, 4), minval=0, maxval=1, dtype=tf.float32)
    # target = tf.random.uniform((2, 5, 4), minval=0, maxval=2, dtype=tf.int32)
    # predict = tf.constant([[[0.370561,   0.7522373,  0.962854,   0.78139925],
    #                         [0.35318124, 0.55829847, 0.21254015, 0.97762775],
    #                         [0.7895678,  0.47163403, 0.20040119, 0.9747666 ],
    #                         [0.0747689,  0.85168886, 0.74427426, 0.7529695 ],
    #                         [0.04231274, 0.12493503, 0.5645406,  0.15124023]],
    #                         [[0.22381783, 0.40880632, 0.740329,   0.00489283],
    #                         [0.33797348, 0.64474237, 0.9960165,  0.33216643],
    #                         [0.9271157,  0.5983815,  0.8915962,  0.91998374],
    #                         [0.8761481,  0.9494686,  0.0476774,  0.58984673],
    #                         [0.5242083,  0.30677414, 0.13274765, 0.8846445 ]]], dtype=tf.float32)
    # target = tf.constant([[[0., 0., 0., 0.],
    #                         [0., 0., 0., 0.],
    #                         [0., 0., 0., 1.],
    #                         [1., 0., 0., 0.],
    #                         [1., 0., 0., 0.]],
    #                         [[0., 0., 0., 0.],
    #                         [0., 0., 1., 0.],
    #                         [0., 0., 1., 0.],
    #                         [0., 0., 1., 0.],
    #                         [0., 0., 0., 1.]]], dtype=tf.float32)
    # target = tf.cast(target, tf.float32)
    # print(predict.eval())
    # print(target.eval())
    # softmax_xent_loss = tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=predict)
    # print(softmax_xent_loss.get_shape())
    # softmax_xent_loss = tf.reduce_mean(softmax_xent_loss)
    # print(softmax_xent_loss.eval())
    clss = tf.random.uniform((4, ), minval=0, maxval=10, dtype=tf.int64)
    print(clss.eval())
    W1 = tf.Variable(tf.random.uniform([10, 8], -1.0, 1.0), name="W")
    embedded_clss = tf.nn.embedding_lookup(W1, clss)
    print(embedded_clss.get_shape())
    embedded_clsses = tf.expand_dims(embedded_clss, -1)
    print(embedded_clsses.get_shape())
