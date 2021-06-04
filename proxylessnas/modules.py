# MIT License

# Copyright (c) 2021 Han Cai

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Modified from mit-han-lab/proxylessnas repository
# https://github.com/mit-han-lab/proxylessnas

from proxylessnas.layers import *
import tensorflow as tf

class MobileInvertedResidualBlock:

    def __init__(self, _id, mobile_inverted_conv, has_residual):
        self.id = _id
        self.mobile_inverted_conv = mobile_inverted_conv
        self.has_residual = has_residual

    def build(self, _input, net, init=None):
        output = _input
        with tf.compat.v1.variable_scope(self.id):
            output = self.mobile_inverted_conv.build(output, net, init)
            if self.has_residual:
                output = output + _input
        return output


class ProxylessNASNets:

    def __init__(self, net_config, net_weights=None):
        self.graph = tf.Graph()

        self.net_config = net_config
        self.n_classes = 1000

        with self.graph.as_default():
            self._define_inputs()
            logits = self.build(init=net_weights)

            prediction = logits
            # losses
            cross_entropy = tf.reduce_mean(
                input_tensor=tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits, labels=tf.stop_gradient(self.labels)))
            self.cross_entropy = cross_entropy

            correct_prediction = tf.equal(
                tf.argmax(input=prediction, axis=1),
                tf.argmax(input=self.labels, axis=1)
            )
            self.accuracy = tf.reduce_mean(
                input_tensor=tf.cast(correct_prediction, tf.float32))

            self.global_variables_initializer = tf.compat.v1.global_variables_initializer()
        self._initialize_session()

    @property
    def bn_eps(self):
        return self.net_config['bn']['eps']

    @property
    def bn_decay(self):
        return 1 - self.net_config['bn']['momentum']

    def _initialize_session(self):
        """ Initialize session, variables """
        config = tf.compat.v1.ConfigProto()  # allow_soft_placement=True, log_device_placement=False
        # restrict model GPU memory utilization to min required
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(graph=self.graph, config=config)
        self.sess.run(self.global_variables_initializer)

    def _define_inputs(self):
        shape = [None, 224, 224, 3]
        self.images = tf.compat.v1.placeholder(
            tf.float32,
            shape=shape,
            name='input_images')
        self.labels = tf.compat.v1.placeholder(
            tf.float32,
            shape=[None, self.n_classes],
            name='labels')
        self.learning_rate = tf.compat.v1.placeholder(
            tf.float32,
            shape=[],
            name='learning_rate')
        self.is_training = tf.compat.v1.placeholder(
            tf.bool, shape=[], name='is_training')

    @staticmethod
    def labels_to_one_hot(n_classes, labels):
        new_labels = np.zeros((labels.shape[0], n_classes), dtype=np.float32)
        new_labels[range(labels.shape[0]), labels] = np.ones(labels.shape)
        return new_labels

    def build(self, init=None):
        output = self.images
        if init is not None:
            for key in init:
                init[key] = tf.compat.v1.constant_initializer(init[key])

        # first conv
        first_conv = ConvLayer(
            'first_conv',
            self.net_config['first_conv']['out_channels'],
            3,
            2)
        output = first_conv.build(output, self, init)

        for i, block_config in enumerate(self.net_config['blocks']):
            if block_config['mobile_inverted_conv']['name'] == 'ZeroLayer':
                continue
            mobile_inverted_conv = MBInvertedConvLayer(
                'mobile_inverted_conv',
                block_config['mobile_inverted_conv']['out_channels'],
                block_config['mobile_inverted_conv']['kernel_size'],
                block_config['mobile_inverted_conv']['stride'],
                block_config['mobile_inverted_conv']['expand_ratio'],
            )
            if block_config['shortcut'] is None or block_config['shortcut']['name'] == 'ZeroLayer':
                has_residual = False
            else:
                has_residual = True
            block = MobileInvertedResidualBlock(
                'blocks/%d' %
                i, mobile_inverted_conv, has_residual)
            output = block.build(output, self, init)

        # feature mix layer
        feature_mix_layer = ConvLayer(
            'feature_mix_layer',
            self.net_config['feature_mix_layer']['out_channels'],
            1,
            1)
        output = feature_mix_layer.build(output, self, init)

        output = avg_pool(output, 7, 7)
        output = flatten(output)
        classifier = LinearLayer(
            'classifier',
            self.n_classes,
            self.net_config['classifier']['dropout_rate'])
        output = classifier.build(output, self, init)
        return output
