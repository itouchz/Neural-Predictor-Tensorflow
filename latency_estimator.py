# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.
# Modified by Patara Trirat

import yaml
import os
import sys
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

class LatencyEstimator(object):
    def __init__(self):
        with open('proxylessnas/pixel_trim.yaml') as f:
            self.lut = yaml.load(f, Loader=yaml.FullLoader)

    @staticmethod
    def repr_shape(shape):
        if isinstance(shape, (list, tuple)):
            return 'x'.join(str(_) for _ in shape)
        elif isinstance(shape, str):
            return shape
        else:
            return TypeError

    def predict(self, ltype: str, _input, output, expand=None, kernel=None, stride=None, idskip=None, ):
        """
        :param ltype:
            Layer type must be one of the followings
                1. `Conv`: The initial 3x3 conv with stride 2.
                2. `Conv_1`: The upsample 1x1 conv that increases num_filters by 4 times.
                3. `Logits`: All operations after `Conv_1`.
                4. `expanded_conv`: MobileInvertedResidual
        :param _input: input shape (h, w, #channels)
        :param output: output shape (h, w, #channels)
        :param expand: expansion ratio
        :param kernel: kernel size
        :param stride:
        :param idskip: indicate whether has the residual connection
        """
        infos = [ltype, 'input:%s' % self.repr_shape(_input), 'output:%s' % self.repr_shape(output), ]

        if ltype in ('expanded_conv',):
            assert None not in (expand, kernel, stride, idskip)
            infos += ['expand:%d' % expand, 'kernel:%d' % kernel, 'stride:%d' % stride, 'idskip:%d' % idskip]
        key = '-'.join(infos)
        return self.lut[key]['mean']