# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=invalid-name

# Modified from tensorflow's MobilenetV2

import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import VersionAwareLayers
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export

BASE_WEIGHT_PATH = (
    "https://storage.googleapis.com/tensorflow/" "keras-applications/mobilenet_v2/"
)
layers = None


# @keras_export(
#     "keras.applications.mobilenet_v2.MobileNetV2", "keras.applications.MobileNetV2"
# )
def SuperProxylessNAS(
    input_shape=None,
    alpha=1.0,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    arch_kernel=[None] * 22,
    **kwargs
):
    global layers
    if "layers" in kwargs:
        layers = kwargs.pop("layers")
    else:
        layers = VersionAwareLayers()
    if kwargs:
        raise ValueError("Unknown argument(s): %s" % (kwargs,))
    if not (weights in {"imagenet", None} or file_io.file_exists_v2(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `imagenet` "
            "(pre-training on ImageNet), "
            "or the path to the weights file to be loaded."
        )

    if weights == "imagenet" and include_top and classes != 1000:
        raise ValueError(
            'If using `weights` as `"imagenet"` with `include_top` '
            "as true, `classes` should be 1000"
        )

    # Determine proper input shape and default size.
    # If both input_shape and input_tensor are used, they should match
    if input_shape is not None and input_tensor is not None:
        try:
            is_input_t_tensor = backend.is_keras_tensor(input_tensor)
        except ValueError:
            try:
                is_input_t_tensor = backend.is_keras_tensor(
                    layer_utils.get_source_inputs(input_tensor)
                )
            except ValueError:
                raise ValueError(
                    "input_tensor: ", input_tensor, "is not type input_tensor"
                )
        if is_input_t_tensor:
            if backend.image_data_format() == "channels_first":
                if backend.int_shape(input_tensor)[1] != input_shape[1]:
                    raise ValueError(
                        "input_shape: ",
                        input_shape,
                        "and input_tensor: ",
                        input_tensor,
                        "do not meet the same shape requirements",
                    )
            else:
                if backend.int_shape(input_tensor)[2] != input_shape[1]:
                    raise ValueError(
                        "input_shape: ",
                        input_shape,
                        "and input_tensor: ",
                        input_tensor,
                        "do not meet the same shape requirements",
                    )
        else:
            raise ValueError(
                "input_tensor specified: ", input_tensor, "is not a keras tensor"
            )

    # If input_shape is None, infer shape from input_tensor
    if input_shape is None and input_tensor is not None:

        try:
            backend.is_keras_tensor(input_tensor)
        except ValueError:
            raise ValueError(
                "input_tensor: ",
                input_tensor,
                "is type: ",
                type(input_tensor),
                "which is not a valid type",
            )

        if input_shape is None and not backend.is_keras_tensor(input_tensor):
            default_size = 224
        elif input_shape is None and backend.is_keras_tensor(input_tensor):
            if backend.image_data_format() == "channels_first":
                rows = backend.int_shape(input_tensor)[2]
                cols = backend.int_shape(input_tensor)[3]
            else:
                rows = backend.int_shape(input_tensor)[1]
                cols = backend.int_shape(input_tensor)[2]

            if rows == cols and rows in [96, 128, 160, 192, 224]:
                default_size = rows
            else:
                default_size = 224

    # If input_shape is None and no input_tensor
    elif input_shape is None:
        default_size = 224

    # If input_shape is not None, assume default size
    else:
        if backend.image_data_format() == "channels_first":
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            rows = input_shape[0]
            cols = input_shape[1]

        if rows == cols and rows in [96, 128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224

    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=default_size,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights,
    )

    if backend.image_data_format() == "channels_last":
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if weights == "imagenet":
        if alpha not in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4]:
            raise ValueError(
                "If imagenet weights are being loaded, "
                "alpha can be one of `0.35`, `0.50`, `0.75`, "
                "`1.0`, `1.3` or `1.4` only."
            )

        if rows != cols or rows not in [96, 128, 160, 192, 224]:
            rows = 224
            logging.warning(
                "`input_shape` is undefined or non-square, "
                "or `rows` is not in [96, 128, 160, 192, 224]."
                " Weights for input shape (224, 224) will be"
                " loaded as the default."
            )

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1

    first_block_filters = _make_divisible(32 * alpha, 8)
    x = layers.Conv2D(
        first_block_filters,
        kernel_size=3,
        strides=(2, 2),
        padding="same",
        name="Conv1",
    )(img_input)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3, momentum=0.99, name="bn_Conv1"
    )(x)
    x = layers.ReLU(6.0, name="Conv1_relu")(x)

    x = _inverted_res_block(
        x,
        filters=16,
        alpha=alpha,
        stride=1,
        block_id=0,
        opcode=arch_kernel[0]
    )

    x = _inverted_res_block(
        x,
        filters=32,
        alpha=alpha,
        stride=2,
        block_id=1,
        opcode=arch_kernel[1]
    )
    x = _inverted_res_block(
        x,
        filters=32,
        alpha=alpha,
        stride=2,
        block_id=2,
        opcode=arch_kernel[2]
    )
    x = _inverted_res_block(
        x,
        filters=32,
        alpha=alpha,
        stride=2,
        block_id=3,
        opcode=arch_kernel[3]
    )
    x = _inverted_res_block(
        x,
        filters=32,
        alpha=alpha,
        stride=2,
        block_id=4,
        opcode=arch_kernel[4]
    )

    x = _inverted_res_block(
        x,
        filters=40,
        alpha=alpha,
        stride=2,
        block_id=5,
        opcode=arch_kernel[5]
    )
    x = _inverted_res_block(
        x,
        filters=40,
        alpha=alpha,
        stride=2,
        block_id=6,
        opcode=arch_kernel[6]
    )
    x = _inverted_res_block(
        x,
        filters=40,
        alpha=alpha,
        stride=2,
        block_id=7,
        opcode=arch_kernel[7]
    )
    x = _inverted_res_block(
        x,
        filters=40,
        alpha=alpha,
        stride=2,
        block_id=8,
        opcode=arch_kernel[8]
    )

    x = _inverted_res_block(
        x,
        filters=80,
        alpha=alpha,
        stride=2,
        block_id=9,
        opcode=arch_kernel[9]
    )
    x = _inverted_res_block(
        x,
        filters=80,
        alpha=alpha,
        stride=2,
        block_id=10,
        opcode=arch_kernel[10]
    )
    x = _inverted_res_block(
        x,
        filters=80,
        alpha=alpha,
        stride=2,
        block_id=11,
        opcode=arch_kernel[11]
    )
    x = _inverted_res_block(
        x,
        filters=80,
        alpha=alpha,
        stride=2,
        block_id=12,
        opcode=arch_kernel[12]
    )

    x = _inverted_res_block(
        x,
        filters=96,
        alpha=alpha,
        stride=1,
        block_id=13,
        opcode=arch_kernel[13]
    )
    x = _inverted_res_block(
        x,
        filters=96,
        alpha=alpha,
        stride=1,
        block_id=14,
        opcode=arch_kernel[14]
    )
    x = _inverted_res_block(
        x,
        filters=96,
        alpha=alpha,
        stride=1,
        block_id=15,
        opcode=arch_kernel[15]
    )
    x = _inverted_res_block(
        x,
        filters=96,
        alpha=alpha,
        stride=1,
        block_id=16,
        opcode=arch_kernel[16]
    )

    x = _inverted_res_block(
        x,
        filters=192,
        alpha=alpha,
        stride=2,
        block_id=17,
        opcode=arch_kernel[17]
    )
    x = _inverted_res_block(
        x,
        filters=192,
        alpha=alpha,
        stride=2,
        block_id=18,
        opcode=arch_kernel[18]
    )
    x = _inverted_res_block(
        x,
        filters=192,
        alpha=alpha,
        stride=2,
        block_id=19,
        opcode=arch_kernel[19]
    )
    x = _inverted_res_block(
        x,
        filters=192,
        alpha=alpha,
        stride=2,
        block_id=20,
        opcode=arch_kernel[20]
    )

    x = _inverted_res_block(
        x,
        filters=320,
        alpha=alpha,
        stride=1,
        block_id=21,
        opcode=arch_kernel[21]
    )

    # no alpha applied to last conv as stated in the paper:
    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    x = layers.Conv2D(last_block_filters, kernel_size=1, name="Conv_1")(
        x
    )
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3, momentum=0.99, name="Conv_1_bn"
    )(x)
    x = layers.ReLU(6.0, name="out_relu")(x)

    if include_top:
        x = layers.GlobalAveragePooling2D()(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Dense(classes, activation=classifier_activation, 
                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
                         name="predictions")(
            x
        )

    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = training.Model(inputs, x, name="mobilenetv2_%0.2f_%s" % (alpha, rows))

    # Load weights.
    if weights == "imagenet":
        if include_top:
            model_name = (
                "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_"
                + str(float(alpha))
                + "_"
                + str(rows)
                + ".h5"
            )
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = data_utils.get_file(
                model_name, weight_path, cache_subdir="models"
            )
        else:
            model_name = (
                "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_"
                + str(float(alpha))
                + "_"
                + str(rows)
                + "_no_top"
                + ".h5"
            )
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = data_utils.get_file(
                model_name, weight_path, cache_subdir="models"
            )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def _inverted_res_block(inputs, stride, alpha, filters, block_id, opcode):
    """Inverted ResNet block."""
    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1

    # Customization
    arch = [
        {"kernel_size": 3, "expansion": 3},
        {"kernel_size": 5, "expansion": 3},
        {"kernel_size": 7, "expansion": 3},
        {"kernel_size": 3, "expansion": 6},
        {"kernel_size": 5, "expansion": 6},
        {"kernel_size": 7, "expansion": 6},
        {"kernel_size": -1, "expansion": -1}
    ]
    kernel_size = arch[opcode]['kernel_size']
    expansion = arch[opcode]['expansion']

    if block_id == 0:
        expansion = 1

    in_channels = backend.int_shape(inputs)[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = "block_{}_".format(block_id)

    # Zero Layer
    if opcode == 6:
        return x

    if block_id:
        # Expand
        x = layers.Conv2D(
            expansion * in_channels,
            kernel_size=1,
            padding="same",
            activation=None,
            name=prefix + "expand",
            kernel_regularizer=tf.keras.regularizers.L2(l2=4e-5),
            kernel_initializer=tf.keras.initializers.HeUniform()
        )(x)
        x = layers.BatchNormalization(
            axis=channel_axis, epsilon=1e-3, momentum=0.99, name=prefix + "expand_BN"
        )(x)
        x = layers.ReLU(6.0, name=prefix + "expand_relu")(x)
    else:
        prefix = "expanded_conv_"

    # Depthwise
    if stride == 2:
        x = layers.ZeroPadding2D(
            padding=imagenet_utils.correct_pad(x, kernel_size), name=prefix + "pad"
        )(x)
    x = layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=stride,
        activation=None,
        padding="same" if stride == 1 else "valid",
        name=prefix + "depthwise",
        depthwise_regularizer=tf.keras.regularizers.L2(l2=4e-5),
        depthwise_initializer=tf.keras.initializers.HeUniform()
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3, momentum=0.99, name=prefix + "depthwise_BN"
    )(x)

    x = layers.ReLU(6.0, name=prefix + "depthwise_relu")(x)

    # Project
    x = layers.Conv2D(
        pointwise_filters,
        kernel_size=1,
        padding="same",
        activation=None,
        name=prefix + "project",
        kernel_regularizer=tf.keras.regularizers.L2(l2=4e-5),
        kernel_initializer=tf.keras.initializers.HeUniform()
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3, momentum=0.99, name=prefix + "project_BN"
    )(x)

    if in_channels == pointwise_filters and stride == 1:
        return layers.Add(name=prefix + "add")([inputs, x])
    return x


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


@keras_export("keras.applications.mobilenet_v2.preprocess_input")
def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(x, data_format=data_format, mode="tf")


@keras_export("keras.applications.mobilenet_v2.decode_predictions")
def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)


preprocess_input.__doc__ = imagenet_utils.PREPROCESS_INPUT_DOC.format(
    mode="",
    ret=imagenet_utils.PREPROCESS_INPUT_RET_DOC_TF,
    error=imagenet_utils.PREPROCESS_INPUT_ERROR_DOC,
)
decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__
