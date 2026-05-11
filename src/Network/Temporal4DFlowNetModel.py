import tensorflow as tf

'''
This is adapted with code partwise copied from Derek Long: https://github.com/dlon450/4DFlowNetv2
'''

class T4DFlowNet():
    def __init__(self, res_increase, high_res_block='resnet_block', low_res_block='resnet_block', upsampling_block='linear', post_processing_block=None):
        self.res_increase = res_increase
        self.high_res_block = high_res_block
        self.low_res_block = low_res_block
        self.upsampling_block = upsampling_block
        self.post_processing_block = post_processing_block

    def build_network(self, u, v, w, u_mag, v_mag, w_mag, low_resblock=8, hi_resblock=4, channel_nr=64, include_mag=True):
        network_blocks = {
            'resnet_block': resnet_block,
            'dense_block':  dense_block,
            'csp_block':    csp_block,
            'unet_block':   u_net_block,
            'lstm_block':   lstm_block,
        }

        upsampling_blocks = {
            'linear':           upsample3d_linear,
            'nearest_neighbor': upsample3d_NN,
            'conv3d_transpose': upsample3d_Conv3DTranspose,
        }

        padding = 'SYMMETRIC'

        if include_mag:
            speed = (u ** 2 + v ** 2 + w ** 2) ** 0.5
            mag   = (u_mag ** 2 + v_mag ** 2 + w_mag ** 2) ** 0.5
            pcmr  = mag * speed

            phase = tf.keras.layers.concatenate([u, v, w])
            pc    = tf.keras.layers.concatenate([pcmr, mag, speed])

            pc    = conv3d(pc, 3, channel_nr, padding, 'relu')
            pc    = conv3d(pc, 3, channel_nr, padding, 'relu')

            phase = conv3d(phase, 3, channel_nr, padding, 'relu')
            phase = conv3d(phase, 3, channel_nr, padding, 'relu')

            concat_layer = tf.keras.layers.concatenate([phase, pc])
        else:
            phase = tf.keras.layers.concatenate([u, v, w])
            concat_layer = conv3d(phase, 3, channel_nr, padding, 'relu')
            concat_layer = conv3d(concat_layer, 3, channel_nr, padding, 'relu')

        concat_layer = conv3d(concat_layer, 1, channel_nr, padding, 'relu')
        concat_layer = conv3d(concat_layer, 3, channel_nr, padding, 'relu')

        # Low res blocks
        rb = concat_layer
        rb = network_blocks[self.low_res_block](rb, low_resblock, channel_nr=channel_nr, pad=padding)

        # Upsampling
        rb = upsampling_blocks[self.upsampling_block](rb, self.res_increase)

        # High res blocks
        rb = network_blocks[self.high_res_block](rb, hi_resblock, channel_nr=channel_nr, pad=padding)

        # Output paths (separate u, v, w)
        if self.post_processing_block is None:
            u_path = conv3d(rb, 3, channel_nr, padding, 'relu')
            u_path = conv3d(u_path, 3, 1, padding, None)

            v_path = conv3d(rb, 3, channel_nr, padding, 'relu')
            v_path = conv3d(v_path, 3, 1, padding, None)

            w_path = conv3d(rb, 3, channel_nr, padding, 'relu')
            w_path = conv3d(w_path, 3, 1, padding, None)
        else:
            u_path = network_blocks[self.post_processing_block](rb, 2, channel_nr=channel_nr, pad=padding)
            u_path = conv3d(u_path, 3, 1, padding, None)

            v_path = network_blocks[self.post_processing_block](rb, 2, channel_nr=channel_nr, pad=padding)
            v_path = conv3d(v_path, 3, 1, padding, None)

            w_path = network_blocks[self.post_processing_block](rb, 2, channel_nr=channel_nr, pad=padding)
            w_path = conv3d(w_path, 3, 1, padding, None)

        return tf.keras.layers.concatenate([u_path, v_path, w_path])


def upsample3d_NN(input_tensor, res_increase):
    """Nearest neighbor upsampling along the temporal dimension."""
    return tf.keras.layers.UpSampling3D(size=(res_increase, 1, 1))(input_tensor)


def upsample3d_Conv3DTranspose(input_tensor, res_increase, padding='same'):
    """Transposed convolution upsampling along the temporal dimension."""
    _, _, _, _, c_size = input_tensor.shape
    return tf.keras.layers.Conv3DTranspose(
        filters=c_size, kernel_size=3, strides=(res_increase, 1, 1), padding=padding
    )(input_tensor)


def upsample3d_linear(input_tensor, res_increase):
    """
        Resize the image by linearly interpolating the input
        using TF resize function.

        :param input_tensor: 5D image tensor, with shape:
            'batch, T, Y, Z, Channels'
        :return: interpolated volume

        Original source: https://niftynet.readthedocs.io/en/dev/_modules/niftynet/layer/linear_resize.html
    """
    if res_increase == 1:
        return input_tensor

    b_size, t_size, y_size, z_size, c_size = input_tensor.shape
    t_size_new = t_size * res_increase

    # Resize Y-Z plane
    squeeze_b_x = tf.reshape(input_tensor, [-1, y_size, z_size, c_size], name='reshape_bx')
    resize_b_x  = tf.image.resize(squeeze_b_x, [y_size, z_size])
    resume_b_x  = tf.reshape(resize_b_x, [-1, t_size, y_size, z_size, c_size], name='resume_bx')

    # Reorient and resize along T
    reoriented  = tf.transpose(resume_b_x, [0, 3, 2, 1, 4])
    squeeze_b_z = tf.reshape(reoriented, [-1, y_size, t_size, c_size], name='reshape_bz')
    resize_b_z  = tf.image.resize(squeeze_b_z, [y_size, t_size_new])
    resume_b_z  = tf.reshape(resize_b_z, [-1, z_size, y_size, t_size_new, c_size], name='resume_bz')

    return tf.transpose(resume_b_z, [0, 3, 2, 1, 4])


def conv3d(x, kernel_size, filters, padding='SYMMETRIC', activation=None, initialization=None, use_bias=True):
    """
        3D convolution with optional symmetric/reflect padding.
        Based on: https://github.com/gitlimlab/CycleGAN-Tensorflow/blob/master/ops.py
    """
    reg_l2 = tf.keras.regularizers.l2(5e-7)

    if padding in ('SYMMETRIC', 'REFLECT'):
        p = (kernel_size - 1) // 2
        x = tf.pad(x, [[0, 0], [p, p], [p, p], [p, p], [0, 0]], padding)
        x = tf.keras.layers.Conv3D(filters, kernel_size, activation=activation,
                                   kernel_initializer=initialization, use_bias=use_bias,
                                   kernel_regularizer=reg_l2)(x)
    else:
        assert padding in ('SAME', 'VALID')
        x = tf.keras.layers.Conv3D(filters, kernel_size, activation=activation,
                                   kernel_initializer=initialization, use_bias=use_bias,
                                   kernel_regularizer=reg_l2)(x)
    return x


def resnet_block(x, num_layers, channel_nr=64, scale=1, pad='SAME'):
    """Residual block with LeakyReLU activations."""
    for _ in range(num_layers):
        tmp = conv3d(x, kernel_size=3, filters=channel_nr, padding=pad, activation=None, use_bias=False)
        tmp = tf.keras.layers.LeakyReLU(alpha=0.2)(tmp)
        tmp = conv3d(tmp, kernel_size=3, filters=channel_nr, padding=pad, activation=None, use_bias=False)
        tmp = x + tmp * scale
        x   = tf.keras.layers.LeakyReLU(alpha=0.2)(tmp)
    return x


def conv_block(x, channel_nr=64, pad='SAME'):
    """Basic conv block with two conv layers and LeakyReLU.
    Copied from Derek Long: https://github.com/dlon450/4DFlowNetv2
    """
    tmp = conv3d(x, kernel_size=3, filters=channel_nr, padding=pad, activation=None, use_bias=False)
    tmp = tf.keras.layers.LeakyReLU(alpha=0.2)(tmp)
    tmp = conv3d(tmp, kernel_size=3, filters=channel_nr, padding=pad, activation=None, use_bias=False)
    tmp = tf.keras.layers.LeakyReLU(alpha=0.2)(tmp)
    return tmp


def dense_block(x, num_layers, channel_nr=64, scale=1, pad='SAME'):
    """Dense block. Copied from Derek Long: https://github.com/dlon450/4DFlowNetv2"""
    k = channel_nr // 4
    for _ in range(int(num_layers)):
        output = conv_block(x, k, 'SYMMETRIC')
        x = tf.concat([x, output], axis=-1)
    return x


def csp_block(x, num_layers, channel_nr=64, scale=1, pad='SAME'):
    """CSP block. Copied from Derek Long: https://github.com/dlon450/4DFlowNetv2"""
    k = channel_nr // 4
    tmp = x[:, :, :, :, :k]
    for _ in range(int(num_layers)):
        output = conv_block(tmp, k, 'SYMMETRIC')
        tmp = tf.concat([tmp, output], axis=-1)
    tmp = tf.concat([x[:, :, :, :, k:], tmp], axis=-1)
    return tmp


def u_net_block(x, num_layers, channel_nr=64, pad='SAME', use_BN=False):
    """U-Net style block with downsampling and upsampling paths."""

    def conv_unet_block(x, num_filters):
        tmp = conv3d(x, kernel_size=3, filters=num_filters, padding=pad, activation=None, use_bias=False)
        if use_BN:
            tmp = tf.keras.layers.BatchNormalization()(tmp)
        tmp = tf.keras.layers.LeakyReLU(alpha=0.2)(tmp)
        tmp = resnet_block(tmp, num_layers=2, channel_nr=num_filters, pad=pad)
        return tmp

    def downsampling_block(x, num_filters):
        tmp = conv_unet_block(x, num_filters)
        p   = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='same')(tmp)
        return tmp, p

    def upsampling_block(x, skip_features, num_filters):
        tmp = tf.keras.layers.Conv3DTranspose(filters=num_filters, kernel_size=3, strides=(2, 2, 2), padding='same')(x)
        tmp = tf.keras.layers.concatenate([tmp, skip_features])
        tmp = conv_unet_block(tmp, num_filters)
        return tmp

    filter_nums = [channel_nr * (2 ** i) for i in range(num_layers + 1)]

    inputs   = conv3d(x, kernel_size=3, filters=channel_nr, padding=pad, activation=None, use_bias=False)
    s1, p1   = downsampling_block(inputs, filter_nums[0])
    s2, p2   = downsampling_block(p1, filter_nums[1])
    b1       = conv_unet_block(p2, filter_nums[2])
    d1       = upsampling_block(b1, s2, filter_nums[1])
    d2       = upsampling_block(d1, s1, filter_nums[0])
    output   = conv_unet_block(d2, filter_nums[0])
    return output


def lstm_block(x, num_layers, channel_nr=64, scale=1, pad='SAME'):
    """2D ConvLSTM block operating over temporal dimension."""
    for _ in range(num_layers):
        x = tf.keras.layers.ConvLSTM2D(channel_nr, 3, use_bias=False, return_sequences=True, padding='same')(x)
    return x