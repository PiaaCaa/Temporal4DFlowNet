import tensorflow as tf

'''
This is adapted with code partwise copied from Derek Long: https://github.com/dlon450/4DFlowNetv2
'''

class STR4DFlowNet():
    def __init__(self, res_increase, high_res_block='resnet_block',low_res_block='resnet_block', upsampling_block = 'default', post_processing_block = None):
        self.res_increase = res_increase
        self.high_res_block = high_res_block
        self.low_res_block = low_res_block
        self.upsampling_block = upsampling_block
        self.post_processing_block= post_processing_block

    def build_network(self, u, v, w, u_mag, v_mag, w_mag, low_resblock=8, hi_resblock=4, channel_nr=64, include_mag=True):
        network_blocks = {
            'resnet_block': resnet_block,
            'dense_block': dense_block,
            'csp_block': csp_block, 
            'unet_block': u_net_block, 
            'lstm_block':lstm_block,
        }

        upsampling_blocks = {
            'linear': upsample3d_linear,
            'nearest_neigbor': upsample3d_NN,
            'Conv3DTranspose': upsample3d_Conv3DTranspose,  
        }

        channel_nr = 64
        padding = 'SYMMETRIC'

        if include_mag:
            speed = (u ** 2 + v ** 2 + w ** 2) ** 0.5
            mag = (u_mag ** 2 + v_mag ** 2 + w_mag ** 2) ** 0.5
            pcmr = mag * speed

            phase = tf.keras.layers.concatenate([u,v,w])
            pc    = tf.keras.layers.concatenate([pcmr, mag, speed])
            
            
            pc = conv3d(pc,3,channel_nr, padding , 'relu')
            pc = conv3d(pc,3,channel_nr, padding, 'relu')

            phase = conv3d(phase,3,channel_nr, padding, 'relu')
            phase = conv3d(phase,3,channel_nr, padding, 'relu')

            concat_layer = tf.keras.layers.concatenate([phase, pc])
            
        else:
            # only phase
            phase = tf.keras.layers.concatenate([u,v,w])

            concat_layer = conv3d(phase,3,channel_nr, padding, 'relu')
            concat_layer = conv3d(concat_layer,3,channel_nr, padding, 'relu')
        
        concat_layer = conv3d(concat_layer, 1, channel_nr, padding, 'relu')
        concat_layer = conv3d(concat_layer, 3, channel_nr, padding, 'relu')
        
        # initial low res blocks
        rb = concat_layer
        rb = network_blocks[self.low_res_block](rb, low_resblock, channel_nr=channel_nr, pad=padding)

        rb = upsampling_blocks[self.upsampling_block](rb, self.res_increase)
        
        rb = network_blocks[self.high_res_block](rb, hi_resblock, channel_nr=channel_nr, pad=padding)

        # 3 separate path version
        if self.post_processing_block is None:
            u_path = conv3d(rb, 3, channel_nr, padding, 'relu')
            u_path = conv3d(u_path, 3, 1, padding, None)

            v_path = conv3d(rb, 3, channel_nr, padding, 'relu')
            v_path = conv3d(v_path, 3, 1, padding, None)

            w_path = conv3d(rb, 3, channel_nr, padding, 'relu')
            w_path = conv3d(w_path, 3, 1, padding, None)
        else:
            #assume that it will be only one post processing block, i.e. # blocks = 1
            u_path = network_blocks[self.post_processing_block](rb, 2, channel_nr=channel_nr, pad = padding )
            u_path = conv3d(u_path, 3, 1, padding, None)

            v_path = network_blocks[self.post_processing_block](rb, 2, channel_nr=channel_nr, pad = padding )
            v_path = conv3d(v_path, 3, 1, padding, None)

            w_path = network_blocks[self.post_processing_block](rb, 2, channel_nr=channel_nr, pad = padding )
            w_path = conv3d(w_path, 3, 1, padding, None)
        

        b_out = tf.keras.layers.concatenate([u_path, v_path, w_path])
        return b_out


def upsample3d_NN(input_tensor, res_increase):
    
    output_tensor = tf.keras.layers.UpSampling3D(size=(res_increase, 1, 1))(input_tensor)
    return output_tensor

def upsample3d_Conv3DTranspose(input_tensor, res_increase):
    b_size, t_size, y_size, z_size, c_size = input_tensor.shape
    ##TODO change this to real padding, same of valida padding
    output_tensor = tf.keras.layers.Conv3DTranspose(filters = c_size,kernel_size = 3, strides = (res_increase, 1, 1) ,padding = 'same')(input_tensor) 
    return output_tensor

def upsample3d_linear(input_tensor, res_increase):
    """
        Resize the image by linearly interpolating the input
        using TF '``'resize_bilinear' function.

        :param input_tensor: 2D/3D image tensor, with shape:
            'batch, T, Y, Z, Channels'
        :return: interpolated volume

        Original source: https://niftynet.readthedocs.io/en/dev/_modules/niftynet/layer/linear_resize.html
    """
    
    b_size, t_size, y_size, z_size, c_size = input_tensor.shape

    t_size_new, y_size_new, z_size_new = t_size * res_increase, y_size , z_size

    if res_increase == 1:
        # already in the target shape
        return input_tensor

    # resize y-z
    squeeze_b_x = tf.reshape(input_tensor, [-1, y_size, z_size, c_size], name='reshape_bx')
    #resize_b_x = tf.compat.v1.image.resize_bilinear(squeeze_b_x, [y_size_new, z_size_new], align_corners=align)
    resize_b_x = tf.image.resize(squeeze_b_x, [y_size_new, z_size_new])#, method=ResizeMethod.BILINEAR)
    resume_b_x = tf.reshape(resize_b_x, [-1, t_size, y_size_new, z_size_new, c_size], name='resume_bx')

    #Reorient
    reoriented = tf.transpose(resume_b_x, [0, 3, 2, 1, 4])
    
    #   squeeze and 2d resize
    #TODO: check, since it works only if reoriented has same input shape as input tensor
    squeeze_b_z = tf.reshape(reoriented, [-1, y_size_new, t_size, c_size], name='reshape_bz')
    #resize_b_z = tf.compat.v1.image.resize_bilinear(squeeze_b_z, [y_size_new, x_size_new], align_corners=align)
    resize_b_z = tf.image.resize(squeeze_b_z, [y_size_new, t_size_new])#, method=ResizeMethod.BILINEAR)
    resume_b_z = tf.reshape(resize_b_z, [-1, z_size_new, y_size_new, t_size_new, c_size], name='resume_bz')
    
    output_tensor = tf.transpose(resume_b_z, [0, 3, 2, 1, 4])
    return output_tensor



def conv3d(x, kernel_size, filters, padding='SYMMETRIC', activation=None, initialization=None, use_bias=True):
    """
        Based on: https://github.com/gitlimlab/CycleGAN-Tensorflow/blob/master/ops.py
        For tf padding, refer to: https://www.tensorflow.org/api_docs/python/tf/pad

    """
    reg_l2 = tf.keras.regularizers.l2(5e-7)

    if padding == 'SYMMETRIC' or padding == 'REFLECT':
        p = (kernel_size - 1) // 2
        x = tf.pad(x, [[0,0],[p,p],[p,p], [p,p],[0,0]], padding)
        x = tf.keras.layers.Conv3D(filters, kernel_size, activation=activation, kernel_initializer=initialization, use_bias=use_bias, kernel_regularizer=reg_l2)(x)
    else:
        assert padding in ['SAME', 'VALID']
        x = tf.keras.layers.Conv3D(filters, kernel_size, activation=activation, kernel_initializer=initialization, use_bias=use_bias, kernel_regularizer=reg_l2)(x)
    return x


def resnet_block(x, num_layers, block_name='ResBlock', channel_nr=64, scale = 1, pad='SAME'):
    for _ in range(num_layers):
        tmp = conv3d(x, kernel_size=3, filters=channel_nr, padding=pad, activation=None, use_bias=False, initialization=None)
        tmp = tf.keras.layers.LeakyReLU(alpha=0.2)(tmp)

        tmp = conv3d(tmp, kernel_size=3, filters=channel_nr, padding=pad, activation=None, use_bias=False, initialization=None)

        tmp = x + tmp * scale
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(tmp)
    return x

def u_net_block(x, num_layers, block_name = 'UnetBlock', channel_nr = 64, pad = 'SAME', use_BN = False):
    
    def conv_unet_block(x, num_filters, use_BN):
        '''
        Convolution block
        '''
        # print('num filters:', num_filters,'shape', x.shape)
        tmp = conv3d(x, kernel_size=3, filters=num_filters, padding=pad, activation=None, use_bias=False, initialization=None)
        if use_BN:
            tmp = tf.keras.layers.BatchNormalization()(tmp)
        tmp = tf.keras.layers.LeakyReLU(alpha=0.2)(tmp)
        tmp = resnet_block(tmp, num_layers=2, channel_nr=num_filters, pad = pad)
        # tmp = conv3d(x, kernel_size=3, filters=num_filters, padding=pad, activation=None, use_bias=False, initialization=None)
        # if use_BN:
        #     tmp = tf.keras.layers.BatchNormalization()(tmp)
        # tmp = tf.keras.layers.LeakyReLU(alpha=0.2)(tmp)

        # tmp = conv3d(tmp, kernel_size=3, filters=num_filters, padding=pad, activation=None, use_bias=False, initialization=None)
        # if use_BN:
        #     tmp = tf.keras.layers.BatchNormalization()(tmp)
        # tmp = tf.keras.layers.LeakyReLU(alpha=0.2)(tmp)
        return tmp
        
    def upsampling_block(x, skip_features, num_filters, use_BN):
        tmp = tf.keras.layers.Conv3DTranspose(filters = num_filters,kernel_size = 3, strides = (2, 2, 2),padding = 'same')(x) 
        print('after convT:', tmp.shape, skip_features.shape)
        tmp = tf.keras.layers.concatenate([tmp, skip_features]) #axis ?? #tf.keras.layers.Concatenate()[tmp, skip_features]
        tmp = conv_unet_block(tmp, num_filters, use_BN)
        return tmp
        
    def downsampling_block(x, num_filters, use_BN):
        tmp = conv_unet_block(x, num_filters, use_BN)
        p = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding = 'same')(tmp)
        return tmp, p

    filter_nums  = [channel_nr*(2**i) for i in range(0, num_layers+1)]
    # print('filter nums:', filter_nums)
    inputs = conv3d(x, kernel_size=3, filters=channel_nr, padding=pad, activation=None, use_bias=False, initialization=None)

    #this is a trial for two downsampling blocks
    s1, p1 = downsampling_block(inputs, filter_nums[0] , use_BN)
    print("shape output block 1, not downsampled:",s1.shape, " downsampled:", p1.shape )
    s2, p2 = downsampling_block(p1, filter_nums[1], use_BN)
    print("shape output block 2, not downsampled:",s2.shape, " downsampled:", p2.shape )

    b1 = conv_unet_block(p2, filter_nums[2], use_BN)
    print("shape output block smallest:",b1.shape )

    d1 = upsampling_block(b1, s2, filter_nums[1], use_BN)
    print("shape output block 3,upsample block ",d1.shape)
    d2 = upsampling_block(d1, s1, filter_nums[0], use_BN)
    print("shape output block 4,upsample block ",d2.shape)
    
    output = conv_unet_block(d2, filter_nums[0], use_BN)
    return output



#copied from derek
def conv_block(x, block_name='ConvBlock', channel_nr=64, pad='SAME'):
    tmp = conv3d(x, kernel_size=3, filters=channel_nr, padding=pad, activation=None, use_bias=False, initialization=None)
    tmp = tf.keras.layers.LeakyReLU(alpha=0.2)(tmp)
    tmp = conv3d(tmp, kernel_size=3, filters=channel_nr, padding=pad, activation=None, use_bias=False, initialization=None)
    tmp = tf.keras.layers.LeakyReLU(alpha=0.2)(tmp)

    return tmp

#copied from derek
def dense_block(x, num_layers, block_name='DenseBlock', channel_nr=64, scale = 1, pad='SAME'):
    k = channel_nr//4
    for _ in range(int(num_layers)):
        output = conv_block(x, 'ConvBlock', k, 'SYMMETRIC')
        x = tf.concat([x, output], axis=-1)
    
    return x

#copied from derek
def csp_block(x, num_layers, block_name='CSPBlock', channel_nr=64, scale = 1, pad='SAME'):
    k = channel_nr//4
    tmp = x[:,:,:,:,:k]
    for _ in range(int(num_layers)):
        output = conv_block(tmp, 'ConvBlock', k, 'SYMMETRIC')
        tmp = tf.concat([tmp, output], axis=-1)
    tmp = tf.concat([x[:,:,:,:,k:], tmp], axis=-1)
    return tmp

# very simple conv lstm block which takes sequences of 2D images and returns sequences of images again
def lstm_block(x, num_layers, block_name='LSTMBlock', channel_nr=64, scale = 1, pad='SAME'):
    for _ in range(num_layers):
        x = tf.keras.layers.ConvLSTM2D(channel_nr, 3, use_bias = False, return_sequences = True, padding = 'same')(x)

    return x
