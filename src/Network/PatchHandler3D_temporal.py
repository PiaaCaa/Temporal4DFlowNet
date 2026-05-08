import tensorflow as tf
import numpy as np
import h5py
import os
import pandas as pd


def get_augmentation_title(flip_1, flip_2, rot_angle, sign_u, sign_v, sign_w, swap_u, swap_v, swap_w):
    title = ''
    if flip_1 == 1:
        title += 'flip_1, '
    if flip_2 == 1:
        title += 'flip_2, '
    if rot_angle != 0:
        title += f'rot_angle={rot_angle}, '
    if sign_u == -1:
        title += 'sign_u=-1, '
    if sign_v == -1:
        title += 'sign_v=-1, '
    if sign_w == -1:
        title += 'sign_w=-1, '
    if swap_u != 'u':
        title += f'swap_u={swap_u}, '
    if swap_v != 'v':
        title += f'swap_v={swap_v}, '
    if swap_w != 'w':
        title += f'swap_w={swap_w}, '
    return title

class PatchHandler4D():
    # constructor
    def __init__(self, data_dir, patch_size, res_increase, batch_size, mask_threshold=0.6):
        self.patch_size = patch_size
        self.res_increase = res_increase
        self.batch_size = batch_size
        self.mask_threshold = mask_threshold

        self.data_directory = data_dir
        self.hr_colnames = ['u','v','w']
        self.lr_colnames = ['u','v','w']
        self.venc_colnames = ['venc_u','venc_v','venc_w']
        self.mag_colnames  = ['mag_u','mag_v','mag_w']
        self.mask_colname  = 'mask'

    def initialize_dataset(self, indexes, shuffle, n_parallel=None):
        '''
            Input pipeline.
            This function accepts a list of filenames with index and patch locations to read.
        '''
        ds = tf.data.Dataset.from_tensor_slices((indexes))
        print("Total dataset:", len(indexes), 'shuffle', shuffle)

        if shuffle:
            # Set a buffer equal to dataset size to ensure randomness
            ds = ds.shuffle(buffer_size=len(indexes)) 

        ds = ds.map(self.load_data_using_patch_index, num_parallel_calls=n_parallel)
        ds = ds.batch(batch_size=self.batch_size)
        
        # prefetch, n=number of items
        ds = ds.prefetch(self.batch_size)
        
        return ds
    
    def load_data_using_patch_index(self, indexes):
        return tf.py_function(func=self.load_patches_from_index_file, 
            # U-LR, HR, MAG, V-LR, HR, MAG, W-LR, HR, MAG, venc, MASK
            inp=[indexes], 
                Tout=[tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.float32])

    def load_patches_from_index_file(self, indexes):
        # Do typecasting, we need to make sure everything has the correct data type
        # Solution for tf2: https://stackoverflow.com/questions/56122670/how-to-get-string-value-out-of-tf-tensor-which-dtype-is-string
        lr_hd5path = '{}/{}'.format(self.data_directory, bytes.decode(indexes[0].numpy()))
        hd5path    = '{}/{}'.format(self.data_directory, bytes.decode(indexes[1].numpy()))
        
        #read out attributes from csvline
        idx = int(indexes[2])
        x_start, y_start, z_start = int(indexes[3]), int(indexes[4]), int(indexes[5])
        is_rotate = int(indexes[6])
        rotation_plane = int(indexes[7])
        rotation_degree_idx = int(indexes[8])

        patch_size = self.patch_size
        hr_patch_size = self.patch_size * self.res_increase
        
        # ============ get the patch ============ 
        #TODO  format is (t, h, w, d) such that [x_start, idx, y_start, z_start]
        # NO INCREASE IN Y AND Z
        patch_t_index  = np.index_exp[x_start:x_start+patch_size,idx, y_start:y_start+patch_size, z_start:z_start+patch_size]
        hr_t_patch_index = np.index_exp[x_start*self.res_increase :x_start*self.res_increase +hr_patch_size,idx, y_start:y_start+patch_size, z_start:z_start+patch_size]
        mask_t_index = np.index_exp[x_start*self.res_increase :x_start*self.res_increase +hr_patch_size,idx  ,y_start:y_start+patch_size, z_start:z_start+patch_size ]
        # mask_index = np.index_exp[0, x_start*self.res_increase :x_start*self.res_increase +hr_patch_size ,y_start*self.res_increase :y_start*self.res_increase +hr_patch_size , z_start*self.res_increase :z_start*self.res_increase +hr_patch_size ]
        u_patch, u_hr_patch, mag_u_patch, v_patch, v_hr_patch, mag_v_patch, w_patch, w_hr_patch, mag_w_patch, venc, mask_patch = self.load_vectorfield(hd5path, lr_hd5path, idx, mask_t_index, patch_t_index, hr_t_patch_index)
        
   
        # Expand dims (for InputLayer)
        return u_patch[...,tf.newaxis], v_patch[...,tf.newaxis], w_patch[...,tf.newaxis], \
                    mag_u_patch[...,tf.newaxis], mag_v_patch[...,tf.newaxis], mag_w_patch[...,tf.newaxis], \
                    u_hr_patch[...,tf.newaxis], v_hr_patch[...,tf.newaxis], w_hr_patch[...,tf.newaxis], \
                    venc, mask_patch
                    
    
    def create_temporal_mask(self, mask, n_frames):
        '''
        from static mask create temporal mask of shape (n_frames, h, w, d)
        '''
        assert(len(mask.shape) == 3), " shape: " + str(mask.shape) # shape of mask is assumed to be 3 dimensional
        return np.repeat(np.expand_dims(mask, 0), n_frames, axis=0)

    def load_vectorfield(self, hd5path, lr_hd5path, idx, mask_index, patch_index, hr_patch_index):
        '''
            Load LowRes velocity and magnitude components, and HiRes velocity components
            Also returns the global venc and HiRes mask
        '''
        hires_images = []
        lowres_images = []
        mag_images = []
        vencs = []
        global_venc = 0

        # Load the U, V, W component of HR, LR, and MAG
        with h5py.File(hd5path, 'r') as hl:
            # Open the file once per row, Loop through all the HR column
            for i in range(len(self.hr_colnames)):
                w_hr = hl.get(self.hr_colnames[i])[hr_patch_index]
                # add them to the list
                hires_images.append(w_hr)

            # We only have 1 mask for all the objects in 1 file
            try:
                mask = hl.get(self.mask_colname)[mask_index] # Mask value [0 .. 1]
            except:
                #TODO this probably has to be chnaged if structure changes
                mask_temp = self.create_temporal_mask(np.asarray(hl.get(self.mask_colname)).squeeze(),  hl.get(self.hr_colnames[i]).shape[0])
                mask = mask_temp[mask_index] 
            mask = (mask >= self.mask_threshold) * 1.
            print("mask shape:", hl.get(self.mask_colname).shape)
            
        with h5py.File(lr_hd5path, 'r') as hl:
            for i in range(len(self.lr_colnames)):
                w = hl.get(self.lr_colnames[i])[patch_index]
                mag_w = hl.get(self.mag_colnames[i])[patch_index]
                #TODO change venc
                w_venc = hl.get(self.venc_colnames[i])[:]

                # add them to the list
                lowres_images.append(w)
                mag_images.append(mag_w)
                vencs.append(w_venc)
        
        global_venc = np.max(vencs)

        # Convert to numpy array
        hires_images = np.asarray(hires_images)
        lowres_images = np.asarray(lowres_images)
        mag_images = np.asarray(mag_images)
        
        # Normalize the values 
        hires_images = self._normalize(hires_images, global_venc) # Velocity normalized to -1 .. 1
        lowres_images = self._normalize(lowres_images, global_venc)
        mag_images = mag_images / 4095. # Magnitude 0 .. 1

        # U-LR, HR, MAG, V-LR, HR, MAG, w-LR, HR, MAG, venc, MASK
        return lowres_images[0].astype('float32'), hires_images[0].astype('float32'), mag_images[0].astype('float32'), \
            lowres_images[1].astype('float32'), hires_images[1].astype('float32'), mag_images[1].astype('float32'), \
            lowres_images[2].astype('float32'), hires_images[2].astype('float32'), mag_images[2].astype('float32'),\
            global_venc.astype('float32'), mask.astype('float32')
    
    def _normalize(self, u, venc):
        return u / venc


class PatchHandler4D_all_axis():
    # constructor
    def __init__(self, data_dir, patch_size, res_increase, batch_size, mask_threshold=0.6):
        self.patch_size = patch_size
        self.res_increase = res_increase
        self.batch_size = batch_size
        self.mask_threshold = mask_threshold

        self.data_directory = data_dir
        self.hr_colnames = ['u','v','w']
        self.lr_colnames = ['u','v','w']
        self.venc_colnames =  ['u_max', 'v_max', 'w_max']# ['venc_u','venc_v','venc_w'] #
        self.mag_colnames  = ['mag_u','mag_v','mag_w']
        self.mask_colname  = 'mask'

    def initialize_dataset(self, indexes, shuffle, n_parallel=None):
        '''
            Input pipeline.
            This function accepts a list of filenames with index and patch locations to read.
        '''
        ds = tf.data.Dataset.from_tensor_slices((indexes))
        print("Total dataset:", len(indexes), 'shuffle', shuffle)

        if shuffle:
            # Set a buffer equal to dataset size to ensure randomness
            ds = ds.shuffle(buffer_size=len(indexes)) 

        # ds = ds.map(self.load_data_using_patch_index, num_parallel_calls=n_parallel)
        ds = ds.map(self.load_data_using_patch_index, num_parallel_calls=tf.data.AUTOTUNE) #chnages 13/06/25
        ds = ds.batch(batch_size=self.batch_size)
        
        # prefetch, n=number of items
        # ds = ds.prefetch(self.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        
        return ds
    
    # def load_data_using_patch_index(self, indexes):
    #     return tf.py_function(func=self.load_patches_from_index_file, 
    #         # U-LR, HR, MAG, V-LR, HR, MAG, W-LR, HR, MAG, venc, MASK
    #         inp=[indexes], 
    #             Tout=[tf.float32, tf.float32, tf.float32,
    #                 tf.float32, tf.float32, tf.float32,
    #                 tf.float32, tf.float32, tf.float32,
    #                 tf.float32, tf.float32])

    def load_data_using_patch_index(self, indexes):
        out = tf.py_function(func=self.load_patches_from_index_file, inp=[indexes],
                            Tout=[tf.float32]*11)
        t, x, y = self.patch_size  # assuming patch_size_tuple = (t, x, y)
        
        # For velocity and magnitude patches: shape (t, x, y, 1)
        for tensor in out[:-2]:
            tensor.set_shape([t, x, y, 1])
        
        # venc is scalar
        out[-2].set_shape([])
        
        # mask shape (t, x, y) without channel dimension
        out[-1].set_shape([t, x, y])
        
        return tuple(out)



    def load_patches_from_index_file(self, indexes):
        # Do typecasting, we need to make sure everything has the correct data type
        # Solution for tf2: https://stackoverflow.com/questions/56122670/how-to-get-string-value-out-of-tf-tensor-which-dtype-is-string
        lr_hd5path = '{}/{}'.format(self.data_directory, bytes.decode(indexes[0].numpy()))
        hd5path    = '{}/{}'.format(self.data_directory, bytes.decode(indexes[1].numpy()))
        
        #read out attributes from csvline
        axis= int(indexes[2])
        idx = int(indexes[3])
        start_t, start_1, start_2 = int(indexes[4]), int(indexes[5]), int(indexes[6])
        step_t = int(indexes[7])
        reverse = int(indexes[8]) # 1 for no reverse, -1 for reverse order, only reverse te first spatial component
       
        patch_size = self.patch_size
        
        # if step is 1, the loaded LR data is already downsampled
        if step_t == 1:
            start_t_lr = start_t
            hr_patch_size = int(patch_size*self.res_increase)
            lr_patch_size = patch_size
            start_t_hr = int(start_t*self.res_increase)
        else:
            start_t_lr = start_t
            hr_patch_size = self.patch_size * step_t    
            lr_patch_size = hr_patch_size
            start_t_hr = start_t


        # ============ get the patch ============ 
        if axis == 0 :
            patch_t_index       = np.index_exp[start_t_lr :start_t_lr+lr_patch_size:step_t,   idx, start_1:start_1+patch_size, start_2:start_2+patch_size]
            hr_t_patch_index    = np.index_exp[start_t_hr :start_t_hr+hr_patch_size,          idx, start_1:start_1+patch_size, start_2:start_2+patch_size]
            mask_t_index        = np.index_exp[start_t_hr :start_t_hr+hr_patch_size,          idx, start_1:start_1+patch_size, start_2:start_2+patch_size]
        elif axis == 1:
            patch_t_index       = np.index_exp[start_t_lr :start_t_lr+lr_patch_size:step_t, start_1:start_1+patch_size,idx, start_2:start_2+patch_size]
            hr_t_patch_index    = np.index_exp[start_t_hr :start_t_hr+hr_patch_size,        start_1:start_1+patch_size,idx, start_2:start_2+patch_size]
            mask_t_index        = np.index_exp[start_t_hr :start_t_hr+hr_patch_size,        start_1:start_1+patch_size,idx, start_2:start_2+patch_size]
        elif axis == 2:
            patch_t_index       = np.index_exp[start_t_lr :start_t_lr+lr_patch_size:step_t, start_1:start_1+patch_size, start_2:start_2+patch_size, idx]
            hr_t_patch_index    = np.index_exp[start_t_hr :start_t_hr+hr_patch_size,        start_1:start_1+patch_size, start_2:start_2+patch_size, idx]
            mask_t_index        = np.index_exp[start_t_hr :start_t_hr+hr_patch_size,        start_1:start_1+patch_size, start_2:start_2+patch_size, idx]

        
        u_patch, u_hr_patch, mag_u_patch, v_patch, v_hr_patch, mag_v_patch, w_patch, w_hr_patch, mag_w_patch, venc, mask_patch = self.load_vectorfield(hd5path, lr_hd5path, idx, mask_t_index, patch_t_index, hr_t_patch_index, reverse)
        
        # Expand dims (for InputLayer)
        return u_patch[...,tf.newaxis], v_patch[...,tf.newaxis], w_patch[...,tf.newaxis], \
                    mag_u_patch[...,tf.newaxis], mag_v_patch[...,tf.newaxis], mag_w_patch[...,tf.newaxis], \
                    u_hr_patch[...,tf.newaxis], v_hr_patch[...,tf.newaxis], w_hr_patch[...,tf.newaxis], \
                    venc, mask_patch
                    

    def create_temporal_mask(self, mask, n_frames):
        '''
        from static mask create temporal mask of shape (n_frames, h, w, d)
        '''
        assert(len(mask.shape) == 3), " shape: " + str(mask.shape) # shape of mask is assumed to be 3 dimensional
        return np.repeat(np.expand_dims(mask, 0), n_frames, axis=0)

    def load_vectorfield(self, hd5path, lr_hd5path, idx, mask_index, patch_index, hr_patch_index, reverse):
        '''
            Load LowRes velocity and magnitude components, and HiRes velocity components
            Also returns the global venc and HiRes mask
        '''
        
        hires_images = []
        lowres_images = []
        mag_images = []
        vencs = []
        global_venc = 0

        # Load the U, V, W component of HR, LR, and MAG
        with h5py.File(hd5path, 'r') as hl:
            # Open the file once per row, Loop through all the HR column
            for i in range(len(self.hr_colnames)):
                w_hr = hl.get(self.hr_colnames[i])[hr_patch_index]
                # add them to the list
                hires_images.append(w_hr)

            # We only have 1 mask for all the objects in 1 file
            try:
                mask = hl.get(self.mask_colname)[mask_index] # Mask value [0 .. 1]
            except:
                print('Temporal static mask created')
                mask_temp = self.create_temporal_mask(np.asarray(hl.get(self.mask_colname)).squeeze(),  hl.get(self.hr_colnames[i]).shape[0])
                mask = mask_temp[mask_index] 
            mask = (mask >= self.mask_threshold) * 1.
            # print("mask shape:", hl.get(self.mask_colname).shape)
        
        with h5py.File(lr_hd5path, 'r') as hl:
            
            for i in range(len(self.lr_colnames)):
                w = hl.get(self.lr_colnames[i])[patch_index]
                mag_w = hl.get(self.mag_colnames[i])[patch_index]
                w_venc = np.array(hl.get(self.venc_colnames[i])).squeeze()
                            
                # add them to the list
                lowres_images.append(w)
                mag_images.append(mag_w)
                vencs.append(w_venc)
        
        global_venc = np.max(vencs)
        
        # Convert to numpy array
        hires_images = np.asarray(hires_images)
        lowres_images = np.asarray(lowres_images)
        mag_images = np.asarray(mag_images)

        if reverse: #reverse images shape now (3, t, x, y) (or other combinations of x, y, z)
            hires_images = hires_images[:, :, ::-1, :] 
            lowres_images = lowres_images[:, :, ::-1, :] 
            mag_images = mag_images[:, :, ::-1, :] 
            mask = mask[:, ::-1,:] # mask shape (t, x, y)
            
        
        # Normalize the values 
        hires_images = self._normalize(hires_images, global_venc) # Velocity normalized to -1 .. 1
        lowres_images = self._normalize(lowres_images, global_venc)
        mag_images = mag_images / 4095. # Magnitude 0 .. 1
        
        # U-LR, HR, MAG, V-LR, HR, MAG, w-LR, HR, MAG, venc, MASK
        return lowres_images[0].astype('float32'), hires_images[0].astype('float32'), mag_images[0].astype('float32'), \
            lowres_images[1].astype('float32'), hires_images[1].astype('float32'), mag_images[1].astype('float32'), \
            lowres_images[2].astype('float32'), hires_images[2].astype('float32'), mag_images[2].astype('float32'),\
            global_venc.astype('float32'), mask.astype('float32')
    
    def _normalize(self, u, venc):
        return u / venc


class PatchHandler4D_extended_data_augmentation():
    # constructor
    def __init__(self, data_dir, patch_size, res_increase, batch_size, mask_threshold=0.6):
        self.patch_size = patch_size # this will be overwritten in augmenttaion file
        self.res_increase = res_increase
        self.batch_size = batch_size
        self.mask_threshold = mask_threshold

        self.data_directory = data_dir
        self.hr_colnames = ['u','v','w']
        self.lr_colnames = ['u','v','w']
        self.venc_colnames =  ['u_max', 'v_max', 'w_max']# ['venc_u','venc_v','venc_w'] #
        self.mag_colnames  = ['mag_u','mag_v','mag_w']
        self.mask_colname  = 'mask'
        self.colname2number = {'u':0, 'v':1, 'w':2}
        self.AUGMENT = True # augmentation, if False, then no augmenttaion is performed - increased speed??

    
    
    def initialize_dataset(self, indexes, shuffle, n_parallel=None):
        '''
            Input pipeline.
            This function accepts a list of filenames with index and patch locations to read.
        '''
        ds = tf.data.Dataset.from_tensor_slices((indexes))
        print("Total dataset:", len(indexes), 'shuffle', shuffle)

        if shuffle:
            # Set a buffer equal to dataset size to ensure randomness
            ds = ds.shuffle(buffer_size=len(indexes)) 

        # ds = ds.map(self.load_data_using_patch_index, num_parallel_calls=n_parallel)
        ds = ds.map(self.load_data_using_patch_index, num_parallel_calls=tf.data.AUTOTUNE) #chnages 13/06/25
        ds = ds.batch(batch_size=self.batch_size)
        
        # prefetch, n=number of items
        # ds = ds.prefetch(self.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        
        return ds
    
    def load_data_using_patch_index(self, indexes):
        return tf.py_function(func=self.load_patches_from_index_file, 
            # U-LR, HR, MAG, V-LR, HR, MAG, W-LR, HR, MAG, venc, MASK
            inp=[indexes], 
                Tout=[tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.float32])
    

    def load_patches_from_index_file(self, indexes):
        
        # Do typecasting, we need to make sure everything has the correct data type
        # Solution for tf2: https://stackoverflow.com/questions/56122670/how-to-get-string-value-out-of-tf-tensor-which-dtype-is-string
        lr_hd5path = '{}/{}'.format(self.data_directory, bytes.decode(indexes[0].numpy()))
        hd5path    = '{}/{}'.format(self.data_directory, bytes.decode(indexes[1].numpy()))
        
        
        #read out attributes from csvline
        axis= int(indexes[2])
        idx = int(indexes[3])
        start_t, start_1, start_2 = int(indexes[4]), int(indexes[5]), int(indexes[6])
        step_t = int(indexes[7])
        s_patchsize = int(indexes[8])
        t_patchsize = int(indexes[9])
        flip_1 = int(indexes[10])
        flip_2 = int(indexes[11])
        rot_angle = int(indexes[12])
        sign_u = int(indexes[13])
        sign_v = int(indexes[14])
        sign_w = int(indexes[15])
        swap_u = bytes.decode(indexes[16].numpy())
        swap_v = bytes.decode(indexes[17].numpy())
        swap_w = bytes.decode(indexes[18].numpy())
        coverage = float(indexes[19])

        
        # if step is 1, the loaded LR data is already downsampled
        if step_t == 1:
            # start_t_lr = int(start_t//self.res_increase)
            # hr_patch_size = int(t_patchsize*self.res_increase)
            # lr_patch_size = t_patchsize
            # start_t_hr = start_t
            # updated loading for temporal patches
            start_t_lr = start_t
            hr_patch_size = int(t_patchsize*self.res_increase)
            lr_patch_size = t_patchsize
            start_t_hr = int(start_t*self.res_increase)
        else:
            start_t_lr = start_t
            hr_patch_size = t_patchsize * step_t    #self.res_increase
            lr_patch_size = hr_patch_size
            start_t_hr = start_t

        # ============ get the patch ============ 
        if axis == 0 :
            patch_t_index       = np.index_exp[start_t_lr :start_t_lr+lr_patch_size:step_t,   idx, start_1:start_1+s_patchsize, start_2:start_2+s_patchsize]
            hr_t_patch_index    = np.index_exp[start_t_hr :start_t_hr+hr_patch_size,          idx, start_1:start_1+s_patchsize, start_2:start_2+s_patchsize]
            mask_t_index        = np.index_exp[start_t_hr :start_t_hr+hr_patch_size,          idx, start_1:start_1+s_patchsize, start_2:start_2+s_patchsize]
        elif axis == 1:
            patch_t_index       = np.index_exp[start_t_lr :start_t_lr+lr_patch_size:step_t,   start_1:start_1+s_patchsize,idx, start_2:start_2+s_patchsize]
            hr_t_patch_index    = np.index_exp[start_t_hr :start_t_hr+hr_patch_size,          start_1:start_1+s_patchsize,idx, start_2:start_2+s_patchsize]
            mask_t_index        = np.index_exp[start_t_hr :start_t_hr+hr_patch_size,          start_1:start_1+s_patchsize,idx, start_2:start_2+s_patchsize]
        elif axis == 2:
            patch_t_index       = np.index_exp[start_t_lr :start_t_lr+lr_patch_size:step_t,   start_1:start_1+s_patchsize, start_2:start_2+s_patchsize, idx]
            hr_t_patch_index    = np.index_exp[start_t_hr :start_t_hr+hr_patch_size,          start_1:start_1+s_patchsize, start_2:start_2+s_patchsize, idx]
            mask_t_index        = np.index_exp[start_t_hr :start_t_hr+hr_patch_size,          start_1:start_1+s_patchsize, start_2:start_2+s_patchsize, idx]

        
        u_patch, u_hr_patch, mag_u_patch, v_patch, v_hr_patch, mag_v_patch, w_patch, w_hr_patch, mag_w_patch, venc, mask_patch = self.load_vectorfield(hd5path, lr_hd5path, mask_t_index, patch_t_index, hr_t_patch_index, 
                                                                                                                                                       flip_1, flip_2, rot_angle, sign_u, sign_v, sign_w, swap_u, swap_v, swap_w)
        
        # Expand dims (for InputLayer)
        return u_patch[...,tf.newaxis], v_patch[...,tf.newaxis], w_patch[...,tf.newaxis], \
                    mag_u_patch[...,tf.newaxis], mag_v_patch[...,tf.newaxis], mag_w_patch[...,tf.newaxis], \
                    u_hr_patch[...,tf.newaxis], v_hr_patch[...,tf.newaxis], w_hr_patch[...,tf.newaxis], \
                    venc, mask_patch
                    

    def create_temporal_mask(self, mask, n_frames):
        '''
        from static mask create temporal mask of shape (n_frames, h, w, d)
        '''
        assert(len(mask.shape) == 3), " shape: " + str(mask.shape) # shape of mask is assumed to be 3 dimensional
        return np.repeat(np.expand_dims(mask, 0), n_frames, axis=0)

    def load_vectorfield(self, hd5path, lr_hd5path,  mask_index, patch_index, hr_patch_index, 
                         flip_1, flip_2, rot_angle, sign_u, sign_v, sign_w, swap_u, swap_v, swap_w):
        '''
            Load LowRes velocity and magnitude components, and HiRes velocity components
            Also returns the global venc and HiRes mask
        '''
        
        hires_images = []
        lowres_images = []
        mag_images = []
        vencs = []
        global_venc = 0

        if os.path.exists(hd5path) == False:
            print(f'File {hd5path} does not exist')
        if os.path.exists(lr_hd5path) == False:
            print(f'File {lr_hd5path} does not exist')

        # Load the U, V, W component of HR, LR, and MAG
        with h5py.File(hd5path, 'r') as hl:
            # Open the file once per row, Loop through all the HR column
            for i in range(len(self.hr_colnames)):
                w_hr = hl.get(self.hr_colnames[i])[hr_patch_index]
                # add them to the list
                hires_images.append(w_hr)

            # We only have 1 mask for all the objects in 1 file
            try:
                mask = hl.get(self.mask_colname)[mask_index] # Mask value [0 .. 1]
            except:
                print('Temporal static mask created')
                mask_temp = self.create_temporal_mask(np.asarray(hl.get(self.mask_colname)).squeeze(),  hl.get(self.hr_colnames[i]).shape[0])
                mask = mask_temp[mask_index] 
            mask = (mask >= self.mask_threshold) * 1.
            # print("mask shape:", hl.get(self.mask_colname).shape)
        
        with h5py.File(lr_hd5path, 'r') as hl:
            for i in range(len(self.lr_colnames)):
                w = hl.get(self.lr_colnames[i])[patch_index]
                mag_w = hl.get(self.mag_colnames[i])[patch_index]
                w_venc = np.array(hl.get(self.venc_colnames[i])).squeeze()
                            
                # add them to the list
                lowres_images.append(w)
                mag_images.append(mag_w)
                vencs.append(w_venc)
        
        global_venc = np.max(vencs)
        
        # Convert to numpy array
        hires_images = np.asarray(hires_images)
        lowres_images = np.asarray(lowres_images)
        mag_images = np.asarray(mag_images)

        # augm_title = get_augmentation_title(flip_1, flip_2, rot_angle, sign_u, sign_v, sign_w, swap_u, swap_v, swap_w)
        # print("Sum pre data augmentation :",augm_title, "--" , np.sum(hires_images, axis=(1, 2, 3)), np.sum(lowres_images, axis=(1, 2, 3)))
        # augm_title = get_augmentation_title(flip_1, flip_2, rot_angle, sign_u, sign_v, sign_w, swap_u, swap_v, swap_w)
        # print("Sum pre data augmentation :",augm_title, "--" , np.sum(hires_images, axis=(1, 2, 3)), np.sum(lowres_images, axis=(1, 2, 3)))

        # swap u and w from beginning 
        # print("Sum pre data augmentation :",augm_title, "--" , np.sum(hires_images, axis=(1, 2, 3)), np.sum(lowres_images, axis=(1, 2, 3)))

        # u_temp = hires_images[self.colname2number['u']].copy()
        # w_temp = hires_images[self.colname2number['w']].copy()
        # hires_images[self.colname2number['u']] = w_temp
        # hires_images[self.colname2number['w']] = u_temp

        # # low res
        # u_temp_lr = lowres_images[self.colname2number['u']].copy()
        # w_temp_lr = lowres_images[self.colname2number['w']].copy()
        # lowres_images[self.colname2number['u']] = w_temp_lr
        # lowres_images[self.colname2number['w']] = u_temp_lr
        # print("Sum pre data augmentation 2:",augm_title, "--" , np.sum(hires_images, axis=(1, 2, 3)), np.sum(lowres_images, axis=(1, 2, 3)))



        if self.AUGMENT:
            if flip_1 ==1 : #reverse images shape now (3, t, x, y) (or other combinations of x, y, z)
                hires_images = hires_images[:, :, ::-1, :] 
                lowres_images = lowres_images[:, :, ::-1, :] 
                mag_images = mag_images[:, :, ::-1, :] 
                mask = mask[:, ::-1,:] # mask shape (t, x, y)
            if flip_2 ==1 :
                hires_images = hires_images[:, :, :, ::-1] 
                lowres_images = lowres_images[:, :, :, ::-1] 
                mag_images = mag_images[:, :, :, ::-1] 
                mask = mask[:, :, ::-1]
            if rot_angle !=0:
                if rot_angle == 90:
                    k=1
                elif rot_angle == 180:
                    k=2
                elif rot_angle == 270:
                    k=3
                hires_images = np.rot90(hires_images, k=k, axes=(2, 3))
                lowres_images = np.rot90(lowres_images, k=k, axes=(2, 3))
                mag_images = np.rot90(mag_images, k=k, axes=(2, 3))
                mask = np.rot90(mask, k=k, axes=(1, 2))
            if sign_u == -1:
                hires_images[0] = -hires_images[0]
                lowres_images[0] = -lowres_images[0]
            if sign_v == -1:
                hires_images[1] = -hires_images[1]
                lowres_images[1] = -lowres_images[1]
            if sign_w == -1:
                hires_images[2] = -hires_images[2]
                lowres_images[2] = -lowres_images[2]
            
            if swap_u != 'u' or swap_v != 'v' :
                # Store the original values in a temporary variable
                temp_images_hr = [
                    hires_images[self.colname2number[swap_u]].copy(),
                    hires_images[self.colname2number[swap_v]].copy(),
                    hires_images[self.colname2number[swap_w]].copy()
                ]
                temp_images_lr = [
                    lowres_images[self.colname2number[swap_u]].copy(),
                    lowres_images[self.colname2number[swap_v]].copy(),
                    lowres_images[self.colname2number[swap_w]].copy()
                ]
                # Assign the temporary values back to hires_images
                hires_images[0], hires_images[1], hires_images[2] = temp_images_hr[0], temp_images_hr[1], temp_images_hr[2]
                lowres_images[0], lowres_images[1], lowres_images[2] = temp_images_lr[0], temp_images_lr[1], temp_images_lr[2]

                temp_images_hr = None
                temp_images_lr = None

            # if swap_u != 'u' and swap_v != 'v' :
            #     # Store the original values in a temporary variable
            #     temp_images_hr = [
            #         hires_images[self.colname2number[swap_u]],
            #         hires_images[self.colname2number[swap_v]],
            #         hires_images[self.colname2number[swap_w]]
            #     ]
            #     temp_images_lr = [
            #         lowres_images[self.colname2number[swap_u]],
            #         lowres_images[self.colname2number[swap_v]],
            #         lowres_images[self.colname2number[swap_w]]
            #     ]
            #     # Assign the temporary values back to hires_images
            #     hires_images[0], hires_images[1], hires_images[2] = temp_images_hr[0], temp_images_hr[1], temp_images_hr[2]
            #     lowres_images[0], lowres_images[1], lowres_images[2] = temp_images_lr[0], temp_images_lr[1], temp_images_lr[2]

            #     temp_images_hr = None
            #     temp_images_lr = None
        else: #no augmentation
            print("NO AUGMENTATION")
        
        # print("Sum post data augmentation:", augm_title, "--", np.sum(hires_images, axis=(1, 2, 3)), np.sum(lowres_images, axis=(1, 2, 3)))

        
        # Normalize the values 
        hires_images = self._normalize(hires_images, global_venc) # Velocity normalized to -1 .. 1
        lowres_images = self._normalize(lowres_images, global_venc)
        mag_images = mag_images / 4095. # Magnitude 0 .. 1
        
        # U-LR, HR, MAG, V-LR, HR, MAG, w-LR, HR, MAG, venc, MASK
        return lowres_images[0].astype('float32'), hires_images[0].astype('float32'), mag_images[0].astype('float32'), \
            lowres_images[1].astype('float32'), hires_images[1].astype('float32'), mag_images[1].astype('float32'), \
            lowres_images[2].astype('float32'), hires_images[2].astype('float32'), mag_images[2].astype('float32'),\
            global_venc.astype('float32'), mask.astype('float32')
    
    def _normalize(self, u, venc):
        return u / venc


class PatchHandler4D_extended_data_augmentation_optimized():
    # constructor
    def __init__(self, data_dir, patch_size, res_increase, batch_size, mask_threshold=0.6, csv_file=None):
        self.patch_size = patch_size # this will be overwritten in augmenttaion file
        self.res_increase = res_increase
        self.batch_size = batch_size
        self.mask_threshold = mask_threshold

        self.data_directory = data_dir
        self.hr_colnames = ['u','v','w']
        self.lr_colnames = ['u','v','w']
        self.venc_colnames =  ['u_max', 'v_max', 'w_max']# ['venc_u','venc_v','venc_w'] #
        self.mag_colnames  = ['mag_u','mag_v','mag_w']
        self.mask_colname  = 'mask'
        self.colname2number = {'u':0, 'v':1, 'w':2}
        self.colname_swap = {'u': 'u', 'v': 'v', 'w': 'w'}  # very important: this is only used of data is misaligned in the first place
        self._find_all_datamodels(csv_file)
        self.lr_files = {}  # Dictionary to hold LR datasets
        self.hr_files = {}  # Dictionary to hold HR datasets
        self._load_all_data()


    def _find_all_datamodels(self, csv_file):
        # Load CSV
        df = pd.read_csv(csv_file)

        # Ensure the LR and HR filename columns are correctly named; adjust if needed
        if 'source' not in df.columns or 'target' not in df.columns:
            raise ValueError("CSV must contain 'lr_filename' and 'hr_filename' columns.")

        # Drop duplicates to get unique combinations
        unique_pairs = df[['source', 'target']].drop_duplicates()

        # Convert to list of tuples
        pairs = list(unique_pairs.itertuples(index=False, name=None))

        # Store LR and HR dataset paths
        self.lr_datasets = ['{}/{}'.format(self.data_directory, lr) for lr, _ in pairs]
        self.hr_datasets = ['{}/{}'.format(self.data_directory, hr) for _, hr in pairs]

        print(f"Found {len(self.lr_datasets)} unique LR-HR pairs.")
        print("LR datasets:", self.lr_datasets)
        print("HR datasets:", self.hr_datasets)


    def _load_all_data(self):

        for lr_name, hr_name in zip(self.lr_datasets, self.hr_datasets):
            base_name_lr = os.path.basename(lr_name)
            base_name_hr = os.path.basename(hr_name)
            self.lr_files[base_name_lr] = {}
            self.hr_files[base_name_hr] = {}
            # Load the data from the HD5 files
            with h5py.File(lr_name, 'r') as lr_file, h5py.File(hr_name, 'r') as hr_file:
                
                
                # get lr data venc
                vencs = [np.array(lr_file[venc]) for venc in self.venc_colnames]
                global_venc = np.max(vencs)

                for lr_vel_colname, lr_mag_colname, hr_vel_colname in zip(self.lr_colnames, self.mag_colnames, self.hr_colnames):
                    # Normalize the velocity data
                    self.lr_files[base_name_lr][self.colname_swap[lr_vel_colname]] = self._normalize(np.array(lr_file[lr_vel_colname]),global_venc)
                    # Normalize the magnitude data
                    self.lr_files[base_name_lr][lr_mag_colname] = np.array(lr_file[lr_mag_colname]) / 4095.0

                    self.hr_files[base_name_hr][self.colname_swap[hr_vel_colname]] = self._normalize(np.array(hr_file[hr_vel_colname]),global_venc)

                    print(f"Loaded {lr_vel_colname} and {hr_vel_colname} and swapped {self.colname_swap[lr_vel_colname]} and {self.colname_swap[hr_vel_colname]}")


                mask = np.array(hr_file[self.mask_colname])
                mask = (mask >= self.mask_threshold) * 1.0  # Apply mask threshold
                if len(mask.shape) == 3:  # if mask is 3D, create temporal mask
                    mask_temp = self.create_temporal_mask(mask, hr_file[self.hr_colnames[0]].shape[0])
                    self.hr_files[base_name_hr][self.mask_colname] = mask_temp
                else:  # if mask is already 4D, just assign it
                    self.hr_files[base_name_hr][self.mask_colname] = mask

                self.hr_files[base_name_hr][self.mask_colname] = mask.astype('float32')
                self.lr_files[base_name_lr]['venc'] = global_venc.astype('float32')
        print("All data loaded from HD5 files.")

        print("LR files structure:", self.lr_files.keys())
        print("HR files structure:", self.hr_files.keys())

        print("LR files example:", self.lr_files[list(self.lr_files.keys())[0]].keys())
        print("HR files example:", self.hr_files[list(self.hr_files.keys())[0]].keys())
                
    
    
    def initialize_dataset(self, indexes, shuffle):
        '''
            Input pipeline.
            This function accepts a list of filenames with index and patch locations to read.
        '''
        ds = tf.data.Dataset.from_tensor_slices((indexes))
        print("Total dataset:", len(indexes), 'shuffle', shuffle)

        if shuffle:
            # Set a buffer equal to dataset size to ensure randomness
            ds = ds.shuffle(buffer_size=len(indexes)) 

        # ds = ds.map(self.load_data_using_patch_index, num_parallel_calls=n_parallel)
        ds = ds.map(self.load_data_using_patch_index, num_parallel_calls=tf.data.AUTOTUNE) #chnages 13/06/25
        ds = ds.batch(batch_size=self.batch_size)
        
        # prefetch, n=number of items
        # ds = ds.prefetch(self.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        
        return ds
    
    def load_data_using_patch_index(self, indexes):
        return tf.py_function(func=self.load_patches_from_index_file, 
            # U-LR, HR, MAG, V-LR, HR, MAG, W-LR, HR, MAG, venc, MASK
            inp=[indexes], 
                Tout=[tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.float32])
    
    def load_patches_from_index_file(self, indexes):
        
        # Do typecasting, we need to make sure everything has the correct data type
        # Solution for tf2: https://stackoverflow.com/questions/56122670/how-to-get-string-value-out-of-tf-tensor-which-dtype-is-string
        lr_hd5path = '{}/{}'.format(self.data_directory, bytes.decode(indexes[0].numpy()))
        hd5path    = '{}/{}'.format(self.data_directory, bytes.decode(indexes[1].numpy()))

        lr_key = os.path.basename(lr_hd5path)
        hr_key = os.path.basename(hd5path)
        
        
        #read out attributes from csvline
        axis= int(indexes[2])
        idx = int(indexes[3])
        start_t, start_1, start_2 = int(indexes[4]), int(indexes[5]), int(indexes[6])
        step_t = int(indexes[7])
        s_patchsize = int(indexes[8])
        t_patchsize = int(indexes[9])
        flip_1 = int(indexes[10])
        flip_2 = int(indexes[11])
        rot_angle = int(indexes[12])
        sign_u = int(indexes[13])
        sign_v = int(indexes[14])
        sign_w = int(indexes[15])
        swap_u = bytes.decode(indexes[16].numpy())
        swap_v = bytes.decode(indexes[17].numpy())
        swap_w = bytes.decode(indexes[18].numpy())

        coverage = float(indexes[19])

        
        # if step is 1, the loaded LR data is already downsampled
        if step_t == 1:
            # start_t_lr = int(start_t//self.res_increase)
            # hr_patch_size = int(t_patchsize*self.res_increase)
            # lr_patch_size = t_patchsize
            # start_t_hr = start_t
            # updated loading for temporal patches
            start_t_lr = start_t
            hr_patch_size = int(t_patchsize*self.res_increase)
            lr_patch_size = t_patchsize
            start_t_hr = int(start_t*self.res_increase)
        else:
            start_t_lr = start_t
            hr_patch_size = t_patchsize * step_t    #self.res_increase
            lr_patch_size = hr_patch_size
            start_t_hr = start_t

        # ============ get the patch ============ 
        if axis == 0 :
            patch_t_index       = np.index_exp[start_t_lr :start_t_lr+lr_patch_size:step_t,   idx, start_1:start_1+s_patchsize, start_2:start_2+s_patchsize]
            hr_t_patch_index    = np.index_exp[start_t_hr :start_t_hr+hr_patch_size,          idx, start_1:start_1+s_patchsize, start_2:start_2+s_patchsize]
            mask_t_index        = np.index_exp[start_t_hr :start_t_hr+hr_patch_size,          idx, start_1:start_1+s_patchsize, start_2:start_2+s_patchsize]
        elif axis == 1:
            patch_t_index       = np.index_exp[start_t_lr :start_t_lr+lr_patch_size:step_t,   start_1:start_1+s_patchsize,idx, start_2:start_2+s_patchsize]
            hr_t_patch_index    = np.index_exp[start_t_hr :start_t_hr+hr_patch_size,          start_1:start_1+s_patchsize,idx, start_2:start_2+s_patchsize]
            mask_t_index        = np.index_exp[start_t_hr :start_t_hr+hr_patch_size,          start_1:start_1+s_patchsize,idx, start_2:start_2+s_patchsize]
        elif axis == 2:
            patch_t_index       = np.index_exp[start_t_lr :start_t_lr+lr_patch_size:step_t,   start_1:start_1+s_patchsize, start_2:start_2+s_patchsize, idx]
            hr_t_patch_index    = np.index_exp[start_t_hr :start_t_hr+hr_patch_size,          start_1:start_1+s_patchsize, start_2:start_2+s_patchsize, idx]
            mask_t_index        = np.index_exp[start_t_hr :start_t_hr+hr_patch_size,          start_1:start_1+s_patchsize, start_2:start_2+s_patchsize, idx]

        
        u_patch, u_hr_patch, mag_u_patch, v_patch, v_hr_patch, mag_v_patch, w_patch, w_hr_patch, mag_w_patch, venc, mask_patch = self.load_vectorfield(hr_key, lr_key, mask_t_index, patch_t_index, hr_t_patch_index, 
                                                                                                                                                       flip_1, flip_2, rot_angle, sign_u, sign_v, sign_w, swap_u, swap_v, swap_w)
        
        # Expand dims (for InputLayer)
        return u_patch[...,tf.newaxis], v_patch[...,tf.newaxis], w_patch[...,tf.newaxis], \
                    mag_u_patch[...,tf.newaxis], mag_v_patch[...,tf.newaxis], mag_w_patch[...,tf.newaxis], \
                    u_hr_patch[...,tf.newaxis], v_hr_patch[...,tf.newaxis], w_hr_patch[...,tf.newaxis], \
                    venc, mask_patch
                    

    def create_temporal_mask(self, mask, n_frames):
        '''
        from static mask create temporal mask of shape (n_frames, h, w, d)
        '''
        assert(len(mask.shape) == 3), " shape: " + str(mask.shape) # shape of mask is assumed to be 3 dimensional
        return np.repeat(np.expand_dims(mask, 0), n_frames, axis=0)

    def load_vectorfield(self, hd5path, lr_hd5path,  mask_index, patch_index, hr_patch_index, 
                         flip_1, flip_2, rot_angle, sign_u, sign_v, sign_w, swap_u, swap_v, swap_w):
        '''
            Load LowRes velocity and magnitude components, and HiRes velocity components
            Also returns the global venc and HiRes mask
        '''
        
        hires_images = []
        lowres_images = []
        mag_images = []

        # shape (3, t, x, y)
        lowres_images   = np.stack([self.lr_files[lr_hd5path][colname][patch_index] for colname in self.lr_colnames], axis=0)
        mag_images      = np.stack([self.lr_files[lr_hd5path][colname][patch_index] for colname in self.mag_colnames], axis=0)
        hires_images    = np.stack([self.hr_files[hd5path][colname][hr_patch_index] for colname in self.hr_colnames], axis=0)

        mask = self.hr_files[hd5path][self.mask_colname][mask_index]  # Mask value [0 .. 1]
        # vencs = [np.array(self.lr_files[lr_hd5path][venc]) for venc in self.venc_colnames]

        global_venc = self.lr_files[lr_hd5path]['venc'] 
        
        
        # Convert to numpy array
        hires_images = np.asarray(hires_images)
        lowres_images = np.asarray(lowres_images)
        mag_images = np.asarray(mag_images)


        if flip_1 ==1 : #reverse images shape now (3, t, x, y) (or other combinations of x, y, z)
            hires_images = hires_images[:, :, ::-1, :] 
            lowres_images = lowres_images[:, :, ::-1, :] 
            mag_images = mag_images[:, :, ::-1, :] 
            mask = mask[:, ::-1,:] # mask shape (t, x, y)
        if flip_2 ==1 :
            hires_images = hires_images[:, :, :, ::-1] 
            lowres_images = lowres_images[:, :, :, ::-1] 
            mag_images = mag_images[:, :, :, ::-1] 
            mask = mask[:, :, ::-1]
        if rot_angle !=0:
            if rot_angle == 90:
                k=1
            elif rot_angle == 180:
                k=2
            elif rot_angle == 270:
                k=3
            hires_images    = np.rot90(hires_images, k=k, axes=(2, 3))
            lowres_images   = np.rot90(lowres_images, k=k, axes=(2, 3))
            mag_images      = np.rot90(mag_images, k=k, axes=(2, 3))
            mask            = np.rot90(mask, k=k, axes=(1, 2))
        if sign_u == -1:
            hires_images[0] *= -1
            lowres_images[0] *= -1
        if sign_v == -1:
            hires_images[1] *= -1
            lowres_images[1] *= -1
        if sign_w == -1:
            hires_images[2] *= -1
            lowres_images[2] *= -1
        
        if swap_u != 'u' or swap_v != 'v' :
            # Store the original values in a temporary variable
            temp_images_hr = [
                hires_images[self.colname2number[swap_u]].copy(),
                hires_images[self.colname2number[swap_v]].copy(),
                hires_images[self.colname2number[swap_w]].copy()
            ]
            temp_images_lr = [
                lowres_images[self.colname2number[swap_u]].copy(),
                lowres_images[self.colname2number[swap_v]].copy(),
                lowres_images[self.colname2number[swap_w]].copy()
            ]
            # Assign the temporary values back to hires_images
            hires_images[0], hires_images[1], hires_images[2] = temp_images_hr[0], temp_images_hr[1], temp_images_hr[2]
            lowres_images[0], lowres_images[1], lowres_images[2] = temp_images_lr[0], temp_images_lr[1], temp_images_lr[2]

            temp_images_hr = None
            temp_images_lr = None

        # print("Sum post data augmentation:", augm_title, "--", np.sum(hires_images, axis=(1, 2, 3)), np.sum(lowres_images, axis=(1, 2, 3)))

        
        # Normalize the values 
        # check if normalized
        # if np.max(hires_images) > 1.0 or np.max(lowres_images) > 1.0 or np.max(mag_images) > 1.0:
        #     print("Warning: Images are not normalized, exiting.")
        #     print("Max values - HiRes:", np.max(hires_images), "LowRes:", np.max(lowres_images), "Mag:", np.max(mag_images))
        #     print("venc is :", global_venc)
        #     print("Max values - HiRes:", np.max(hires_images), "LowRes:", np.max(lowres_images), "Mag:", np.max(mag_images))
        
        # hires_images = self._normalize(hires_images, global_venc) # Velocity normalized to -1 .. 1
        # lowres_images = self._normalize(lowres_images, global_venc)
        # mag_images = mag_images / 4095. # Magnitude 0 .. 1
        
        # U-LR, HR, MAG, V-LR, HR, MAG, w-LR, HR, MAG, venc, MASK
        return lowres_images[0].astype('float32'), hires_images[0].astype('float32'), mag_images[0].astype('float32'), \
            lowres_images[1].astype('float32'), hires_images[1].astype('float32'), mag_images[1].astype('float32'), \
            lowres_images[2].astype('float32'), hires_images[2].astype('float32'), mag_images[2].astype('float32'),\
            global_venc.astype('float32'), mask.astype('float32')
    
    def _normalize(self, u, venc):
        return u / venc



