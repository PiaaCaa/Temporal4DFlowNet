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
    """
    On-the-fly HDF5 patch loader with multi-axis support and optional data augmentation.
    Covers the functionality of the old PatchHandler4D, PatchHandler4D_all_axis,
    and PatchHandler4D_extended_data_augmentation.

    Args:
        data_dir:        Path to data directory.
        patch_size:      Patch size (t, x, y).
        res_increase:    Temporal resolution increase factor.
        batch_size:      Batch size for the dataset.
        mask_threshold:  Threshold for binarizing the mask (default 0.6).
        augment:         If True, apply data augmentation (flip, rotation, sign, swap).
                         If False, no augmentation is applied (default True).
    """

    def __init__(self, data_dir, patch_size, res_increase, batch_size, mask_threshold=0.6, augment=True):
        self.patch_size = patch_size
        self.res_increase = res_increase
        self.batch_size = batch_size
        self.mask_threshold = mask_threshold
        self.AUGMENT = augment  # Exposed as constructor parameter (was hardcoded True before)

        self.data_directory = data_dir
        self.hr_colnames = ['u', 'v', 'w']
        self.lr_colnames = ['u', 'v', 'w']
        self.venc_colnames = ['u_max', 'v_max', 'w_max']
        self.mag_colnames = ['mag_u', 'mag_v', 'mag_w']
        self.mask_colname = 'mask'
        self.colname2number = {'u': 0, 'v': 1, 'w': 2}

    def initialize_dataset(self, indexes, shuffle, n_parallel=None):
        '''
            Input pipeline.
            This function accepts a list of filenames with index and patch locations to read.
        '''
        ds = tf.data.Dataset.from_tensor_slices((indexes))
        print("Total dataset:", len(indexes), 'shuffle', shuffle)

        if shuffle:
            ds = ds.shuffle(buffer_size=len(indexes))

        ds = ds.map(self.load_data_using_patch_index, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size=self.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds

    def load_data_using_patch_index(self, indexes):
        out = tf.py_function(func=self.load_patches_from_index_file, inp=[indexes],
                             Tout=[tf.float32] * 11)

        # Set shape hints for graph optimization (from PatchHandler4D_all_axis)
        t, x, y = self.patch_size
        for tensor in out[:-2]:
            tensor.set_shape([t, x, y, 1])
        out[-2].set_shape([])       # venc is scalar
        out[-1].set_shape([t, x, y])  # mask has no channel dim

        return tuple(out)

    def load_patches_from_index_file(self, indexes):
        lr_hd5path = '{}/{}'.format(self.data_directory, bytes.decode(indexes[0].numpy()))
        hd5path    = '{}/{}'.format(self.data_directory, bytes.decode(indexes[1].numpy()))

        # Read attributes from CSV line
        axis = int(indexes[2])
        idx = int(indexes[3])
        start_t, start_1, start_2 = int(indexes[4]), int(indexes[5]), int(indexes[6])
        step_t = int(indexes[7])
        s_patchsize = int(indexes[8])
        t_patchsize = int(indexes[9])
        flip_1 = int(indexes[10])   # replaces 'reverse' from PatchHandler4D_all_axis
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
            start_t_lr = start_t
            hr_patch_size = int(t_patchsize * self.res_increase)
            lr_patch_size = t_patchsize
            start_t_hr = int(start_t * self.res_increase)
        else:
            start_t_lr = start_t
            hr_patch_size = t_patchsize * step_t
            lr_patch_size = hr_patch_size
            start_t_hr = start_t

        # ============ get the patch ============
        if axis == 0:
            patch_t_index    = np.index_exp[start_t_lr:start_t_lr+lr_patch_size:step_t, idx, start_1:start_1+s_patchsize, start_2:start_2+s_patchsize]
            hr_t_patch_index = np.index_exp[start_t_hr:start_t_hr+hr_patch_size,        idx, start_1:start_1+s_patchsize, start_2:start_2+s_patchsize]
            mask_t_index     = np.index_exp[start_t_hr:start_t_hr+hr_patch_size,        idx, start_1:start_1+s_patchsize, start_2:start_2+s_patchsize]
        elif axis == 1:
            patch_t_index    = np.index_exp[start_t_lr:start_t_lr+lr_patch_size:step_t, start_1:start_1+s_patchsize, idx, start_2:start_2+s_patchsize]
            hr_t_patch_index = np.index_exp[start_t_hr:start_t_hr+hr_patch_size,        start_1:start_1+s_patchsize, idx, start_2:start_2+s_patchsize]
            mask_t_index     = np.index_exp[start_t_hr:start_t_hr+hr_patch_size,        start_1:start_1+s_patchsize, idx, start_2:start_2+s_patchsize]
        elif axis == 2:
            patch_t_index    = np.index_exp[start_t_lr:start_t_lr+lr_patch_size:step_t, start_1:start_1+s_patchsize, start_2:start_2+s_patchsize, idx]
            hr_t_patch_index = np.index_exp[start_t_hr:start_t_hr+hr_patch_size,        start_1:start_1+s_patchsize, start_2:start_2+s_patchsize, idx]
            mask_t_index     = np.index_exp[start_t_hr:start_t_hr+hr_patch_size,        start_1:start_1+s_patchsize, start_2:start_2+s_patchsize, idx]

        u_patch, u_hr_patch, mag_u_patch, \
        v_patch, v_hr_patch, mag_v_patch, \
        w_patch, w_hr_patch, mag_w_patch, \
        venc, mask_patch = self.load_vectorfield(hd5path, lr_hd5path, mask_t_index, patch_t_index, hr_t_patch_index,
                                                 flip_1, flip_2, rot_angle, sign_u, sign_v, sign_w, swap_u, swap_v, swap_w)

        return u_patch[..., tf.newaxis], v_patch[..., tf.newaxis], w_patch[..., tf.newaxis], \
               mag_u_patch[..., tf.newaxis], mag_v_patch[..., tf.newaxis], mag_w_patch[..., tf.newaxis], \
               u_hr_patch[..., tf.newaxis], v_hr_patch[..., tf.newaxis], w_hr_patch[..., tf.newaxis], \
               venc, mask_patch

    def create_temporal_mask(self, mask, n_frames):
        '''
        from static mask create temporal mask of shape (n_frames, h, w, d)
        '''
        assert(len(mask.shape) == 3), " shape: " + str(mask.shape)
        return np.repeat(np.expand_dims(mask, 0), n_frames, axis=0)

    def load_vectorfield(self, hd5path, lr_hd5path, mask_index, patch_index, hr_patch_index,
                         flip_1, flip_2, rot_angle, sign_u, sign_v, sign_w, swap_u, swap_v, swap_w):
        '''
            Load LowRes velocity and magnitude components, and HiRes velocity components.
            Also returns the global venc and HiRes mask.
        '''
        hires_images = []
        lowres_images = []
        mag_images = []
        vencs = []

        if not os.path.exists(hd5path):
            print(f'File {hd5path} does not exist')
        if not os.path.exists(lr_hd5path):
            print(f'File {lr_hd5path} does not exist')

        with h5py.File(hd5path, 'r') as hl:
            for i in range(len(self.hr_colnames)):
                w_hr = hl.get(self.hr_colnames[i])[hr_patch_index]
                hires_images.append(w_hr)

            try:
                mask = hl.get(self.mask_colname)[mask_index]
            except:
                print('Temporal static mask created')
                mask_temp = self.create_temporal_mask(np.asarray(hl.get(self.mask_colname)).squeeze(),
                                                      hl.get(self.hr_colnames[i]).shape[0])
                mask = mask_temp[mask_index]
            mask = (mask >= self.mask_threshold) * 1.

        with h5py.File(lr_hd5path, 'r') as hl:
            for i in range(len(self.lr_colnames)):
                w = hl.get(self.lr_colnames[i])[patch_index]
                mag_w = hl.get(self.mag_colnames[i])[patch_index]
                w_venc = np.array(hl.get(self.venc_colnames[i])).squeeze()

                lowres_images.append(w)
                mag_images.append(mag_w)
                vencs.append(w_venc)

        global_venc = np.max(vencs)

        hires_images  = np.asarray(hires_images)
        lowres_images = np.asarray(lowres_images)
        mag_images    = np.asarray(mag_images)

        if self.AUGMENT:
            if flip_1 == 1:  # replaces 'reverse' from PatchHandler4D_all_axis
                hires_images  = hires_images[:, :, ::-1, :]
                lowres_images = lowres_images[:, :, ::-1, :]
                mag_images    = mag_images[:, :, ::-1, :]
                mask          = mask[:, ::-1, :]
            if flip_2 == 1:
                hires_images  = hires_images[:, :, :, ::-1]
                lowres_images = lowres_images[:, :, :, ::-1]
                mag_images    = mag_images[:, :, :, ::-1]
                mask          = mask[:, :, ::-1]
            if rot_angle != 0:
                k = {90: 1, 180: 2, 270: 3}[rot_angle]
                hires_images  = np.rot90(hires_images, k=k, axes=(2, 3))
                lowres_images = np.rot90(lowres_images, k=k, axes=(2, 3))
                mag_images    = np.rot90(mag_images, k=k, axes=(2, 3))
                mask          = np.rot90(mask, k=k, axes=(1, 2))
            if sign_u == -1:
                hires_images[0] = -hires_images[0]
                lowres_images[0] = -lowres_images[0]
            if sign_v == -1:
                hires_images[1] = -hires_images[1]
                lowres_images[1] = -lowres_images[1]
            if sign_w == -1:
                hires_images[2] = -hires_images[2]
                lowres_images[2] = -lowres_images[2]
            if swap_u != 'u' or swap_v != 'v':
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
                hires_images[0], hires_images[1], hires_images[2]   = temp_images_hr[0], temp_images_hr[1], temp_images_hr[2]
                lowres_images[0], lowres_images[1], lowres_images[2] = temp_images_lr[0], temp_images_lr[1], temp_images_lr[2]
        else:
            print("NO AUGMENTATION")

        hires_images  = self._normalize(hires_images, global_venc)
        lowres_images = self._normalize(lowres_images, global_venc)
        mag_images    = mag_images / 4095.

        return lowres_images[0].astype('float32'), hires_images[0].astype('float32'), mag_images[0].astype('float32'), \
               lowres_images[1].astype('float32'), hires_images[1].astype('float32'), mag_images[1].astype('float32'), \
               lowres_images[2].astype('float32'), hires_images[2].astype('float32'), mag_images[2].astype('float32'), \
               global_venc.astype('float32'), mask.astype('float32')

    def _normalize(self, u, venc):
        return u / venc


class PatchHandler4D_preloaded():
    """
    Pre-loads all HDF5 data into RAM at init for faster patch loading during training.
    Normalization is applied once at load time. Supports the same augmentation as PatchHandler4D.

    Args:
        data_dir:        Path to data directory.
        patch_size:      Patch size (t, x, y).
        res_increase:    Temporal resolution increase factor.
        batch_size:      Batch size for the dataset.
        mask_threshold:  Threshold for binarizing the mask (default 0.6).
        csv_file:        Path to CSV file listing LR/HR file pairs (required).
    """

    def __init__(self, data_dir, patch_size, res_increase, batch_size, mask_threshold=0.6, csv_file=None):
        self.patch_size = patch_size
        self.res_increase = res_increase
        self.batch_size = batch_size
        self.mask_threshold = mask_threshold

        self.data_directory = data_dir
        self.hr_colnames = ['u', 'v', 'w']
        self.lr_colnames = ['u', 'v', 'w']
        self.venc_colnames = ['u_max', 'v_max', 'w_max']
        self.mag_colnames = ['mag_u', 'mag_v', 'mag_w']
        self.mask_colname = 'mask'
        self.colname2number = {'u': 0, 'v': 1, 'w': 2}
        self.colname_swap = {'u': 'u', 'v': 'v', 'w': 'w'}
        self._find_all_datamodels(csv_file)
        self.lr_files = {}
        self.hr_files = {}
        self._load_all_data()

    def _find_all_datamodels(self, csv_file):
        df = pd.read_csv(csv_file)

        if 'source' not in df.columns or 'target' not in df.columns:
            raise ValueError("CSV must contain 'source' and 'target' columns.")

        unique_pairs = df[['source', 'target']].drop_duplicates()
        pairs = list(unique_pairs.itertuples(index=False, name=None))

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

            with h5py.File(lr_name, 'r') as lr_file, h5py.File(hr_name, 'r') as hr_file:
                vencs = [np.array(lr_file[venc]) for venc in self.venc_colnames]
                global_venc = np.max(vencs)

                for lr_vel_colname, lr_mag_colname, hr_vel_colname in zip(self.lr_colnames, self.mag_colnames, self.hr_colnames):
                    self.lr_files[base_name_lr][self.colname_swap[lr_vel_colname]] = self._normalize(np.array(lr_file[lr_vel_colname]), global_venc)
                    self.lr_files[base_name_lr][lr_mag_colname] = np.array(lr_file[lr_mag_colname]) / 4095.0
                    self.hr_files[base_name_hr][self.colname_swap[hr_vel_colname]] = self._normalize(np.array(hr_file[hr_vel_colname]), global_venc)
                    print(f"Loaded {lr_vel_colname} and {hr_vel_colname} and swapped {self.colname_swap[lr_vel_colname]} and {self.colname_swap[hr_vel_colname]}")

                mask = np.array(hr_file[self.mask_colname])
                mask = (mask >= self.mask_threshold) * 1.0
                if len(mask.shape) == 3:
                    mask_temp = self.create_temporal_mask(mask, hr_file[self.hr_colnames[0]].shape[0])
                    self.hr_files[base_name_hr][self.mask_colname] = mask_temp
                else:
                    self.hr_files[base_name_hr][self.mask_colname] = mask

                self.hr_files[base_name_hr][self.mask_colname] = mask.astype('float32')
                self.lr_files[base_name_lr]['venc'] = global_venc.astype('float32')

        print("All data loaded from HD5 files.")
        print("LR files structure:", self.lr_files.keys())
        print("HR files structure:", self.hr_files.keys())
        print("LR files example:", self.lr_files[list(self.lr_files.keys())[0]].keys())
        print("HR files example:", self.hr_files[list(self.hr_files.keys())[0]].keys())

    def initialize_dataset(self, indexes, shuffle):
        ds = tf.data.Dataset.from_tensor_slices((indexes))
        print("Total dataset:", len(indexes), 'shuffle', shuffle)

        if shuffle:
            ds = ds.shuffle(buffer_size=len(indexes))

        ds = ds.map(self.load_data_using_patch_index, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size=self.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds

    def load_data_using_patch_index(self, indexes):
        return tf.py_function(func=self.load_patches_from_index_file,
                              inp=[indexes],
                              Tout=[tf.float32] * 11)

    def load_patches_from_index_file(self, indexes):
        lr_hd5path = '{}/{}'.format(self.data_directory, bytes.decode(indexes[0].numpy()))
        hd5path    = '{}/{}'.format(self.data_directory, bytes.decode(indexes[1].numpy()))

        lr_key = os.path.basename(lr_hd5path)
        hr_key = os.path.basename(hd5path)

        axis = int(indexes[2])
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

        if step_t == 1:
            start_t_lr = start_t
            hr_patch_size = int(t_patchsize * self.res_increase)
            lr_patch_size = t_patchsize
            start_t_hr = int(start_t * self.res_increase)
        else:
            start_t_lr = start_t
            hr_patch_size = t_patchsize * step_t
            lr_patch_size = hr_patch_size
            start_t_hr = start_t

        if axis == 0:
            patch_t_index    = np.index_exp[start_t_lr:start_t_lr+lr_patch_size:step_t, idx, start_1:start_1+s_patchsize, start_2:start_2+s_patchsize]
            hr_t_patch_index = np.index_exp[start_t_hr:start_t_hr+hr_patch_size,        idx, start_1:start_1+s_patchsize, start_2:start_2+s_patchsize]
            mask_t_index     = np.index_exp[start_t_hr:start_t_hr+hr_patch_size,        idx, start_1:start_1+s_patchsize, start_2:start_2+s_patchsize]
        elif axis == 1:
            patch_t_index    = np.index_exp[start_t_lr:start_t_lr+lr_patch_size:step_t, start_1:start_1+s_patchsize, idx, start_2:start_2+s_patchsize]
            hr_t_patch_index = np.index_exp[start_t_hr:start_t_hr+hr_patch_size,        start_1:start_1+s_patchsize, idx, start_2:start_2+s_patchsize]
            mask_t_index     = np.index_exp[start_t_hr:start_t_hr+hr_patch_size,        start_1:start_1+s_patchsize, idx, start_2:start_2+s_patchsize]
        elif axis == 2:
            patch_t_index    = np.index_exp[start_t_lr:start_t_lr+lr_patch_size:step_t, start_1:start_1+s_patchsize, start_2:start_2+s_patchsize, idx]
            hr_t_patch_index = np.index_exp[start_t_hr:start_t_hr+hr_patch_size,        start_1:start_1+s_patchsize, start_2:start_2+s_patchsize, idx]
            mask_t_index     = np.index_exp[start_t_hr:start_t_hr+hr_patch_size,        start_1:start_1+s_patchsize, start_2:start_2+s_patchsize, idx]

        u_patch, u_hr_patch, mag_u_patch, \
        v_patch, v_hr_patch, mag_v_patch, \
        w_patch, w_hr_patch, mag_w_patch, \
        venc, mask_patch = self.load_vectorfield(hr_key, lr_key, mask_t_index, patch_t_index, hr_t_patch_index,
                                                 flip_1, flip_2, rot_angle, sign_u, sign_v, sign_w, swap_u, swap_v, swap_w)

        return u_patch[..., tf.newaxis], v_patch[..., tf.newaxis], w_patch[..., tf.newaxis], \
               mag_u_patch[..., tf.newaxis], mag_v_patch[..., tf.newaxis], mag_w_patch[..., tf.newaxis], \
               u_hr_patch[..., tf.newaxis], v_hr_patch[..., tf.newaxis], w_hr_patch[..., tf.newaxis], \
               venc, mask_patch

    def create_temporal_mask(self, mask, n_frames):
        assert(len(mask.shape) == 3), " shape: " + str(mask.shape)
        return np.repeat(np.expand_dims(mask, 0), n_frames, axis=0)

    def load_vectorfield(self, hd5path, lr_hd5path, mask_index, patch_index, hr_patch_index,
                         flip_1, flip_2, rot_angle, sign_u, sign_v, sign_w, swap_u, swap_v, swap_w):

        lowres_images = np.stack([self.lr_files[lr_hd5path][colname][patch_index] for colname in self.lr_colnames], axis=0)
        mag_images    = np.stack([self.lr_files[lr_hd5path][colname][patch_index] for colname in self.mag_colnames], axis=0)
        hires_images  = np.stack([self.hr_files[hd5path][colname][hr_patch_index] for colname in self.hr_colnames], axis=0)
        mask          = self.hr_files[hd5path][self.mask_colname][mask_index]
        global_venc   = self.lr_files[lr_hd5path]['venc']

        hires_images  = np.asarray(hires_images)
        lowres_images = np.asarray(lowres_images)
        mag_images    = np.asarray(mag_images)

        if flip_1 == 1:
            hires_images  = hires_images[:, :, ::-1, :]
            lowres_images = lowres_images[:, :, ::-1, :]
            mag_images    = mag_images[:, :, ::-1, :]
            mask          = mask[:, ::-1, :]
        if flip_2 == 1:
            hires_images  = hires_images[:, :, :, ::-1]
            lowres_images = lowres_images[:, :, :, ::-1]
            mag_images    = mag_images[:, :, :, ::-1]
            mask          = mask[:, :, ::-1]
        if rot_angle != 0:
            k = {90: 1, 180: 2, 270: 3}[rot_angle]
            hires_images  = np.rot90(hires_images, k=k, axes=(2, 3))
            lowres_images = np.rot90(lowres_images, k=k, axes=(2, 3))
            mag_images    = np.rot90(mag_images, k=k, axes=(2, 3))
            mask          = np.rot90(mask, k=k, axes=(1, 2))
        if sign_u == -1:
            hires_images[0]  *= -1
            lowres_images[0] *= -1
        if sign_v == -1:
            hires_images[1]  *= -1
            lowres_images[1] *= -1
        if sign_w == -1:
            hires_images[2]  *= -1
            lowres_images[2] *= -1
        if swap_u != 'u' or swap_v != 'v':
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
            hires_images[0], hires_images[1], hires_images[2]   = temp_images_hr[0], temp_images_hr[1], temp_images_hr[2]
            lowres_images[0], lowres_images[1], lowres_images[2] = temp_images_lr[0], temp_images_lr[1], temp_images_lr[2]

        return lowres_images[0].astype('float32'), hires_images[0].astype('float32'), mag_images[0].astype('float32'), \
               lowres_images[1].astype('float32'), hires_images[1].astype('float32'), mag_images[1].astype('float32'), \
               lowres_images[2].astype('float32'), hires_images[2].astype('float32'), mag_images[2].astype('float32'), \
               global_venc.astype('float32'), mask.astype('float32')

    def _normalize(self, u, venc):
        return u / venc