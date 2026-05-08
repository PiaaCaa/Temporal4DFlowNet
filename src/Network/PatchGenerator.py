import numpy as np
import h5py
from utils import ImageDataset_temporal

class PatchGenerator():
    def __init__(self, patch_size, res_increase, include_all_axis = False, downsample_input_first = True, downsampling_factor = 2):

        # check instance of patch size
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
            print("Patch size is set to a cube of size", patch_size)
        elif isinstance(patch_size, tuple):
            if len(patch_size) != 3:
                raise ValueError("Patch size must be a tuple of three integers.")
        else:
            raise TypeError("Patch size must be an integer or a tuple of three integers.")

        self.patch_size_tuple = patch_size
        # lower for temporal
        self.effective_patch_size_tuple = (patch_size[0]-4, patch_size[1]-4,patch_size[2]-4,) # patch_size - 4 # we strip down 2 from each sides (on LR)
        self.res_increase = res_increase
        self.all_axis = include_all_axis
        self.downsample_input_first = downsample_input_first
        self.downsampling_factor = downsampling_factor
        # we make sure we pad it on the far side of x,y,z so the division will match
        self.padding = (0,0,0) 
        if not downsample_input_first: print("Data will NOT get downsampled first for prediction")

    def patchify(self, dataset: ImageDataset_temporal):
        """
            Create overlapping patch of size of patch_size
            On LR, we exclude 2 px from each side, effectively the size being used is patch_size-4
            On HR, the excluded pixels are (2*res_increase) from each side
        """
        u_stacks, i,j,k = self._generate_overlapping_patches(dataset.u)
        v_stacks, i,j,k = self._generate_overlapping_patches(dataset.v)
        w_stacks, i,j,k = self._generate_overlapping_patches(dataset.w)
        umag_stacks, i,j,k = self._generate_overlapping_patches(dataset.mag_u)
        vmag_stacks, i,j,k = self._generate_overlapping_patches(dataset.mag_v)
        wmag_stacks, i,j,k = self._generate_overlapping_patches(dataset.mag_w)
        
        # Store this info for unpatchify
        self.nr_x = i
        self.nr_y = j
        self.nr_z = k
        
        # Expand dims for tf.keras input shape
        u_stacks = np.expand_dims(u_stacks, -1)
        v_stacks = np.expand_dims(v_stacks, -1)
        w_stacks = np.expand_dims(w_stacks, -1)

        umag_stacks = np.expand_dims(umag_stacks, -1)
        vmag_stacks = np.expand_dims(vmag_stacks, -1)
        wmag_stacks = np.expand_dims(wmag_stacks, -1)

        return (u_stacks, v_stacks, w_stacks), (umag_stacks, vmag_stacks, wmag_stacks)
    
    def unpatchify(self, results):
        """
            Reconstruct the 3-velocity components back to its original shape
        """
        prediction_u = self._patchup_with_overlap(results[:,:,:,:,0], self.nr_x, self.nr_y, self.nr_z)
        prediction_v = self._patchup_with_overlap(results[:,:,:,:,1], self.nr_x, self.nr_y, self.nr_z)
        prediction_w = self._patchup_with_overlap(results[:,:,:,:,2], self.nr_x, self.nr_y, self.nr_z)

        #return predictions
        return prediction_u, prediction_v, prediction_w

    def _pad_to_patch_size_with_overlap(self, img):
        """
            Pad image to the right, until it is exactly divisible by patch size
        """
        if (self.all_axis and self.downsample_input_first):
            
            img = img[::self.downsampling_factor, :, :]  # from shape (T, X, Y) to (1/2 T, X, Y) (or other combinations of X; Y; Z)

        side_pad_x = (self.patch_size_tuple[0]-self.effective_patch_size_tuple[0]) // 2
        side_pad_y = (self.patch_size_tuple[1]-self.effective_patch_size_tuple[1]) // 2
        side_pad_z = (self.patch_size_tuple[2]-self.effective_patch_size_tuple[2]) // 2
        
        # mandatory padding
        img = np.pad(img, ((0, 0),(side_pad_y, side_pad_y),(side_pad_z, side_pad_z)), 'constant')
        img = np.pad(img, ((side_pad_x, side_pad_x),(0, 0),(0, 0)), 'wrap')
        
        res_x = (img.shape[0] % self.effective_patch_size_tuple[0])
        if (res_x > (2* side_pad_x)):
            pad_x = self.patch_size_tuple[0] - res_x
        else:
            pad_x = (2 * side_pad_x) - res_x

        res_y = (img.shape[1] % self.effective_patch_size_tuple[1])
        if (res_y > (2* side_pad_y)):
            pad_y = self.patch_size_tuple[1] - res_y
        else:
            pad_y = (2 * side_pad_y) - res_y

        res_z = (img.shape[2] % self.effective_patch_size_tuple[2])
        if (res_z > (2* side_pad_z)):
            pad_z = self.patch_size_tuple[2] - res_z
        else:
            pad_z = (2 * side_pad_z) - res_z
        
        img = np.pad(img, ((0, pad_x),(0, pad_y),(0, pad_z)), 'constant')

        # the padding is for the HiRes version because we need to reconstruct the result later
        #changed from spatial to temporal SR
        self.padding = (pad_x*self.res_increase, pad_y, pad_z)

        return img

    def _generate_overlapping_patches(self, img):
        
        patch_size = self.patch_size_tuple
        
        img = self._pad_to_patch_size_with_overlap(img)

        # all_pads = (self.patch_size_tuple - self.effective_patch_size_tuple)
        pad_x = self.patch_size_tuple[0] - self.effective_patch_size_tuple[0]
        pad_y = self.patch_size_tuple[1] - self.effective_patch_size_tuple[1]
        pad_z = self.patch_size_tuple[2] - self.effective_patch_size_tuple[2]

        u_stack = []
        
        nr_x = (img.shape[0]-pad_x) // self.effective_patch_size_tuple[0]
        nr_y = (img.shape[1]-pad_y) // self.effective_patch_size_tuple[1]
        nr_z = (img.shape[2]-pad_z) // self.effective_patch_size_tuple[2]
        
        for i in range(nr_x):
            x_start = i * self.effective_patch_size_tuple[0] #stride x
            for j in range(nr_y):
                y_start = j * self.effective_patch_size_tuple[1] #stride y
                for k in range(nr_z):
                    z_start = k * self.effective_patch_size_tuple[2] #stride z
                    
                    patch_index  = np.index_exp[x_start:x_start+patch_size[0], y_start:y_start+patch_size[1], z_start:z_start+patch_size[2]]
                    
                    u_loop = img[patch_index]
                    u_stack.append(u_loop)
                                  
        return np.asarray(u_stack), nr_x, nr_y, nr_z
            # return the number of of i j k elements        

    def _patchup_with_overlap(self, patches, x, y, z):
        """
            Reconstruct the image from the patches
        """
        print("Prediction size:", patches.shape)
        # TODO check this new outcome
        #patches size: n patches, p1, p2, p3(p1 patches size in 1)

        side_pad_x = (self.patch_size_tuple[0] - self.effective_patch_size_tuple[0]) // 2
        side_pax_y = (self.patch_size_tuple[1] - self.effective_patch_size_tuple[1]) // 2
        side_pad_hr_x =  side_pad_x * self.res_increase # size pad hr = 0
        
        side_pad_hr_spatial = side_pax_y # no increase in spatial domain 
        patch_size_thr = patches.shape[1]  # patchsize = 20 # temproal high resolution
        patch_size_shr = patches.shape[2]   # spatial high resolution

        n = patch_size_thr-side_pad_hr_x
        n_spatial = patch_size_shr - side_pad_hr_spatial

        patches = patches[:,side_pad_hr_x:n, side_pad_hr_spatial:n_spatial, side_pad_hr_spatial:n_spatial]
        
        z_stacks = []
        for k in range(len(patches) // z):
            
            z_start =k*z
            z_stack = np.concatenate(patches[z_start:z_start+z], axis=2)
            z_stacks.append(z_stack)

        y_stacks = []
        for j in range(len(z_stacks) // y):
            y_start =j*y 
            y_stack = np.concatenate(z_stacks[y_start:y_start+y], axis=1)
            y_stacks.append(y_stack)

        end_results = np.concatenate(y_stacks, axis=0)

        # crop based on the padding we did during patchify
        if self.padding[0] > 0:
            end_results = end_results[:-self.padding[0],:, :]
        if self.padding[1] > 0:
            end_results = end_results[:, :-self.padding[1],:]
        if self.padding[2] > 0:
            end_results = end_results[:, :, :-self.padding[2]]

        return end_results     
    
    