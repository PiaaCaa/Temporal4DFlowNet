import random as rnd
import numpy as np
import csv
import h5py
import json

def write_header(filename):
    fieldnames = ['source', 'target','index', 'start_x', 'start_y', 'start_z', 'rotate', 'rotation_plane', 'rotation_degree_idx', 'coverage']
    with open(filename, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

def write_header_temporal(filename):
    fieldnames = ['source', 'target','axis', 'index', 'start_t', 'start_1', 'start_2', 'step_t', 'reverse', 'coverage']
    with open(filename, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

def write_header_temporal_extended_data_augmentation(filename):
    fieldnames = ['source', 'target','axis', 'index', 'start_t', 'start_1', 'start_2', 'step_t', 's_patchsize', 't_patchsize',
                  'flip_1','flip_2','rot', 'sign_u', 'sign_v', 'sign_w', 'swap_u', 'swap_v', 'swap_w', 
                  'coverage']
    with open(filename, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

def save_csv_file_settings_json(input_filename, target_filename, output_filename, n_patch, binary_mask, s_patchsize,t_patchsize, minimum_coverage, empty_patch_allowed, 
                                                                flipping,  all_rotation, swap_velocity_components, change_sign_velocity_components,  step_t ,
                                                                save_nonaugmented_patch, n_patches_augmented_from_original_patch, only_choose_apply_one_augmentation_technique, sign_change_on_all_components):
    #save as json file
    settings = {
        'data': {

            'input_filename': input_filename,
            'target_filename': target_filename,
            'output_filename': output_filename,
        },
        'patch_settings' : {
            'n_patch_per_axis': n_patch,
            's_patchsize': s_patchsize,
            't_patchsize': t_patchsize,
            'step_t': step_t,
            'shape_binary_mask': binary_mask.shape,
            'minimum_coverage': minimum_coverage,
            'empty_patch_allowed': empty_patch_allowed,
        },
        'augmentation_settings' : {
            'flipping': flipping,
            'all_rotation': all_rotation,
            'swap_velocity_components': swap_velocity_components,
            'change_sign_velocity_components': change_sign_velocity_components,
            'save_nonaugmented_patch': save_nonaugmented_patch,
            'n_patches_augmented_from_original_patch': n_patches_augmented_from_original_patch,
            'only_choose_apply_one_augmentation_technique': only_choose_apply_one_augmentation_technique,
            'sign_change_on_all_components': sign_change_on_all_components
        }
    }
    json_filename = output_filename.split('.')[0] + '_settings.json'
    with open(json_filename, 'w') as f:
        json.dump(settings, f, indent=4)
    return 

def create_temporal_mask(mask, n_frames):
    '''
    from static mask create temporal mask of shape (n_frames, h, w, d)
    '''
    assert(len(mask.shape) == 3), " shape: " + str(mask.shape) # shape of mask is assumed to be 3 dimensional
    return np.repeat(np.expand_dims(mask, 0), n_frames, axis=0)
    

def generate_patches(input_filename, target_filename, output_filename, axis, index, n_patch, binary_mask, s_patchsize,t_patchsize, minimum_coverage, empty_patch_allowed, 
                                                                flipping= False,  all_rotation = False, swap_velocity_components = False, change_sign_velocity_components = False,  step_t =2,
                                                                save_nonaugmented_patch= False, n_patches_augmented_from_original_patch =1, only_choose_apply_one_augmentation_technique=True, sign_change_on_all_components=True):
    """
    Generate temporal random patches along a specified axis with more data augmentation parameters.
    """

    # settings:
    # if False, we create for each patch an augmented version of the patch, and apply only to patches with coverage above threshold
    # if true, we create only the original patch
    # apply_augmentation_to_all_patches = False 
    # n_patches_augmented_from_original_patch = 1
    # # if TRUE, only one of the augmentation techniques is applied to the patch, e.g. only flipping, only rotation, etc.
    # only_choose_apply_one_augmentation_technique = False
    # # if TRUE, we apply sign change to all components, if FALSE, we apply sign change to each component individually randomly
    # sign_change_on_all_components = True
    
            
    # foreach row, create n number of patches
    empty_patch_counter = 0
    j = 0
    not_found = 0

    # #take away h path, i.e. only patches of size (t, w, d)
    if axis == 0:   binary_mask=binary_mask[:,index,  :,        :]
    elif axis == 1: binary_mask=binary_mask[:,:,     index,     :]
    elif axis == 2: binary_mask=binary_mask[:,:,      : ,   index]
    
    augmentations = [
    (flipping, 'flipping'),
    (all_rotation, 'all_rotation'),
    (swap_velocity_components, 'swap_velocity_components'),
    (change_sign_velocity_components, 'change_sign_velocity_components')
    ]

    # Apply augmentations and store applied ones along with their names
    augmentations_applied = [(func, name) for func, name in augmentations if func]


    # if len(augmentations_applied) == 0:
    #     print('No augmentation applied, only creating non-augmented patches')

    while j < n_patch:
        
        if not_found > 100:
            print(f"Cannot find enough patches above {minimum_coverage} coverage, please lower the minimum_coverage")
            break

        can_still_take_empty_patch = empty_patch_counter < empty_patch_allowed

        patch = TemporalPatchData(input_filename, target_filename, s_patchsize, t_patchsize)
        
        # default, no augmentation
        patch.create_random_patch(binary_mask, index, axis, step_t=step_t)
        patch.calculate_patch_coverage(binary_mask, minimum_coverage)

        # before saving, do a check
        if patch.coverage < minimum_coverage:
            if can_still_take_empty_patch:
                
                print('Taking this empty one',patch.coverage)
                empty_patch_counter += 1
                
            else:
                # skip empty patch because we already have one
                not_found += 1
                continue
        
        if save_nonaugmented_patch:
            patch.write_to_csv(output_filename)
        j+=1 # count individual patches

        # if TRUE, we create for each patch an augmented version of the patch and apply only to patches with coverage above threshold
        if len(augmentations_applied) > 0 and patch.coverage >= minimum_coverage:
            
            for _ in range(n_patches_augmented_from_original_patch):
                # reset all augmenttaion parameters
                patch.flip_1 = 0
                patch.flip_2 = 0
                patch.rot = 0
                patch.sign_u = 1
                patch.sign_v = 1
                patch.sign_w = 1
                patch.swap_u = 'u'
                patch.swap_v = 'v'
                patch.swap_w = 'w'


                if only_choose_apply_one_augmentation_technique:
                    
                    _, augmentation = rnd.choice(augmentations_applied)
                    print(f'Applying only {augmentation} augmentation technique')

                    if augmentation == 'flipping':
                        
                        patch.flip_1 = np.random.choice([0, 1])
                        patch.flip_2 = np.random.choice([0, 1])
                    
                    elif augmentation == 'all_rotation':
                        # one of the rotations is true
                        rotation_options = [90, 180, 270]
                        selected_rotation = np.random.choice(rotation_options)
                        patch.rot= selected_rotation
                    
                    elif augmentation == 'swap_velocity_components':
                        # choose two of the three components to swap
                        swap = ['u', 'v', 'w']
                        patch.swap_u = rnd.choice(swap)
                        swap.remove(patch.swap_u)
                        patch.swap_v = rnd.choice(swap)
                        swap.remove(patch.swap_v)
                        patch.swap_w = swap[0]
                    
                    elif augmentation == 'change_sign_velocity_components':
                        if sign_change_on_all_components:
                            patch.sign_u = -1
                            patch.sign_v = -1
                            patch.sign_w = -1
                        else:
                            patch.sign_u = np.random.choice([-1, 1])
                            patch.sign_v = np.random.choice([-1, 1])
                            patch.sign_w = np.random.choice([-1, 1])

                else:
                    # apply all augmentation techniques which are true
                    if flipping:
                        patch.flip_1 = np.random.choice([0, 1])
                        patch.flip_2 = np.random.choice([0, 1])
                    if all_rotation:
                        # one of the rotations is true
                        rotation_options = [90, 180, 270]
                        selected_rotation = np.random.choice(rotation_options)
                        patch.rot= selected_rotation
                    if swap_velocity_components:
                        # choose two of the three components to swap
                        swap = ['u', 'v', 'w']
                        patch.swap_u = rnd.choice(swap)
                        swap.remove(patch.swap_u)
                        patch.swap_v = rnd.choice(swap)
                        swap.remove(patch.swap_v)
                        patch.swap_w = swap[0]
                    if change_sign_velocity_components:
                        if sign_change_on_all_components:
                            patch.sign_u = -1
                            patch.sign_v = -1
                            patch.sign_w = -1
                        else:
                            patch.sign_u = np.random.choice([-1, 1])
                            patch.sign_v = np.random.choice([-1, 1])
                            patch.sign_w = np.random.choice([-1, 1])

                patch.write_to_csv(output_filename)


class TemporalPatchData:

    def __init__(self, source_file, target_file, spatial_patch_size, temporal_patch_size):
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.source_file = source_file
        self.target_file = target_file # this fila has same size as source file with added noise
        self.axis = None 
        self.idx = None
        self.start_t = None
        self.start_1 = None
        self.start_2 = None
        self.step_t = 2
        self.coverage = 0
        # Augmentation parameters
        self.flip_1 = 0
        self.flip_2 = 0
        self.rot = 0
        self.sign_u = 1
        self.sign_v = 1
        self.sign_w = 1
        self.swap_u = 'u'
        self.swap_v = 'v'
        self.swap_w = 'w'

    def create_random_patch(self, u, index, axis, step_t=2):
        self.step_t = step_t
        self.axis = axis
        self.idx = index
        self.start_t = rnd.randrange(2, u.shape[0] - self.temporal_patch_size*self.step_t + 1) #TODO delete this later- only to not consider first two frames in cs data
        if self.start_t <2:
            print('START IS BEFORE 2!!')
        self.start_1 = rnd.randrange(0, u.shape[1] - self.spatial_patch_size) 
        self.start_2 = rnd.randrange(0, u.shape[2] - self.spatial_patch_size) 

    def set_patch(self, index, x, y, z):
        self.idx = index
        self.start_t = x
        self.start_1 = y
        self.start_2 = z

    def calculate_patch_coverage(self, binary_mask, minimum_coverage=0.2):
        #important: take every second time step
        patch_region = np.index_exp[self.start_t:self.start_t+self.temporal_patch_size:self.step_t, self.start_1:self.start_1+self.spatial_patch_size, self.start_2:self.start_2+self.spatial_patch_size]
        patch = binary_mask[patch_region]

        self.coverage = np.count_nonzero(patch) / (self.spatial_patch_size ** 2*self.temporal_patch_size)
        self.coverage = np.round(self.coverage * 1000) / 1000 # round the number to 3 decimal digit

    def check_patch_consistency(self):
        # check if swap variables are unique and contain 'u', 'v', 'w'
        list_swap = [self.swap_u, self.swap_v, self.swap_w]
        if len(list_swap) != len(set(list_swap)):
            print(f"Swap variables should be unique {list_swap}")
        assert len(list_swap) == len(set(list_swap)), "Swap variables should be unique "
        assert all(x in list_swap for x in ['u', 'v', 'w']), "Swap variables should contain 'u', 'v', 'w'"

        # check that only one of the rotations is true
        list_rot = [0, 90, 180, 270]
        assert self.rot in list_rot, "Rotation should be 0, 90, 180, or 270"
        return

    def write_to_csv(self, output_filename):
        
        fieldnames = ['source', 'target','axis', 'index', 'start_t', 'start_1', 'start_2', 'step_t', 's_patchsize', 't_patchsize',
                    'flip_1','flip_2','rot',  'sign_u', 'sign_v', 'sign_w', 'swap_u', 'swap_v', 'swap_w', 
                    'coverage']
        
        self.check_patch_consistency()

        with open(output_filename, mode='a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow({'source': self.source_file, 'target': self.target_file,'axis': self.axis, 'index': self.idx, 
            'start_t': self.start_t, 'start_1': self.start_1, 'start_2': self.start_2, 'step_t': self.step_t, 's_patchsize': self.spatial_patch_size, 't_patchsize': self.temporal_patch_size,
            'flip_1': self.flip_1, 'flip_2': self.flip_2, 'rot': self.rot,  'sign_u': self.sign_u, 'sign_v': self.sign_v, 'sign_w': self.sign_w, 
            'swap_u': self.swap_u, 'swap_v': self.swap_v, 'swap_w': self.swap_w, 'coverage': self.coverage})
