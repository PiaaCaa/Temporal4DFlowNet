import random as rnd
import numpy as np
import csv
import json
import os


# Physics consistent augmentations:
# Each entry defines the geometric transform and the corresponding
# velocity sign/swap changes needed to maintain physical consistency.
# Flipping along axis 1 (start_1) requires sign_u=-1 to stay divergence-free.
# Flipping along axis 2 (start_2) requires sign_v=-1.
# Rotations require swapping u/v and adjusting signs accordingly.
PHYSICS_CONSISTENT_AUGMENTATIONS = {
    'flip_1':  {'flip_1': 1, 'flip_2': 0, 'rot': 0,   'sign_u': -1, 'sign_v':  1, 'sign_w': 1, 'swap_u': 'u', 'swap_v': 'v', 'swap_w': 'w'},
    'flip_2':  {'flip_1': 0, 'flip_2': 1, 'rot': 0,   'sign_u':  1, 'sign_v': -1, 'sign_w': 1, 'swap_u': 'u', 'swap_v': 'v', 'swap_w': 'w'},
    'flip_12': {'flip_1': 1, 'flip_2': 1, 'rot': 0,   'sign_u': -1, 'sign_v': -1, 'sign_w': 1, 'swap_u': 'u', 'swap_v': 'v', 'swap_w': 'w'},
    'rot90':   {'flip_1': 0, 'flip_2': 0, 'rot': 90,  'sign_u':  1, 'sign_v': -1, 'sign_w': 1, 'swap_u': 'v', 'swap_v': 'u', 'swap_w': 'w'},
    'rot180':  {'flip_1': 0, 'flip_2': 0, 'rot': 180, 'sign_u':  1, 'sign_v':  1, 'sign_w': 1, 'swap_u': 'u', 'swap_v': 'v', 'swap_w': 'w'},
    'rot270':  {'flip_1': 0, 'flip_2': 0, 'rot': 270, 'sign_u': -1, 'sign_v':  1, 'sign_w': 1, 'swap_u': 'v', 'swap_v': 'u', 'swap_w': 'w'},
}


def write_header(filename):
    """Write CSV header for physics-consistent augmentation patches."""
    fieldnames = ['source', 'target','axis', 'index', 'start_t', 'start_1', 'start_2', 'step_t', 's_patchsize', 't_patchsize',
                'flip_1','flip_2','rot', 'sign_u', 'sign_v', 'sign_w', 'swap_u', 'swap_v', 'swap_w', 
                'coverage']
    with open(filename, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()


def save_settings_json(input_filename, target_filename, output_filename, n_patch, binary_mask,
                       s_patchsize, t_patchsize, minimum_coverage, empty_patch_allowed,
                       step_t, save_nonaugmented_patch, n_patches_augmented_from_original_patch,
                       random_w_orientation):
    """Save patch generation settings to a JSON file alongside the CSV."""
    settings = {
        'data': {
            'input_filename': input_filename,
            'target_filename': target_filename,
            'output_filename': output_filename,
        },
        'patch_settings': {
            'n_patch_per_axis': n_patch,
            's_patchsize': s_patchsize,
            't_patchsize': t_patchsize,
            'step_t': step_t,
            'shape_binary_mask': binary_mask.shape,
            'minimum_coverage': minimum_coverage,
            'empty_patch_allowed': empty_patch_allowed,
        },
        'augmentation_settings': {
            'physics_consistent': True,
            'augmentations': list(PHYSICS_CONSISTENT_AUGMENTATIONS.keys()),
            'save_nonaugmented_patch': save_nonaugmented_patch,
            'n_patches_augmented_from_original_patch': n_patches_augmented_from_original_patch,
            'random_w_orientation': random_w_orientation,
        }
    }
    json_filename = os.path.splitext(output_filename)[0] + '_settings.json'
    with open(json_filename, 'w') as f:
        json.dump(settings, f, indent=4, default=str)


def create_temporal_mask(mask, n_frames):
    """Create temporal mask of shape (n_frames, h, w, d) from a static 3D mask."""
    assert len(mask.shape) == 3, f"Expected 3D mask, got shape: {mask.shape}"
    return np.repeat(np.expand_dims(mask, 0), n_frames, axis=0)


def generate_patches(input_filename, target_filename, output_filename, axis, index,
                     n_patch, binary_mask, s_patchsize, t_patchsize, minimum_coverage,
                     empty_patch_allowed, step_t=2, save_nonaugmented_patch=True,
                     n_patches_augmented_from_original_patch=4, random_w_orientation=False):
    """
    Generate temporal random patches along a specified axis with physics-consistent augmentation.

    Augmentations are always physics-consistent — each geometric transform is paired
    with the correct velocity sign/swap to maintain divergence-free flow.
    Augmentations are sampled without replacement per patch.

    Args:
        input_filename:                         LR data filename
        target_filename:                        HR data filename
        output_filename:                        Output CSV filename
        axis:                                   Axis to slice along (0, 1, or 2)
        index:                                  Index along the axis
        n_patch:                                Number of patches to generate
        binary_mask:                            Binary mask (t, x, y, z)
        s_patchsize:                            Spatial patch size
        t_patchsize:                            Temporal patch size
        minimum_coverage:                       Minimum fluid coverage required [0-1]
        empty_patch_allowed:                    Max number of low-coverage patches allowed
        step_t:                                 Temporal step size
        save_nonaugmented_patch:                Save the original patch before augmentation
        n_patches_augmented_from_original_patch: Number of augmented versions per patch
        random_w_orientation:                   Randomly flip sign_w independently
    """
    # Slice the mask along the chosen axis
    if axis == 0:   binary_mask = binary_mask[:, index, :, :]
    elif axis == 1: binary_mask = binary_mask[:, :, index, :]
    elif axis == 2: binary_mask = binary_mask[:, :, :, index]

    empty_patch_counter = 0
    j = 0
    not_found = 0
    rows = []

    while j < n_patch:
        if not_found > 100:
            print(f"Cannot find enough patches above {minimum_coverage} coverage, please lower minimum_coverage")
            break

        patch = TemporalPatchData(input_filename, target_filename, s_patchsize, t_patchsize)
        patch.create_random_patch(binary_mask, index, axis, step_t=step_t)
        patch.calculate_patch_coverage(binary_mask, minimum_coverage)

        if patch.coverage < minimum_coverage:
            if empty_patch_counter < empty_patch_allowed:
                print(f'Taking low-coverage patch: {patch.coverage}')
                empty_patch_counter += 1
            else:
                not_found += 1
                continue

        if save_nonaugmented_patch:
            rows.append(patch.to_dict())
        j += 1

        # Sample augmentations without replacement so each type used at most once per patch
        if patch.coverage >= minimum_coverage:
            n_to_apply = min(n_patches_augmented_from_original_patch, len(PHYSICS_CONSISTENT_AUGMENTATIONS))
            selected = rnd.sample(list(PHYSICS_CONSISTENT_AUGMENTATIONS.keys()), n_to_apply)
            for augmentation_name in selected:
                patch.reset_augmentation()
                patch.apply_physics_consistent_augmentation(augmentation_name, random_w_orientation)
                patch.check_consistency()
                rows.append(patch.to_dict())

    # Write all rows at once
    fieldnames = ['source', 'target', 'axis', 'index', 'start_t', 'start_1', 'start_2', 'step_t',
                  's_patchsize', 't_patchsize', 'flip_1', 'flip_2', 'rot',
                  'sign_u', 'sign_v', 'sign_w', 'swap_u', 'swap_v', 'swap_w', 'coverage']
    with open(output_filename, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerows(rows)


class TemporalPatchData:
    """Patch data class for temporal super-resolution with physics-consistent data augmentation."""

    def __init__(self, source_file, target_file, spatial_patch_size, temporal_patch_size):
        self.spatial_patch_size  = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.source_file = source_file
        self.target_file = target_file
        self.axis    = None
        self.idx     = None
        self.start_t = None
        self.start_1 = None
        self.start_2 = None
        self.step_t  = 2
        self.coverage = 0
        self.reset_augmentation()

    def reset_augmentation(self):
        """Reset all augmentation parameters to their defaults (no augmentation)."""
        self.flip_1 = 0
        self.flip_2 = 0
        self.rot    = 0
        self.sign_u = 1
        self.sign_v = 1
        self.sign_w = 1
        self.swap_u = 'u'
        self.swap_v = 'v'
        self.swap_w = 'w'

    def create_random_patch(self, u, index, axis, step_t=2):
        self.step_t  = step_t
        self.axis    = axis
        self.idx     = index
        self.start_t = rnd.randrange(2, u.shape[0] - self.temporal_patch_size * self.step_t + 1)
        self.start_1 = rnd.randrange(0, u.shape[1] - self.spatial_patch_size)
        self.start_2 = rnd.randrange(0, u.shape[2] - self.spatial_patch_size)

    def calculate_patch_coverage(self, binary_mask, minimum_coverage=0.2):
        patch_region = np.index_exp[
            self.start_t:self.start_t + self.temporal_patch_size:self.step_t,
            self.start_1:self.start_1 + self.spatial_patch_size,
            self.start_2:self.start_2 + self.spatial_patch_size
        ]
        patch = binary_mask[patch_region]
        n_voxels = self.spatial_patch_size ** 2 * self.temporal_patch_size
        self.coverage = round(np.count_nonzero(patch) / n_voxels, 3)

    def apply_physics_consistent_augmentation(self, augmentation_name, random_w_orientation=False):
        """
        Apply a named physics-consistent augmentation.
        Each augmentation pairs geometric transforms with the correct velocity
        sign/swap changes to maintain physical consistency (divergence-free flow).
        """
        assert augmentation_name in PHYSICS_CONSISTENT_AUGMENTATIONS, \
            f"Unknown augmentation: {augmentation_name}. Choose from {list(PHYSICS_CONSISTENT_AUGMENTATIONS.keys())}"

        params = PHYSICS_CONSISTENT_AUGMENTATIONS[augmentation_name]
        self.flip_1 = params['flip_1']
        self.flip_2 = params['flip_2']
        self.rot    = params['rot']
        self.sign_u = params['sign_u']
        self.sign_v = params['sign_v']
        self.sign_w = params['sign_w']
        self.swap_u = params['swap_u']
        self.swap_v = params['swap_v']
        self.swap_w = params['swap_w']

        # Optionally randomize sign_w independently (w is the through-plane component)
        if random_w_orientation:
            self.sign_w = np.random.choice([-1, 1])

    def check_consistency(self):
        """Validate augmentation parameters."""
        swap_list = [self.swap_u, self.swap_v, self.swap_w]
        assert len(swap_list) == len(set(swap_list)), f"Swap variables must be unique: {swap_list}"
        assert all(x in swap_list for x in ['u', 'v', 'w']), "Swap variables must contain 'u', 'v', 'w'"
        assert self.rot in [0, 90, 180, 270], f"Rotation must be 0, 90, 180, or 270, got {self.rot}"

    def to_dict(self):
        """Return patch data as a dictionary for CSV writing."""
        return {
            'source': self.source_file, 'target': self.target_file,
            'axis': self.axis, 'index': self.idx,
            'start_t': self.start_t, 'start_1': self.start_1, 'start_2': self.start_2,
            'step_t': self.step_t, 's_patchsize': self.spatial_patch_size,
            't_patchsize': self.temporal_patch_size,
            'flip_1': self.flip_1, 'flip_2': self.flip_2, 'rot': self.rot,
            'sign_u': self.sign_u, 'sign_v': self.sign_v, 'sign_w': self.sign_w,
            'swap_u': self.swap_u, 'swap_v': self.swap_v, 'swap_w': self.swap_w,
            'coverage': self.coverage
        }