import numpy as np
import h5py
import PatchData as pd
import os
import argparse
import yaml
import shutil


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data_shape(input_filepath):
    """Load and print dataset shape from HDF5 file."""
    with h5py.File(input_filepath, mode='r') as hdf5:
        t, x, y, z = hdf5['u'].shape
    print(f"Dataset of size: {t, x, y, z}")
    return t, x, y, z


def determine_step_t(t_lr, t_hr):
    """
    Determine temporal step size based on LR and HR frame counts.
    If equal, LR is downsampled on the fly (step_t=2).
    If HR has 2x frames, LR is already downsampled (step_t=1).
    """
    if t_hr == t_lr:
        print('Same number of frames in LR and HR — downsampling on the fly (step_t=2).')
        return 2, 1
    else:
        print('HR has more frames than LR — LR is already downsampled (step_t=1).')
        return 1, 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare patches for temporal super-resolution training.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--lrdata", type=str, help="Optional override for LR data filename")
    parser.add_argument("--hrdata", type=str, help="Optional override for HR data filename")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ---- Paths ----
    base_path = cfg['base_path']
    lr_file   = args.lrdata if args.lrdata else cfg['lr_file']
    hr_file   = args.hrdata if args.hrdata else cfg['hr_file']
    csv_dir   = f"{base_path}/csv_files"
    os.makedirs(csv_dir, exist_ok=True)
    output_filename = f"{csv_dir}/{cfg['output_filename']}"

    lr_path = f"{base_path}/{lr_file}"
    hr_path = f"{base_path}/{hr_file}"

    # ---- Patch settings ----
    spatial_patch_size  = cfg['spatial_patch_size']
    temporal_patch_size = cfg['temporal_patch_size']
    n_patch             = cfg['n_patch']
    minimum_coverage    = cfg['minimum_coverage']
    n_empty_patch_allowed = cfg['n_empty_patch_allowed']
    mask_threshold      = cfg['mask_threshold']

    # ---- Augmentation settings ----
    save_nonaugmented_patch                  = cfg['save_nonaugmented_patch']
    n_patches_augmented_from_original_patch  = cfg['n_patches_augmented_from_original_patch']
    random_w_orientation                     = cfg.get('random_w_orientation', False)

    # ---- Verify files exist ----
    assert os.path.isfile(lr_path), f"LR file not found: {lr_path}"
    assert os.path.isfile(hr_path), f"HR file not found: {hr_path}"
    print(f"LR file: {lr_path}")
    print(f"HR file: {hr_path}")

    # ---- Load data shapes and masks ----
    with h5py.File(lr_path, mode='r') as hdf5:
        t_lr, x_lr, y_lr, z_lr = hdf5['u'].shape
        mask_lr = np.asarray(hdf5['mask']).squeeze()
        if len(mask_lr.shape) == 3:
            mask_lr = pd.create_temporal_mask(mask_lr, t_lr)

    with h5py.File(hr_path, 'r') as hf:
        t_hr, x_hr, y_hr, z_hr = hf['u'].shape
        mask_hr = np.asarray(hf['mask']).squeeze()

    print(f"LR shape: {t_lr, x_lr, y_lr, z_lr}")
    print(f"HR shape: {t_hr, x_hr, y_hr, z_hr}")

    # ---- Determine step size and validate ----
    step_t, check_t = determine_step_t(t_lr, t_hr)

    assert (x_lr, y_lr, z_lr) == (x_hr, y_hr, z_hr), "Spatial dimensions of LR and HR must match"
    assert np.sum(np.abs(mask_lr - mask_hr[::check_t])) == 0, "LR and HR masks do not match after temporal alignment"

    # ---- Prepare binary mask ----
    binary_mask = (mask_lr >= mask_threshold) * 1
    print(f"Binary mask shape: {binary_mask.shape}")

    # ---- Write CSV header and save settings ----
    pd.write_header(output_filename)
    pd.save_settings_json(
        lr_file, hr_file, output_filename, n_patch, binary_mask,
        spatial_patch_size, temporal_patch_size, minimum_coverage, n_empty_patch_allowed,
        step_t, save_nonaugmented_patch, n_patches_augmented_from_original_patch,
        random_w_orientation
    )

    # ---- Generate patches for all axes ----
    axis_labels = {0: '(t, y, z)', 1: '(t, x, z)', 2: '(t, x, y)'}
    axis_ranges = {0: range(1, x_lr), 1: range(1, y_lr), 2: range(1, z_lr)}

    for axis in [0, 1, 2]:
        print(f"______ Creating patches for {axis_labels[axis]} slices _____________")
        for idx in axis_ranges[axis]:
            pd.generate_patches(
                lr_file, hr_file, output_filename, axis, idx,
                n_patch, binary_mask, spatial_patch_size, temporal_patch_size,
                minimum_coverage, n_empty_patch_allowed,
                step_t=step_t,
                save_nonaugmented_patch=save_nonaugmented_patch,
                n_patches_augmented_from_original_patch=n_patches_augmented_from_original_patch,
                random_w_orientation=random_w_orientation
            )

    # ---- Save this script and config for reproducibility ----
    shutil.copy2(__file__, csv_dir)
    shutil.copy2(args.config, csv_dir)

    print(f"Done. Patches saved to {output_filename}")