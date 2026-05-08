import numpy as np
import os
import csv
import yaml
from Network.PatchHandler import PatchHandler4D_preload
from Network.TrainerController_temporal import TrainerController_temporal


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_indexes(index_file):
    """Load patch index file (csv). Used to load patches based on x,y,z index."""
    return np.genfromtxt(index_file, delimiter=',', skip_header=True, dtype='unicode')


def check_csv_header(index_file, expected_headers):
    """Check whether the csv file contains the expected headers."""
    with open(index_file, 'r') as file:
        header = file.readline().strip().split(',')
    return all(h in header for h in expected_headers)


def write_settings_into_csv_file(filename, name, training_file, validation_file, test_file,
                                  epochs, batch_size, patch_size, low_resblock, high_resblock,
                                  upsampling_type, low_block_type, high_block_type,
                                  post_block_type, sampling, notes):
    """Write training settings into overview csv file."""
    print(f"Write settings into overview file {filename}")
    fieldnames = [
        "Name", "training_file", "validation_file", "test_file", "epochs",
        "batch_size", "patch_size", "res_increase", "low_resblock", "high_resblock",
        "upsampling_type", "low_block_type", "high_block_type", "post_block_type",
        "sampling", "notes"
    ]
    with open(filename, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow({
            'Name': name,
            "training_file": training_file,
            "validation_file": validation_file,
            "test_file": test_file,
            "epochs": epochs,
            "batch_size": batch_size,
            "patch_size": patch_size,
            "res_increase": res_increase,
            "low_resblock": low_resblock,
            "high_resblock": high_resblock,
            "upsampling_type": upsampling_type,
            'low_block_type': low_block_type,
            'high_block_type': high_block_type,
            'post_block_type': post_block_type,
            'sampling': sampling,
            "notes": notes
        })


if __name__ == "__main__":
    config_path = '/proj/multipress/users/x_piaca/Temporal4DFlowNet/configs/train.yaml'
    cfg = load_config(config_path)

    # ---- Paths ----
    data_dir   = cfg['data_dir']
    csv_dir    = f"{data_dir}/csv_files"
    training_file  = f"{csv_dir}/{cfg['training_file']}"
    validate_file  = f"{csv_dir}/{cfg['validate_file']}"
    benchmark_file = f"{csv_dir}/{cfg['benchmark_file']}"
    overview_csv   = cfg['overview_csv']

    # ---- Hyperparameters ----
    initial_learning_rate = cfg['initial_learning_rate']
    epochs        = cfg['epochs']
    batch_size    = cfg['batch_size']
    mask_threshold = cfg['mask_threshold']
    lr_decay_epochs = cfg['lr_decay_epochs']

    # ---- Network settings ----
    network_name        = cfg['network_name']
    patch_size_tuple    = tuple(cfg['patch_size'])
    res_increase        = cfg['res_increase']
    n_low_resblock      = cfg['n_low_resblock']
    n_hi_resblock       = cfg['n_hi_resblock']
    low_res_block       = cfg['low_res_block']
    high_res_block      = cfg['high_res_block']
    upsampling_block    = cfg['upsampling_block']
    post_processing_block = cfg['post_processing_block']
    include_mag_input   = cfg['include_mag_input']

    # ---- Flags ----
    QUICKSAVE            = cfg['quicksave']
    restore              = cfg['restore']
    load_patches_all_axis = cfg['load_patches_all_axis']
    shuffle              = cfg['shuffle']
    sampling             = cfg.get('sampling', '-')
    notes                = cfg['notes']

    # --- Loss params (with defaults clearly visible) ---
    loss_params = {
        'alpha':              cfg.get('alpha', 0.8),
        'epsilon':            cfg.get('epsilon', 1),
        'weighting_fluid':    cfg.get('weighting_fluid', 1.0),
        'weighting_non_fluid':cfg.get('weighting_non_fluid', 1.0),
        'separate_mse':       cfg.get('separate_mse', True),
        'loss_type':          cfg.get('loss_type', 'l1_projected'),
    }

    # --- Regularization ---
    training_params = {
        'L2_regularization':  cfg.get('L2_regularization', 0.001),
        'lr_decay_epochs':    cfg.get('lr_decay_epochs', 0),
        'initial_learning_rate': cfg.get('learning_rate', 1e-4),
    }

    # ---- Verify files exist ----
    print('Checking that all files exist:')
    for f in [training_file, validate_file, benchmark_file, overview_csv]:
        print(f'  {f}: {os.path.isfile(f)}')

    # ---- Check for extended data augmentation ----
    aug_headers = ['s_patchsize', 't_patchsize', 'flip_1', 'flip_2', 'rot',
                   'sign_u', 'sign_v', 'sign_w', 'swap_u', 'swap_v', 'swap_w']
    extended_data_augmentation = check_csv_header(training_file, aug_headers)
    if extended_data_augmentation:
        print('Data augmentation parameters found in csv file')

    # ---- Check for extended data augmentation headers ----
    aug_headers = ['s_patchsize', 't_patchsize', 'flip_1', 'flip_2', 'rot',
                'sign_u', 'sign_v', 'sign_w', 'swap_u', 'swap_v', 'swap_w']

    if not check_csv_header(training_file, aug_headers):
        raise ValueError(f"Training file {training_file} is missing required augmentation headers: {aug_headers}")

    print('Augmentation headers verified.')

    # ---- Build datasets ----
    trainset = PatchHandler4D_preload(
        data_dir, patch_size_tuple, res_increase, batch_size, mask_threshold, csv_file=training_file
    ).initialize_dataset(load_indexes(training_file), shuffle=shuffle)

    valset = PatchHandler4D_preload(
        data_dir, patch_size_tuple, res_increase, batch_size, mask_threshold, csv_file=validate_file
    ).initialize_dataset(load_indexes(validate_file), shuffle=shuffle)

    # ---- Optional benchmark set ----
    testset = None
    if QUICKSAVE and benchmark_file is not None:
        bench_handler = PatchHandler4D_preload(
                        data_dir, patch_size_tuple, res_increase, batch_size, mask_threshold, csv_file=benchmark_file
                    ).initialize_dataset(load_indexes(benchmark_file), shuffle=shuffle)
        testset = bench_handler.initialize_dataset(load_indexes(benchmark_file), shuffle=False)

    # ---- Initialize network ----
    print(f"4DFlowNet | Patch {patch_size_tuple} | LR {initial_learning_rate} | Batch {batch_size}")
    network = TrainerController_temporal(
        patch_size_tuple, res_increase, initial_learning_rate, QUICKSAVE,
        network_name, n_low_resblock, n_hi_resblock, low_res_block, high_res_block,
        **loss_params,
        **training_params,
        upsampling_block=upsampling_block, post_processing_block=post_processing_block,
        lr_decay_epochs=lr_decay_epochs, include_mag_input=include_mag_input
    )
    network.init_model_dir(config_path=config_path)

    if restore:
        print(f"Restoring model {cfg['model_file']}...")
        network.restore_model(cfg['model_dir'], cfg['model_file'])

    # ---- Log settings ----
    write_settings_into_csv_file(
        overview_csv, network.unique_model_name,
        os.path.basename(training_file), os.path.basename(validate_file),
        os.path.basename(benchmark_file), epochs, batch_size, patch_size_tuple,
        n_low_resblock, n_hi_resblock, upsampling_block, low_res_block,
        high_res_block, post_processing_block, sampling, notes
    )

    # ---- Train ----
    network.train_network(trainset, valset, n_epoch=epochs, testset=testset)