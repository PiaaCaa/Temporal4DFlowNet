import numpy as np
import os
import argparse
import csv
import yaml
from Network.PatchHandler import PatchHandler4D_preload, PatchHandler4D
from Network.TrainerController import TrainerController


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
                                  epochs, batch_size, patch_size, network_params, loss_params,
                                  training_params, sampling, notes):
    """Write training settings into overview csv file."""
    print(f"Write settings into overview file {filename}")
    fieldnames = [
        "Name", "training_file", "validation_file", "test_file",
        "epochs", "batch_size", "patch_size",
        # network
        "res_increase", "n_low_resblock", "n_hi_resblock",
        "low_res_block", "high_res_block", "upsampling_block", "post_processing_block", "include_mag_input",
        # loss
        "loss_type", "use_directional_loss", "alpha", "epsilon",
        "weighting_fluid", "weighting_non_fluid", "separate_mse",
        # training
        "initial_learning_rate", "lr_decay_epochs", "L2_regularization",
        # misc
        "sampling", "notes"
    ]
    row = {
        'Name': name,
        'training_file': training_file,
        'validation_file': validation_file,
        'test_file': test_file,
        'epochs': epochs,
        'batch_size': batch_size,
        'patch_size': patch_size,
        'sampling': sampling,
        'notes': notes,
        **network_params,
        **loss_params,
        **training_params,
    }
    with open(filename, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    config_path = args.config

    # ---- Paths ----
    data_dir       = cfg['data_dir']
    csv_dir        = f"{data_dir}/csv_files"
    training_file  = f"{csv_dir}/{cfg['training_file']}"
    validate_file  = f"{csv_dir}/{cfg['validate_file']}"
    benchmark_file = f"{csv_dir}/{cfg['benchmark_file']}"
    overview_csv   = cfg['overview_csv']

    # ---- Flags ----
    preload_data = cfg.get('preload_data', False)
    QUICKSAVE    = cfg.get('quicksave', True)
    restore      = cfg.get('restore', False)
    shuffle      = cfg.get('shuffle', True)
    sampling     = cfg.get('sampling', '-')
    notes        = cfg.get('notes', '')
    epochs       = cfg['epochs']
    batch_size   = cfg['batch_size']
    mask_threshold = cfg.get('mask_threshold', 0.6)
    patch_size_tuple = tuple(cfg['patch_size'])

    # ---- Network params (with defaults) ----
    network_params = {
        'res_increase':         cfg.get('res_increase', 2),
        'n_low_resblock':       cfg.get('n_low_resblock', 8),
        'n_hi_resblock':        cfg.get('n_hi_resblock', 4),
        'low_res_block':        cfg.get('low_res_block', 'resnet_block'),
        'high_res_block':       cfg.get('high_res_block', 'resnet_block'),
        'upsampling_block':     cfg.get('upsampling_block', 'default'),
        'post_processing_block':cfg.get('post_processing_block', None),
        'include_mag_input':    cfg.get('include_mag_input', True),
    }

    # ---- Loss params (with defaults) ----
    loss_params = {
        'loss_type':            cfg.get('loss_type', 'mse'),
        'use_directional_loss': cfg.get('use_directional_loss', True),
        'alpha':                cfg.get('alpha', 0.8),
        'epsilon':              cfg.get('epsilon', 1),
        'weighting_fluid':      cfg.get('weighting_fluid', 1.0),
        'weighting_non_fluid':  cfg.get('weighting_non_fluid', 1.0),
        'separate_mse':         cfg.get('separate_mse', True),
    }

    # ---- Training params (with defaults) ----
    training_params = {
        'initial_learning_rate': cfg.get('initial_learning_rate', 1e-4),
        'lr_decay_epochs':       cfg.get('lr_decay_epochs', 0),
        'L2_regularization':     cfg.get('L2_regularization', 0.001),
    }

    # ---- Verify files exist ----
    print('Checking that all files exist:')
    for f in [training_file, validate_file, benchmark_file, overview_csv]:
        print(f'  {f}: {os.path.isfile(f)}')

    # ---- Check for required augmentation headers ----
    aug_headers = ['s_patchsize', 't_patchsize', 'flip_1', 'flip_2', 'rot',
                   'sign_u', 'sign_v', 'sign_w', 'swap_u', 'swap_v', 'swap_w']
    if not check_csv_header(training_file, aug_headers):
        raise ValueError(f"Training file {training_file} is missing required augmentation headers: {aug_headers}")
    print('Augmentation headers verified.')

    # ---- Select patch handler ----
    PatchHandler = PatchHandler4D_preload if preload_data else PatchHandler4D

    # ---- Build datasets ----
    trainset = PatchHandler(
        data_dir, patch_size_tuple, network_params['res_increase'], batch_size, mask_threshold, csv_file=training_file
    ).initialize_dataset(load_indexes(training_file), shuffle=shuffle)

    valset = PatchHandler(
        data_dir, patch_size_tuple, network_params['res_increase'], batch_size, mask_threshold, csv_file=validate_file
    ).initialize_dataset(load_indexes(validate_file), shuffle=False)

    testset = None
    if QUICKSAVE and benchmark_file is not None:
        testset = PatchHandler(
            data_dir, patch_size_tuple, network_params['res_increase'], batch_size, mask_threshold, csv_file=benchmark_file
        ).initialize_dataset(load_indexes(benchmark_file), shuffle=False)

    # ---- Initialize network ----
    print(f"4DFlowNet | Patch {patch_size_tuple} | LR {training_params['initial_learning_rate']} | Batch {batch_size}")
    network = TrainerController(
        patch_size=patch_size_tuple,
        quicksave_enable=QUICKSAVE,
        network_name=cfg.get('network_name', '4DFlowNet'),
        **network_params,
        **loss_params,
        **training_params,
    )
    network.init_model_dir(config_path=config_path)

    if restore:
        print(f"Restoring model {cfg['model_file']}...")
        network.restore_model(cfg['model_dir'], cfg['model_file'])

    # ---- Log settings to overview CSV ----
    write_settings_into_csv_file(
        overview_csv, network.unique_model_name,
        os.path.basename(training_file), os.path.basename(validate_file),
        os.path.basename(benchmark_file), epochs, batch_size, patch_size_tuple,
        network_params, loss_params, training_params, sampling, notes
    )

    # ---- Train ----
    network.train_network(trainset, valset, n_epoch=epochs, testset=testset)