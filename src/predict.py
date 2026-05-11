"""
Prediction script for Temporal4DFlowNet.

Runs prediction for any input file (insilico, paired invivo, or invivo).
All inputs and output filenames are defined in the config YAML.
Network params are loaded from model_config if specified in the config YAML,
otherwise defaults are used — ensure these match the trained model.

downsample_input_first:
    False - upsample temporally (invivo, insilico with already downsampled LR)
    True  - downsample first then predict (paired invivo for direct comparison)
"""

import tensorflow as tf
import numpy as np
import time
import os
import h5py
import argparse
import yaml
from Network.Temporal4DFlowNetModel import T4DFlowNet
from Network.PatchGenerator import PatchGenerator
from utils import prediction_utils
from utils.ImageDataset import ImageDataset


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_network(patch_size, res_increase, n_low_resblock, n_hi_resblock,
                    low_res_block, high_res_block, upsampling_block,
                    post_processing_block, include_mag_input):
    """Build and return the Temporal4DFlowNet model."""
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size, patch_size)
    elif not isinstance(patch_size, tuple):
        raise ValueError("patch_size must be an int or a tuple of 3 ints")

    input_shape = (*patch_size, 1)
    u     = tf.keras.layers.Input(shape=input_shape, name='u')
    v     = tf.keras.layers.Input(shape=input_shape, name='v')
    w     = tf.keras.layers.Input(shape=input_shape, name='w')
    u_mag = tf.keras.layers.Input(shape=input_shape, name='u_mag')
    v_mag = tf.keras.layers.Input(shape=input_shape, name='v_mag')
    w_mag = tf.keras.layers.Input(shape=input_shape, name='w_mag')

    input_layer = [u, v, w, u_mag, v_mag, w_mag] if include_mag_input else [u, v, w]

    net = T4DFlowNet(res_increase, low_res_block=low_res_block, high_res_block=high_res_block,
                       upsampling_block=upsampling_block, post_processing_block=post_processing_block)
    prediction = net.build_network(u, v, w, u_mag, v_mag, w_mag,
                                   n_low_resblock, n_hi_resblock, include_mag=include_mag_input)
    return tf.keras.Model(input_layer, prediction)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Temporal4DFlowNet prediction script")
    parser.add_argument("--config", type=str, required=True, help="Path to prediction config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ---- Paths ----
    model_name   = cfg['model_name']
    data_dir     = cfg['data_dir']
    output_root  = cfg['output_root']
    model_path = cfg.get('model_path', f'Temporal4DFlowNet/models/Temporal4DFlowNet_{model_name}/Temporal4DFlowNet-best.h5')

    # ---- Prediction params ----
    res_increase          = cfg.get('res_increase', 2)
    batch_size            = cfg.get('batch_size', 64)
    downsample_input_first = cfg.get('downsample_input_first', False)
    round_small_values    = cfg.get('round_small_values', False)

    # ---- Network params: load from config if available, otherwise use hardcoded defaults ----
    # These must match the architecture the model was trained with.
    # If a model_config path is provided in the config, it will be loaded from there.

    model_config_path = cfg.get('model_config', None)
    if model_config_path is not None:
        if not os.path.exists(model_config_path):
            raise ValueError(f"model_config specified but not found: {model_config_path}")
        model_cfg = load_config(model_config_path)
        print(f"Loaded network params from model config: {model_config_path}")
    else:
        model_cfg = cfg  # fall back to current config or hardcoded defaults
        print("Warning: no model_config provided — using hardcoded network defaults.")

    patch_size_tuple      = tuple(model_cfg.get('patch_size', [16, 16, 16]))
    include_mag_input     = model_cfg.get('include_mag_input', False)   
    n_low_resblock        = model_cfg.get('n_low_resblock', 8)
    n_hi_resblock         = model_cfg.get('n_hi_resblock', 4)
    low_res_block         = model_cfg.get('low_res_block', 'resnet_block')
    high_res_block        = model_cfg.get('high_res_block', 'resnet_block')
    upsampling_block      = model_cfg.get('upsampling_block', 'linear')
    post_processing_block = model_cfg.get('post_processing_block', None)

    assert os.path.exists(model_path), f"Model file does not exist: {model_path}"

    # ---- Load network once ----
    network = prepare_network(patch_size_tuple, res_increase, n_low_resblock, n_hi_resblock,
                              low_res_block, high_res_block, upsampling_block,
                              post_processing_block, include_mag_input)
    network.load_weights(model_path)
    print(f"Model loaded from {model_path}")

    # ---- Loop over inputs ----
    for entry in cfg['inputs']:
        filename        = entry['filename']
        output_filename = entry['output_filename']

        input_filepath  = f"{data_dir}/{filename}"
        output_dir      = f"{output_root}/Temporal4DFlowNet_{model_name}"
        output_filepath = f"{output_dir}/{output_filename}"

        print(f"\n{'='*60}")
        print(f"Input:  {input_filepath}")
        print(f"Output: {output_filepath}")

        if os.path.exists(output_filepath):
            print("Output file already exists. Skipping.")
            continue

        assert os.path.exists(input_filepath), f"Input file does not exist: {input_filepath}"
        os.makedirs(output_dir, exist_ok=True)

        t0 = time.time()

        pgen    = PatchGenerator(patch_size_tuple, res_increase,
                                 include_all_axis=True,
                                 downsample_input_first=downsample_input_first)
        dataset = ImageDataset(venc_colnames=['u_max', 'v_max', 'w_max'])

        with h5py.File(input_filepath, mode='r') as h5:
            lr_shape = np.asarray(h5.get("u")).squeeze().shape
            print(f"Input shape: {lr_shape}")
            N_frames, X, Y, Z = lr_shape

        upsampling_factor = 1 if downsample_input_first else res_increase
        u_combined = np.zeros((upsampling_factor * N_frames, X, Y, Z))
        v_combined = np.zeros((upsampling_factor * N_frames, X, Y, Z))
        w_combined = np.zeros((upsampling_factor * N_frames, X, Y, Z))

        axis = [0, 1, 2]

        # ---- Loop over all axes ----
        for a in axis:
            print(f"\n____ Predicting axis {a} ____")
            nr_rows = dataset.get_dataset_len(input_filepath, a)
            print(f"Number of slices: {nr_rows}")

            volume = np.zeros((3, u_combined.shape[0], u_combined.shape[1],
                               u_combined.shape[2], u_combined.shape[3]))

            for nrow in range(nr_rows):
                print(f"\nSlice ({nrow+1}/{nr_rows}) - {time.ctime()}")

                dataset.load_vectorfield(input_filepath, nrow, axis=a)

                velocities, magnitudes = pgen.patchify(dataset)
                data_size = len(velocities[0])

                results = np.zeros((0, patch_size_tuple[0] * res_increase,
                                    patch_size_tuple[1], patch_size_tuple[2], 3))
                start_time = time.time()

                for current_idx in range(0, data_size, batch_size):
                    print(f"\rBatch {current_idx}/{data_size} - {time.time()-start_time:.1f}s", end='\r')
                    patch_index = np.index_exp[current_idx:current_idx + batch_size]

                    if include_mag_input:
                        sr_images = network.predict([velocities[0][patch_index],
                                                     velocities[1][patch_index],
                                                     velocities[2][patch_index],
                                                     magnitudes[0][patch_index],
                                                     magnitudes[1][patch_index],
                                                     magnitudes[2][patch_index]])
                    else:
                        sr_images = network.predict([velocities[0][patch_index],
                                                     velocities[1][patch_index],
                                                     velocities[2][patch_index]])

                    results = np.append(results, sr_images, axis=0)

                print(f"\rDone. {data_size}/{data_size} - {time.time()-start_time:.1f}s")

                for i in range(3):
                    vel = pgen._patchup_with_overlap(results[:, :, :, :, i],
                                                   pgen.nr_x, pgen.nr_y, pgen.nr_z)
                    vel = vel * dataset.venc

                    if round_small_values:
                        vel[np.abs(vel) < dataset.velocity_per_px] = 0

                    if vel.shape[0] != u_combined.shape[0]:
                        print(f"Warning: Shape mismatch — expected {u_combined.shape[0]} frames, got {vel.shape[0]}. Padding/truncating.")
                        if vel.shape[0] < u_combined.shape[0]:
                            vel = np.pad(vel, ((0, u_combined.shape[0]-vel.shape[0]), (0,0), (0,0)))
                        else:
                            vel = vel[:u_combined.shape[0], :, :]

                    if a == 0:   volume[i, :, nrow, :,    :] = vel
                    elif a == 1: volume[i, :, :,    nrow, :] = vel
                    elif a == 2: volume[i, :, :,    :,    nrow] = vel

            u_combined += volume[0]
            v_combined += volume[1]
            w_combined += volume[2]

        # ---- Save results ----
        print(f"\nSaving results to {output_filepath}")
        n_axis = len(axis)
        prediction_utils.save_to_h5(output_filepath, "u_combined", u_combined / n_axis, compression='gzip')
        prediction_utils.save_to_h5(output_filepath, "v_combined", v_combined / n_axis, compression='gzip')
        prediction_utils.save_to_h5(output_filepath, "w_combined", w_combined / n_axis, compression='gzip')
        prediction_utils.save_to_h5(output_filepath, "input_filepath", np.array([input_filepath], dtype='S'))

        if dataset.venc is not None:
            prediction_utils.save_to_h5(output_filepath, "venc",
                                         np.array(dataset.venc, dtype='float32'))

        patch_array = np.array(patch_size_tuple)
        prediction_utils.save_to_h5(output_filepath, "patch_size", patch_array)

        print(f"Total time: {time.time()-t0:.1f}s")
        print(f"Done: {output_filename}")

    print("\nAll predictions complete.")
