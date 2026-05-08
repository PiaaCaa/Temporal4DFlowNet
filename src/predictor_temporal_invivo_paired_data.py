import tensorflow as tf
import numpy as np
import time
import os
from Network.STR4DFlowNet_adapted import STR4DFlowNet
from Network.PatchGenerator import PatchGenerator
from utils import prediction_utils
from utils.ImageDataset_temporal import ImageDataset_temporal
from matplotlib import pyplot as plt
import h5py
import timeit
import argparse
from scipy.interpolate import CubicSpline, RegularGridInterpolator
# os.environ["CUDA_VISIBLE_DEVICES"]="1"


# def prepare_temporal_network(patch_size, res_increase, n_low_resblock, n_hi_resblock, low_res_block, high_res_block, upsampling_block):
#     # Prepare input
#     input_shape = (patch_size,patch_size,patch_size,1)
#     u = tf.keras.layers.Input(shape=input_shape, name='u')
#     v = tf.keras.layers.Input(shape=input_shape, name='v')
#     w = tf.keras.layers.Input(shape=input_shape, name='w')

#     u_mag = tf.keras.layers.Input(shape=input_shape, name='u_mag')
#     v_mag = tf.keras.layers.Input(shape=input_shape, name='v_mag')
#     w_mag = tf.keras.layers.Input(shape=input_shape, name='w_mag')

#     input_layer = [u,v,w,u_mag, v_mag, w_mag]

#     # network & output
#     net = STR4DFlowNet(res_increase,low_res_block=low_res_block, high_res_block=high_res_block,  upsampling_block=upsampling_block )
#     prediction = net.build_network(u, v, w, u_mag, v_mag, w_mag, n_low_resblock, n_hi_resblock)
#     model = tf.keras.Model(input_layer, prediction)

#     return model

def prepare_temporal_network(patch_size, res_increase, n_low_resblock, n_hi_resblock, low_res_block, high_res_block, upsampling_block, post_processing_block, include_mag_input=True):
    # Prepare input
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size, patch_size)
    elif isinstance(patch_size, tuple):
        patch_size = patch_size
    else:
        raise ValueError("patch_size must be an int or a tuple of 3 ints")
    input_shape = (*patch_size,1)
    u = tf.keras.layers.Input(shape=input_shape, name='u')
    v = tf.keras.layers.Input(shape=input_shape, name='v')
    w = tf.keras.layers.Input(shape=input_shape, name='w')
   
    u_mag = tf.keras.layers.Input(shape=input_shape, name='u_mag')
    v_mag = tf.keras.layers.Input(shape=input_shape, name='v_mag')
    w_mag = tf.keras.layers.Input(shape=input_shape, name='w_mag')

    if include_mag_input:
        input_layer = [u,v,w,u_mag, v_mag, w_mag]
    else:
        input_layer = [u,v,w]

    # network & output
    net = STR4DFlowNet(res_increase,low_res_block=low_res_block, high_res_block=high_res_block,  upsampling_block=upsampling_block , post_processing_block=post_processing_block)
    prediction = net.build_network(u, v, w, u_mag, v_mag, w_mag, n_low_resblock, n_hi_resblock, include_mag=include_mag_input)
    model = tf.keras.Model(input_layer, prediction)

    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="My script description")
    parser.add_argument("--model", type=str, help="Optional argument to pass the name of the model")
    args = parser.parse_args()

    # Define directories and filenames
    if args.model is not None:
        model_name = args.model
    else:
        model_name = '20250618-0223' 
    print(f"-----------------IN-VIVO PREDICTION OF PAIRED DATA WITH TEMPORAL 4DFLOWNET of model {model_name}-----------------------")

    # set filenames and directories
    data_dir = 'Temporal4DFlowNet/data/paired_invivo'
    patients = ['v3_wholeheart_25mm_40ms', 'v4_wholeheart_25mm_40ms', 'v5_wholeheart_25mm_40ms', 
                'v6_wholeheart_25mm_40ms', 'v7_wholeheart_25mm_40ms'] 
    # patients = ['v3_wholeheart_25mm_40ms_transformed', 'v4_wholeheart_25mm_40ms_transformed', 'v5_wholeheart_25mm_40ms_transformed', 
    #             'v6_wholeheart_25mm_40ms_transformed', 'v7_wholeheart_25mm_40ms_transformed'] 
    
    output_root = f'Temporal4DFlowNet/results/in_vivo/paired_data'


    for patient in patients:
        print('Patient:', patient)
        t_0 = time.time()
        
        # Change this for every new dataset
        # Setting up
        input_filepath = f'{data_dir}/{patient}.h5' 
        output_dir = f'{output_root}/{patient}'

        output_filepath = f'{output_dir}/{patient}_{model_name}.h5'    
        
        model_path = f'Temporal4DFlowNet/models/Temporal4DFlowNet_{model_name}/Temporal4DFlowNet-best.h5'

        # Params
        patch_size = 16 # take larger patchsize for only upsampling operation
        res_increase = 2
        batch_size = 64
        round_small_values = False
        downsample_input_first = False # This is important for invivo data: either only upsample (visual evaluation) or downsample and compare to original

        # Network - default 8-4
        include_mag_input = False
        n_low_resblock = 8
        n_hi_resblock = 4
        low_res_block  = 'resnet_block'     
        high_res_block = 'resnet_block'     
        upsampling_block = 'linear'
        post_processing_block = None       

        venc_colnames = ['u_max', 'v_max', 'w_max']

        if os.path.exists(output_filepath):
            print(f"Output file already exists: {output_filepath}")
            print("Skipping this patient..")
            continue
        assert os.path.exists(input_filepath), f"Input file does not exist: {input_filepath}"
        assert os.path.exists(model_path), f"Model file does not exist: {model_path}"

        pgen = PatchGenerator(patch_size, res_increase,include_all_axis = True, downsample_input_first = downsample_input_first)
        dataset = ImageDataset_temporal(venc_colnames=['u_max', 'v_max', 'w_max'])
        
        os.makedirs(output_dir, exist_ok=True)

        axis = [0, 1, 2]

        with h5py.File(input_filepath, mode = 'r' ) as h5:
            lr_shape = np.asarray(h5.get("u")).squeeze().shape
            print("Shape of in-vivo data", lr_shape)
            N_frames, X, Y, Z = lr_shape


        if downsample_input_first:
            u_combined = np.zeros(lr_shape)
            v_combined = np.zeros(lr_shape)
            w_combined = np.zeros(lr_shape)
        else:
            N_frames = N_frames*2
            u_combined = np.zeros((N_frames, X, Y, Z))
            v_combined = np.zeros((N_frames, X, Y, Z))
            w_combined = np.zeros((N_frames, X, Y, Z))

        # Loop over all axis
        for a in axis:
            print("________________________________Predict patches with axis: ", a, " ____________________________________________________")
            
            # Check the number of rows in the file
            nr_rows = dataset.get_dataset_len(input_filepath, a)
            print(f"Number of rows in dataset: {nr_rows}")
            
            print(f"Loading 4DFlowNet: {res_increase}x upsample")
            # Load the network
            network = prepare_temporal_network(patch_size, res_increase, n_low_resblock, n_hi_resblock, low_res_block, high_res_block, upsampling_block, post_processing_block=post_processing_block, include_mag_input=include_mag_input)
            network.load_weights(model_path)

            volume = np.zeros((3, u_combined.shape[0],  u_combined.shape[1], u_combined.shape[2],  u_combined.shape[3] ))
            # loop through all the rows in the input file
            for nrow in range(nr_rows):
                print("\n--------------------------")
                print(f"\nProcessed ({nrow+1}/{nr_rows}) - {time.ctime()}")

                # Load data file and indexes
                dataset.load_vectorfield(input_filepath, nrow, axis = a)
                print(f"Original image shape: {dataset.u.shape}")
                
                velocities, magnitudes = pgen.patchify(dataset)
                data_size = len(velocities[0])
                print(f"Patchified. Nr of patches: {data_size} - {velocities[0].shape}")

                # Predict the patches
                results = np.zeros((0,patch_size*res_increase, patch_size, patch_size, 3))
                start_time = time.time()

                for current_idx in range(0, data_size, batch_size):
                    time_taken = time.time() - start_time
                    print(f"\rProcessed {current_idx}/{data_size} Elapsed: {time_taken:.2f} secs.", end='\r')
                    # Prepare the batch to predict
                    patch_index = np.index_exp[current_idx:current_idx+batch_size]
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
                # End of batch loop    
            
                time_taken = time.time() - start_time
                print(f"\rProcessed {data_size}/{data_size} Elapsed: {time_taken:.2f} secs.")

                
                for i in range (0,3):
                    v = pgen._patchup_with_overlap(results[:,:,:,:,i], pgen.nr_x, pgen.nr_y, pgen.nr_z)
                    
                    # Denormalized
                    v = v * dataset.venc 
                    if round_small_values:
                        print(f"Zero out velocity component less than {dataset.velocity_per_px}")
                        # remove small velocity values
                        v[np.abs(v) < dataset.velocity_per_px] = 0
                    
                    v = np.expand_dims(v, axis=0)
                    # prediction_utils.save_to_h5(f'{output_dir}/{output_filename}', f'{dataset.velocity_colnames[i]}_axis{a}', v, compression='gzip')
                    print('Original volume: ', volume.shape, 'shape of predicition', v.shape)
                    if v.shape[1] != N_frames:
                        print('reshaped v from: ', v.shape)
                        if v.shape[1] < N_frames:
                            v = np.pad(v, (0, 0), (0, N_frames - v.shape[1]), (0, 0), (0, 0))
                        else:
                            v = v[:, :N_frames, :, :]
                        print(v.shape)
                    #volume u/v/w, T, X, Y, Z
                    if a == 0:      volume[i, :, nrow,  :,      :] = v
                    elif a == 1:    volume[i, :, :,     nrow,   :] = v
                    elif a == 2:    volume[i, :, :,     :,   nrow] = v


                if dataset.dx is not None:
                    new_spacing = dataset.dx / res_increase
                    new_spacing = np.expand_dims(new_spacing, axis=0) 

            u_combined += volume[0, :, :, :] 
            v_combined += volume[1, :, :, :] 
            w_combined += volume[2, :, :, :] 


        print("Elapsed time: ", time.time() - t_0)
        # save and divide by 3 to get average
        prediction_utils.save_to_h5(output_filepath, "u_combined", u_combined/len(axis), compression='gzip')
        prediction_utils.save_to_h5(output_filepath, "v_combined", v_combined/len(axis), compression='gzip')
        prediction_utils.save_to_h5(output_filepath, "w_combined", w_combined/len(axis), compression='gzip')

        # save the venc, input_filepath, and patch size
        prediction_utils.save_to_h5(output_filepath, "input_filepath", np.array([input_filepath], dtype='S'))

        if dataset.venc is not None:
            prediction_utils.save_to_h5(output_filepath, "venc", np.array(dataset.venc, dtype='float32'))

        # Save the patch size as a dataset
        patch_size_tuple = (patch_size, patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        patch_array = np.array(patch_size_tuple) if isinstance(patch_size_tuple, (tuple, list)) else patch_size_tuple
        prediction_utils.save_to_h5(output_filepath, "patch_size", patch_array)

    print("Done!")
    

