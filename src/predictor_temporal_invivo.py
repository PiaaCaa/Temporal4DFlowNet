import tensorflow as tf
import numpy as np
import time
import os
from Network.Temporal4DFlowNetModel import T4DFlowNet
from Network.PatchGenerator import PatchGenerator
from utils import prediction_utils
from Temporal4DFlowNet.src.utils.ImageDataset import ImageDataset
from matplotlib import pyplot as plt
import h5py
import timeit
import argparse
from scipy.interpolate import CubicSpline, RegularGridInterpolator
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# temporal insilico

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
    net = T4DFlowNet(res_increase,low_res_block=low_res_block, high_res_block=high_res_block,  upsampling_block=upsampling_block , post_processing_block=post_processing_block)
    prediction = net.build_network(u, v, w, u_mag, v_mag, w_mag, n_low_resblock, n_hi_resblock, include_mag=include_mag_input)
    model = tf.keras.Model(input_layer, prediction)

    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="My script description")
    parser.add_argument("--model", type=str, help="Optional argument to pass the name of the model")
    args = parser.parse_args()

    if args.model is not None:
        model_name = args.model
    else:
        model_name = '20241018-1552' # this model: training 2, 3, validation: 1, test:4 
    print(f"-----------------IN-VIVO PREDICTION (one resolution) WITH TEMPORAL 4DFLOWNET of model {model_name}-----------------------")

    # Define directories and filenames
    # data_dir = 'Temporal4DFlowNet/data/PIA/THORAX'
    # patients = ['P01', 'P02',  'P03', 'P04', 'P05'] 
    data_dir = 'Temporal4DFlowNet/data/Aortic_Phantom_Stanford/transformed_phaseunwrapped'
    patients = ['mc1_4DFlowWIP_Ao_v120_2.5_16frames', 'mc1_4DFlowWIP_Ao_v120_2.5_25frames', 
                'mc2_4DFlowWIP_Ao_v120_2.5_16frames', 'mc2_4DFlowWIP_Ao_v120_2.5_25frames', 
                'mr_4DFlowWIP_Ao_v120_2.5_16frames', 'mr_4DFlowWIP_Ao_v120_2.5_25frames']
    # data_dir = 'Temporal4DFlowNet/data/Brandon_Aortic_Dissection'
    # patients = ['TBAD_OR_Pia_new_transformed']
    output_root = f'Temporal4DFlowNet/results/in_vivo/Aortic_Phantom_Stanford/transformed_phaseunwrapped'
    # output_root = f'Temporal4DFlowNet/results/in_vivo/THORAX'
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
    low_res_block  = 'resnet_block'     # 'resnet_block' 'dense_block' 'csp_block'
    high_res_block = 'resnet_block'     
    upsampling_block = 'linear'         #'Conv3DTranspose' 'nearest_neigbor' 'linear'
    post_processing_block = None        

    venc_colnames = ['u_max', 'v_max', 'w_max']



    for patient in patients:
        print('Patient:', patient)
        t_0 = time.time()
        
        # Change this for every new dataset
        # input_filepath = f'{data_dir}/{patient}/h5/{patient}.h5' 
        input_filepath = f'{data_dir}/{patient}.h5'
        output_dir = f'{output_root}/{patient}'
        output_filepath = f'{output_dir}/{os.path.basename(input_filepath[:-3])}_{model_name}.h5'    
        model_path = f'Temporal4DFlowNet/models/Temporal4DFlowNet_{model_name}/Temporal4DFlowNet-best.h5'


        if os.path.exists(output_filepath):
            print(f"Output file already exists: {output_filepath}, skipping prediction")
            continue

        print("Path exists:", os.path.exists(input_filepath), os.path.exists(model_path))
        print("Outputfile exists already: ", os.path.exists(output_filepath))

        os.makedirs(output_dir, exist_ok=True)

        pgen = PatchGenerator(patch_size, res_increase,include_all_axis = True, downsample_input_first = downsample_input_first)
        dataset = ImageDataset(venc_colnames=['u_max', 'v_max', 'w_max'])

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

        axis = [0, 1, 2]
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
                print(f"Velocities shape: {velocities[0].shape}, Magnitudes shape: {magnitudes[0].shape}")
                print(f'Max velocity: {np.max(velocities[0])}, Min velocity: {np.min(velocities[0])}')
                print(f'Max velocity: {np.max(velocities[1])}, Min velocity: {np.min(velocities[1])}')
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
                    print(f'Shapes of velocities: {velocities[0][patch_index].shape}, {velocities[1][patch_index].shape}, {velocities[2][patch_index].shape}')
                    print(f'Shapes of magnitudes: {magnitudes[0][patch_index].shape}, {magnitudes[1][patch_index].shape}, {magnitudes[2][patch_index].shape}')
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
                    # prediction_utils.save_to_h5(f'{output_dir}/{output_filename}', f'{dataset.velocity_colnames[i]}__axis{a}', v, compression='gzip')
                    print('Original volume: ', volume.shape, 'shape of predicition', v.shape)
                    if v.shape[1] != N_frames:
                        if v.shape[1] < N_frames:
                            v = np.pad(v, (0, 0), (0, N_frames - v.shape[1]), (0, 0), (0, 0))
                        else:
                            v = v[:, :N_frames, :, :]
                        
                    #volume u/v/w, T, X, Y, Z
                    if a == 0:      volume[i, :, nrow,  :,      :] = v
                    elif a == 1:    volume[i, :, :,     nrow,   :] = v
                    elif a == 2:    volume[i, :, :,     :,   nrow] = v


                if dataset.dx is not None:
                    new_spacing = dataset.dx / res_increase
                    new_spacing = np.expand_dims(new_spacing, axis=0) 
                    #prediction_utils.save_to_h5(f'{output_dir}/{output_filename}', dataset.dx_colname, new_spacing, compression='gzip')

            u_combined += volume[0, :, :, :] 
            v_combined += volume[1, :, :, :] 
            w_combined += volume[2, :, :, :] 

        print(f"save combined predictions to {output_filepath}" )
        print("Elapsed time: ", time.time() - t_0)
        # save and divide by 3 to get average
        prediction_utils.save_to_h5(output_filepath, "u_combined", u_combined/len(axis), compression='gzip')
        prediction_utils.save_to_h5(output_filepath, "v_combined", v_combined/len(axis), compression='gzip')
        prediction_utils.save_to_h5(output_filepath, "w_combined", w_combined/len(axis), compression='gzip')

        # save the venc, input_filepath, and patch size
        prediction_utils.save_to_h5(output_filepath, "input_filepath", np.array([input_filepath], dtype='S'))

        if dataset.venc is not None:
            print("Save venc: ", dataset.venc)
            prediction_utils.save_to_h5(output_filepath, "venc", np.array(dataset.venc, dtype='float32'))

        # Save the patch size as a dataset
        patch_size_tuple = (patch_size, patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        patch_array = np.array(patch_size_tuple) if isinstance(patch_size_tuple, (tuple, list)) else patch_size_tuple
        prediction_utils.save_to_h5(output_filepath, "patch_size", patch_array)

        print("Done!")

