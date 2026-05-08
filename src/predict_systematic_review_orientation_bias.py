# We want to estimate and analyse the orientation bias in the systematic review dataset

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
import argparse
# from utils.evaluate_utils import *
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

def calculate_relative_error_normalized(u_pred, v_pred, w_pred, u_hi, v_hi, w_hi, binary_mask):
    '''
    Calculate relative error with tanh as normalization
    '''

    # if epsilon is set to 0, we will get nan and inf
    epsilon = 1e-5 

    if len(binary_mask.squeeze().shape) ==3:
        print('Create temporal mask to calculate relative error')
        binary_mask = np.repeat(binary_mask[np.newaxis, :, :, :], u_pred.shape[0], axis=0) # repeat the mask for all 3 components

    u_diff = np.square(u_pred - u_hi)
    v_diff = np.square(v_pred - v_hi)
    w_diff = np.square(w_pred - w_hi)

    diff_speed = np.sqrt(u_diff + v_diff + w_diff)
    actual_speed = np.sqrt(np.square(u_hi) + np.square(v_hi) + np.square(w_hi)) 

    relative_speed_loss = np.tanh(diff_speed / (actual_speed + epsilon))


    # Apply correction, only use the diff speed if actual speed is zero
    condition = np.not_equal(actual_speed, np.array(0.)) # chnages from condition = np.not_equal(actual_speed, np.array(tf.constant(0.)))
    corrected_speed_loss = np.where(condition, relative_speed_loss, diff_speed)

    multiplier = 1e4 # round it so we don't get any infinitesimal number
    corrected_speed_loss = np.round(corrected_speed_loss * multiplier) / multiplier

    binary_mask_condition = np.equal(binary_mask, 1.0)          
    corrected_speed_loss = np.where(binary_mask_condition, corrected_speed_loss, np.zeros_like(corrected_speed_loss))

    mean_err = np.sum(corrected_speed_loss, axis=(1,2,3)) / (np.sum(binary_mask, axis=(1, 2, 3)) + 1) 

    # now take the actual mean
    # mean_err = tf.reduce_mean(mean_err) * 100 # in percentage
    mean_err = mean_err * 100

    return mean_err

def prepare_temporal_network(patch_size, res_increase, n_low_resblock, n_hi_resblock, low_res_block, high_res_block, upsampling_block, post_processing_block):
    # Prepare input
    input_shape = (patch_size,patch_size,patch_size,1)
    u = tf.keras.layers.Input(shape=input_shape, name='u')
    v = tf.keras.layers.Input(shape=input_shape, name='v')
    w = tf.keras.layers.Input(shape=input_shape, name='w')

    u_mag = tf.keras.layers.Input(shape=input_shape, name='u_mag')
    v_mag = tf.keras.layers.Input(shape=input_shape, name='v_mag')
    w_mag = tf.keras.layers.Input(shape=input_shape, name='w_mag')

    input_layer = [u,v,w,u_mag, v_mag, w_mag]

    # network & output
    net = STR4DFlowNet(res_increase,low_res_block=low_res_block, high_res_block=high_res_block,  upsampling_block=upsampling_block , post_processing_block=post_processing_block)
    prediction = net.build_network(u, v, w, u_mag, v_mag, w_mag, n_low_resblock, n_hi_resblock)
    model = tf.keras.Model(input_layer, prediction)

    return model

def inverse_permutation(perm):
    inv_perm = np.empty_like(perm)
    for i, p in enumerate(perm):
        inv_perm[p] = i
    return inv_perm

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="My script description")
    parser.add_argument("--model", type=str, help="Optional argument to pass the name of the model")
    args = parser.parse_args()

    # Define directories and filenames
    if args.model is not None:
        model_name = args.model
    else:
        model_name = '20240709-2057' # this model: training 2, 3, validation: 1, test:4 

    print("Model name: ", model_name)

    # set_names = ['Validation']
    # data_models= [ '1', ]
    # steps = [ 2]
    # file_names = ['M1_2mm_step2_cs_invivoP01_lr.h5', ]
    # hr_dir = 'Temporal4DFlowNet/data/CARDIAC'
    # hr_filename = ['M1_2mm_step2_static_dynamic.h5']
    # venc_colnames = ['u_max', 'v_max', 'w_max']

    set_names = ['']
    data_models= [ '', ]
    steps = [ 2]
    file_names = ['v3_wholeheart_25mm_40ms_transformed.h5', ]
    hr_dir = 'Temporal4DFlowNet/data/paired_invivo'
    hr_filename = ['v3_wholeheart_25mm_20ms_transformed.h5']
    venc_colnames = ['u_max', 'v_max', 'w_max']

    # set filenamaes and directories
    data_dir = 'Temporal4DFlowNet/data/paired_invivo'
    output_dir = f'Temporal4DFlowNet/results/Temporal4DFlowNet_{model_name}/orientation_bias'
    model_dir = f'Temporal4DFlowNet/models/Temporal4DFlowNet_{model_name}'

    #TODO next step: try to mitigate orientation bias and retrain network
    # think about which variation makes sense


    orientation_variations = {
        '0': {
            'transpose_order': (0, 1, 2),
            'sign_variation': (1, 1, 1), 
            'description': 'Original, reference result', 
            'name': 'Original',
        },
        '1': {
            'transpose_order': (0, 1, 2),
            'sign_variation': (-1, -1, -1),
            'description': 'Signs inverted on all components',
            'name': 'Signs_inverted'
        },
        '2': {
            'transpose_order': (0, 1, 2),
            'sign_variation': (-1, 1, 1),
            'description': 'Signs inverted on u-component',
            'name': 'Signs_inverted_u'
        },
        '3': {
            'transpose_order': (0, 1, 2),
            'sign_variation': (1, -1, 1),
            'description': 'Signs inverted on v-component',
            'name': 'Signs_inverted_v'
        },
        '4': {
            'transpose_order': (0, 1, 2),
            'sign_variation': (1, 1, -1),
            'description': 'Signs inverted on w-component',
            'name': 'Signs_inverted_w'
        },
        '5': {
            'transpose_order': (0, 1, 2),
            'sign_variation': (-1, -1, 1),
            'description': 'Signs inverted on u and v-component',
            'name': 'Signs_inverted_uv'
        },
        '6': {
            'transpose_order': (2, 1, 0),
            'sign_variation': (1, 1, 1),
            'description': 'Orientation 3, 2, 1',
            'name': 'Transposed_1'
        },
        '7': {
            'transpose_order': (1, 0, 2),
            'sign_variation': (1, 1, 1),
            'description': 'Orientation 2, 1, 3',
            'name': 'Transposed_2'
        },
        '8': {
            'transpose_order': (0, 2, 1),
            'sign_variation': (1, 1, 1),
            'description': 'Orientation 1, 3, 2',
            'name': 'Transposed_3'
        },
        '9': {
            'transpose_order': (2, 0, 1),
            'sign_variation': (-1, -1, -1),
            'description': 'Orientation 3, 1, 2 and signs inverted on all components',
            'name': 'Transposed_4'
        },

    }

    for i_ter, (set_name, data_model, step, filename) in enumerate(zip(set_names, data_models, steps, file_names)):
        print('Start predicition of:', set_name, data_model, filename)
        results_one_dataset = {}
        keys_variation = list(orientation_variations.keys())
        for var in keys_variation:
            print("---------Orientation variation: ", var, " - ", orientation_variations[var]["description"], "-----------------")

            inv_perm = inverse_permutation(np.array(orientation_variations[var]['transpose_order']))
            print("permutation and inverse permutation", orientation_variations[var]['transpose_order'], inv_perm)

            # set filenamaes and directories
            output_dir = f'Temporal4DFlowNet/results/Temporal4DFlowNet_{model_name}'
            # output_filename = f'{set_name}set_result_model{data_model}_2mm_step{step}_{model_name[-4::]}_temporal_{orientation_variations[var]["name"]}.h5'
            output_filename = f'{data_model}_{model_name}_{orientation_variations[var]["name"]}.h5'

            model_path = f'{model_dir}/Temporal4DFlowNet-best.h5'

            # Params
            patch_size = 16
            res_increase = 2
            batch_size = 64
            round_small_values = False
            downsample_input_first = False # For LR data, which is downsampled on the fly this should be set to True
            upsampling_factor = 2

            # Network - default 8-4
            n_low_resblock = 8
            n_hi_resblock = 4
            low_res_block  = 'resnet_block'
            high_res_block = 'resnet_block'   
            upsampling_block = 'linear'        
            post_processing_block = None    

            # Setting up
            input_filepath = '{}/{}'.format(data_dir, filename)
            output_filepath = '{}/{}'.format(output_dir, output_filename)
            print("Output file path: ", output_filepath)
            if not os.path.exists(f'{output_dir}/{output_filename}'):

                pgen = PatchGenerator(patch_size, res_increase,include_all_axis = True, downsample_input_first=downsample_input_first)
                dataset = ImageDataset_temporal(venc_colnames=venc_colnames)#['venc_u', 'venc_v', 'venc_w'])

                print("Path exists:", os.path.exists(input_filepath), os.path.exists(model_path))
                print("Outputfile exists already: ", os.path.exists(output_filename))

                os.makedirs(output_dir, exist_ok=True)

                with h5py.File(input_filepath, mode = 'r' ) as h5:
                    lr_shape = h5.get("u").shape

                if downsample_input_first:
                    u_combined = np.zeros(lr_shape)
                    v_combined = np.zeros(lr_shape)
                    w_combined = np.zeros(lr_shape)
                else:
                    u_combined = np.zeros((upsampling_factor*lr_shape[0], *lr_shape[1::]))
                    v_combined = np.zeros((upsampling_factor*lr_shape[0], *lr_shape[1::]))
                    w_combined = np.zeros((upsampling_factor*lr_shape[0], *lr_shape[1::]))

                # Loop over all axis
                for a in [0, 1, 2]:
                    print("________________________________Predict patches with axis: ", a, " ____________________________________________________")
                    
                    # Check the number of rows in the file
                    nr_rows = dataset.get_dataset_len(input_filepath, a)
                    
                    print(f"Number of rows in dataset: {nr_rows}")
                    print(f"Loading 4DFlowNet: {res_increase}x upsample")

                    # Load the network
                    network = prepare_temporal_network(patch_size, res_increase, n_low_resblock, n_hi_resblock, low_res_block, high_res_block, upsampling_block, post_processing_block)
                    network.load_weights(model_path)

                    volume = np.zeros((3, u_combined.shape[0],  u_combined.shape[1], u_combined.shape[2],  u_combined.shape[3] ))
                    # loop through all the rows in the input file
                    for nrow in range(nr_rows):
                        print("\n--------------------------")
                        print(f"\nProcessed ({nrow+1}/{nr_rows}) - {time.ctime()}")

                        # Load data file and indexes
                        dataset.load_vectorfield(input_filepath, nrow, axis = a)# , transpose=orientation_variations[orientation_variation]['transpose_order']
                        print(f"Original image shape: {dataset.u.shape}")
                        
                        velocities, magnitudes = pgen.patchify(dataset)
                        data_size = len(velocities[0])
                        print(f"Patchified. Nr of patches: {data_size} - {velocities[0].shape}")

                        # Predict the patches
                        results_temp = np.zeros((0,patch_size*res_increase, patch_size, patch_size, 3))
                        
                        start_time = time.time()

                        # make changes on velocities and magnitudes
                        velocities = (velocities[0]*orientation_variations[var]['sign_variation'][0], velocities[1]*orientation_variations[var]['sign_variation'][1], velocities[2]*orientation_variations[var]['sign_variation'][2])
                        velocities = (velocities[orientation_variations[var]['transpose_order'][0]], velocities[orientation_variations[var]['transpose_order'][1]], velocities[orientation_variations[var]['transpose_order'][2]])

                        
                        for current_idx in range(0, data_size, batch_size):
                            time_taken = time.time() - start_time
                            print(f"\rProcessed {current_idx}/{data_size} Elapsed: {time_taken:.2f} secs.", end='\r')
                            # Prepare the batch to predict
                            patch_index = np.index_exp[current_idx:current_idx+batch_size]
                            sr_images = network.predict([velocities[0][patch_index],
                                                    velocities[1][patch_index],
                                                    velocities[2][patch_index],
                                                    magnitudes[0][patch_index],
                                                    magnitudes[1][patch_index],
                                                    magnitudes[2][patch_index]])

                            results_temp = np.append(results_temp, sr_images, axis=0)
                        
                        results = np.zeros_like(results_temp)
                        results[:, :, :, :, 0] = results_temp[:, :, :, :, inv_perm[0]].copy()
                        results[:, :, :, :, 1] = results_temp[:, :, :, :, inv_perm[1]].copy()
                        results[:, :, :, :, 2] = results_temp[:, :, :, :, inv_perm[2]].copy()

                        results[:, :, :, :, 0] *= orientation_variations[var]['sign_variation'][0]
                        results[:, :, :, :, 1] *= orientation_variations[var]['sign_variation'][1]
                        results[:, :, :, :, 2] *= orientation_variations[var]['sign_variation'][2]

                    
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
                            if a == 0:      volume[i, :, nrow,  :,      :] = v
                            elif a == 1:    volume[i, :, :,     nrow,   :] = v
                            elif a == 2:    volume[i, :, :,     :,   nrow] = v

                    u_combined += volume[0, :, :, :] 
                    v_combined += volume[1, :, :, :] 
                    w_combined += volume[2, :, :, :] 

                print("save combined predictions")
                # save and divide by 3 to get average
                prediction_utils.save_to_h5(f'{output_dir}/{output_filename}', "u_combined", u_combined/3, compression='gzip')
                prediction_utils.save_to_h5(f'{output_dir}/{output_filename}', "v_combined", v_combined/3, compression='gzip')
                prediction_utils.save_to_h5(f'{output_dir}/{output_filename}', "w_combined", w_combined/3, compression='gzip')

                print('-----------------------------Done with prediction: ', set_name, data_model, '----------------------------------------------')

                # Now evaluate the orientation bias
                u_pred = u_combined/3
                v_pred = v_combined/3
                w_pred = w_combined/3
            else:
                print(f"Load the prediction file from {output_filepath}")
                with h5py.File(f'{output_dir}/{output_filename}', mode = 'r' ) as h5:
                    u_pred = np.array(h5["u_combined"])
                    v_pred = np.array(h5["v_combined"])
                    w_pred = np.array(h5["w_combined"])

            print('Prediction shape: ', u_pred.shape, v_pred.shape, w_pred.shape)
            # 1. Load the HR data
            hr_filepath = f'{hr_dir}/{hr_filename[i_ter]}'
            with h5py.File(hr_filepath, mode = 'r' ) as h5:
                u_hr = np.array(h5["u"])
                v_hr = np.array(h5["v"])
                w_hr = np.array(h5["w"])
                mask = np.array(h5["mask"])
            
            with h5py.File(f'{input_filepath}', mode = 'r' ) as h5:
                u_lr = np.array(h5["u"])
                v_lr = np.array(h5["v"])
                w_lr = np.array(h5["w"])

            # evaluate RMSE; K, R2, mean vel plot

            # 2. Calculate the RMSE
            rmse_u = np.sqrt(np.mean((u_pred - u_hr)**2, where=mask.astype(bool), axis=(1, 2, 3)))
            rmse_v = np.sqrt(np.mean((v_pred - v_hr)**2, where=mask.astype(bool), axis=(1, 2, 3)))
            rmse_w = np.sqrt(np.mean((w_pred - w_hr)**2, where=mask.astype(bool), axis=(1, 2, 3)))

            RE = calculate_relative_error_normalized(u_pred, v_pred, w_pred, u_hr, v_hr, w_hr, mask)
            results_one_dataset[var] = {
                'rmse_u': rmse_u,
                'rmse_v': rmse_v,
                'rmse_w': rmse_w,
                'RE': RE,
                'mean_u': np.mean(u_pred, where=mask.astype(bool), axis = (1, 2, 3)),
                'mean_v': np.mean(v_pred, where=mask.astype(bool), axis = (1, 2, 3)),
                'mean_w': np.mean(w_pred, where=mask.astype(bool), axis = (1, 2, 3)),
            }

            # RE = calculate_relative_error_normalized(u_pred, v_pred, w_pred, u_hr, v_hr, w_hr, mask)

            # 3. Calculate the K
            # print("Results rmse: ", rmse_u, rmse_v, rmse_w)
            print("mean of rmse: ", np.mean(rmse_u), np.mean(rmse_v), np.mean(rmse_w))
            evaluate_dir = f'{output_dir}/plots/directional_bias'
            os.makedirs(evaluate_dir, exist_ok=True)


         # plot rmse subplots
        fig = plt.figure(figsize=(15, 5))
        for var in keys_variation:
            plt.subplot(1, 3, 1)
            plt.plot(results_one_dataset[var]['rmse_u'], label=orientation_variations[var]['name'])
            plt.xlabel('Time frame')
            plt.ylabel('RMSE')
            plt.title('RMSE u-component')

            plt.subplot(1, 3, 2)
            plt.plot(results_one_dataset[var]['rmse_v'], label=orientation_variations[var]['name'])
            plt.xlabel('Time frame')
            plt.ylabel('RMSE')
            plt.title('RMSE v-component')
            plt.legend()

            plt.subplot(1, 3, 3)
            plt.plot(results_one_dataset[var]['rmse_w'], label=orientation_variations[var]['name'])
            plt.xlabel('Time frame')
            plt.ylabel('RMSE')
            plt.title('RMSE w-component')
            plt.legend()
        plt.tight_layout()
        plt.savefig(f'{evaluate_dir}/rmse_orientation_bias.png')

        # plot RE subplots
        fig = plt.figure(figsize=(10, 5))
        for var in keys_variation:
            plt.plot(results_one_dataset[var]['RE'], label=orientation_variations[var]['name'])
            plt.xlabel('Time frame')
            plt.ylabel('RE')
            plt.title('Relative error')
            plt.legend()
        plt.tight_layout()
        plt.savefig(f'{evaluate_dir}/RE_orientation_bias.png')
        plt.show()

        # plot mean velocity
        fig = plt.figure(figsize=(15, 5))
        for var in keys_variation:
            plt.subplot(1, 3, 1)
            plt.plot(results_one_dataset[var]['mean_u'], label=orientation_variations[var]['name'])
            plt.xlabel('Time frame')
            plt.ylabel('mean u')
            plt.title('Mean vel')
            plt.legend()

            plt.subplot(1, 3, 2)
            plt.plot(results_one_dataset[var]['mean_v'], label=orientation_variations[var]['name'])
            plt.xlabel('Time frame')
            plt.ylabel('mean v')
            plt.title('Mean vel')
            plt.legend()

            plt.subplot(1, 3, 3)
            plt.plot(results_one_dataset[var]['mean_w'], label=orientation_variations[var]['name'])
            plt.xlabel('Time frame')
            plt.ylabel('mean w')
            plt.title('Mean vel')

        plt.subplot(1, 3, 1)
        plt.plot(np.mean(u_hr, where=mask.astype(bool), axis = (1, 2, 3)), label='HR', color = 'black')
        plt.plot(range(0, w_hr.shape[0], 2),np.mean(u_lr, where=mask.astype(bool)[::2], axis = (1, 2, 3)), label='LR', color = 'black', linestyle='dashed')
        plt.plot
        plt.subplot(1, 3, 2)
        plt.plot(np.mean(v_hr, where=mask.astype(bool), axis = (1, 2, 3)), label='HR', color = 'black')
        plt.plot(range(0, w_hr.shape[0], 2),np.mean(v_lr, where=mask.astype(bool)[::2], axis = (1, 2, 3)), label='LR', color = 'black', linestyle='dashed')
        plt.subplot(1, 3, 3)
        plt.plot(np.mean(w_hr, where=mask.astype(bool), axis = (1, 2, 3)), label='HR', color = 'black')
        plt.plot(range(0, w_hr.shape[0], 2), np.mean( w_lr, where=mask.astype(bool)[::2], axis = (1, 2, 3)), label='LR', color = 'black', linestyle='dashed')
        plt.legend
        plt.tight_layout()
        plt.savefig(f'{evaluate_dir}/mean_velocity_orientation_bias.png')
        plt.show()

        mean_results = {}
        for var in keys_variation:
            mean_results[var] = {
                'rmse_u': np.mean(results_one_dataset[var]['rmse_u']),
                'rmse_v': np.mean(results_one_dataset[var]['rmse_v']),
                'rmse_w': np.mean(results_one_dataset[var]['rmse_w']),
                'RE': np.mean(results_one_dataset[var]['RE']),
                'mean_u': np.mean(results_one_dataset[var]['mean_u']),
                'mean_v': np.mean(results_one_dataset[var]['mean_v']),
                'mean_w': np.mean(results_one_dataset[var]['mean_w']),
            }
        print("Results: ", mean_results)






            











    print("Done!")