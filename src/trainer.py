import numpy as np
import os
import csv
import tensorflow as tf
from Network.PatchHandler3D_temporal import PatchHandler4D, PatchHandler4D_all_axis, PatchHandler4D_extended_data_augmentation, PatchHandler4D_extended_data_augmentation_optimized
from Network.TrainerController_temporal import TrainerController_temporal

def load_indexes(index_file):
    """
        Load patch index file (csv). This is the file that is used to load the patches based on x,y,z index
    """
    indexes = np.genfromtxt(index_file, delimiter=',', skip_header=True, dtype='unicode') # 'unicode' or None
    return indexes

def check_csv_header(index_file, expected_headers):
    with open(index_file, 'r') as file:
        header = file.readline().strip().split(',')
    is_valid = all(header_name in header for header_name in expected_headers)
    return is_valid

def write_settings_into_csv_file(filename,name, training_file, validation_file, test_file, epochs,batch_size,patch_size, low_resblock, high_resblock, upsampling_type, low_block_type, high_block_type, post_block_type, sampling, notes):
    """
        Write settings into csv file to store training runs
    """
    print(f"Write settings into overview file {filename}")
    fieldnames = ["Name","training_file","validation_file","test_file","epochs","batch_size","patch_size","res_increase","low_resblock","high_resblock","upsampling_type", "low_block_type", "high_block_type", "post_block_type", "sampling",  "notes"]
    with open(filename, mode='a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow({'Name':name, "training_file":training_file, "validation_file":validation_file, "test_file":test_file, "epochs":epochs, "batch_size":batch_size, "patch_size":patch_size, "res_increase":res_increase, "low_resblock":low_resblock, "high_resblock":high_resblock, 
                             "upsampling_type": upsampling_type, 'low_block_type': low_block_type, 'high_block_type':high_block_type, 'post_block_type':post_block_type, 'sampling':sampling, "notes":notes })

if __name__ == "__main__":
    data_dir = 'Temporal4DFlowNet/data/CARDIAC'
    csv_dir = f'{data_dir}/csv_files'
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("GPUs detected by TensorFlow:", tf.config.list_physical_devices('GPU'))
    print("Using GPU: ", tf.test.is_gpu_available())

    # tf.debugging.set_log_device_placement(True)

    # ---- Patch index files ---- 
    # training_file = '{}/Temporal16MODEL2356_2mm_step2_cs_invivomagn_exclfirst2frames_WITHaugmentation_tpatchsize16_mixed_more_less_noise.csv'.format(csv_dir) 
    training_file = '{}/Temporal16MODEL2356_2mm_step4_TMI_physicsconsistent2_minusW_more_noise.csv'.format(csv_dir)
    validate_file = '{}/Temporal16MODEL1_2mm_step2_cs_invivomagn_exclfirst2frames_NOaugmentation.csv'.format(csv_dir) 
    # training_file = '{}/Temp23MODEL1356_physc_jessi5_23_23.csv'.format(csv_dir)
    # validate_file = '{}/Temp23MODEL2_physc_jessi5_23_23.csv'.format(csv_dir)
    # training_file = '{}/Temporal16MODEL2235_2mm_step4_targetSNRdb1445_TMI.csv'.format(csv_dir)
    # validate_file = '{}/Temporal16MODEL1_2mm_step4_targetSNRdb1445_TMI.csv'.format(csv_dir)
    QUICKSAVE = False
    benchmark_file = '{}/Temporal16MODEL4_2mm_step2_cs_invivomagn_exclfirst2frames_NOaugmentation.csv'.format(csv_dir)
    # benchmark_file = '{}/Temp23MODEL4_physc_jessi5_23_23.csv'.format(csv_dir)
    # benchmark_file = '{}/Temporal16MODEL4_2mm_step4_targetSNRdb1445_TMI.csv'.format(csv_dir)

    overview_csv = '/proj/multipress/users/x_piaca/Temporal4DFlowNet/results/Overview_models.csv'

    restore = False
    if restore:
        model_dir = "Temporal4DFlowNet/models/Temporal4DFlowNet_20250625-1642"
        model_file = "Temporal4DFlowNet-best.h5"

    # Adapt how patches are saved for temporal domain if True a different loading scheme is used
    load_patches_all_axis = True

    print('Check, that all the files exist:', os.path.isfile(training_file), os.path.isfile(validate_file), os.path.isfile(benchmark_file), os.path.isfile(overview_csv))

    # Hyperparameters optimisation variables
    initial_learning_rate = 2e-4
    epochs =  130
    batch_size = 32
    mask_threshold = 0.6
    lr_decay_epochs = 0

    # Network setting
    network_name = 'Temporal4DFlowNet'
    patch_size_tuple = (16, 16, 16)#(5, 23, 23)#(16, 16, 16)#(5, 20, 20)#(16, 16, 16) #(5, 20, 20)# (16, 16, 16)#
    res_increase = 2
    
    # Residual blocks, default (8 LR ResBlocks and 4 HR ResBlocks)
    include_mag_input = False
    n_low_resblock = 8
    n_hi_resblock = 4
    low_res_block  = 'resnet_block' # 'resnet_block' 'dense_block' csp_block
    high_res_block = 'resnet_block' #'resnet_block'
    upsampling_block = 'linear' #'linear'     #'Conv3DTranspose'#'nearest_neigbor'#'linear' #' 'linear'  'nearest_neigbor' 'Conv3DTranspose'
    post_processing_block = None #  'unet_block'#None#'unet_block'
    sampling = '-' # this is not used for training but saved in the csv file for a better overview of what data it was trained on 

    shuffle = True       

    #notes: if something about this training is more 'special' is can be added to the overview csv file
    # notes= f'Rerun baselinesetup - physics consistent run, {patch_size_tuple}, batchsize 32'
    notes = f'Training with only velocity and not magnitude as input, {patch_size_tuple}, batchsize 32'
    # Load data file and indexes
    trainset = load_indexes(training_file)
    valset =   load_indexes(validate_file)
    # check wether is contains data augmenttaion parameters
    if check_csv_header(training_file, ['s_patchsize','t_patchsize','flip_1','flip_2','rot','sign_u','sign_v','sign_w','swap_u','swap_v','swap_w']):
        print('Data augmentation parameters found in csv file')
        extended_data_augmentation = True
    else:
        extended_data_augmentation = False
    
    # ----------------- TensorFlow stuff -------------------
    # TRAIN dataset iterator
    if load_patches_all_axis:
        if extended_data_augmentation:
            # z = PatchHandler4D_extended_data_augmentation(data_dir, patch_size_tuple, res_increase, batch_size, mask_threshold)
            z = PatchHandler4D_extended_data_augmentation_optimized(data_dir, patch_size_tuple, res_increase, batch_size, mask_threshold, csv_file=training_file)
        else:
            z = PatchHandler4D_all_axis(data_dir, patch_size_tuple, res_increase, batch_size, mask_threshold)
    else:
        z = PatchHandler4D(data_dir, patch_size_tuple, res_increase, batch_size, mask_threshold)
    trainset = z.initialize_dataset(trainset, shuffle=shuffle)
    
    # VALIDATION iterator
    if load_patches_all_axis: 
        if extended_data_augmentation:
            # valdh = PatchHandler4D_extended_data_augmentation(data_dir, patch_size_tuple, res_increase, batch_size, mask_threshold)
            valdh = PatchHandler4D_extended_data_augmentation_optimized(data_dir, patch_size_tuple, res_increase, batch_size, mask_threshold, csv_file=validate_file)
        else:
            valdh = PatchHandler4D_all_axis(data_dir, patch_size_tuple, res_increase, batch_size, mask_threshold)
    else:
        valdh = PatchHandler4D(data_dir, patch_size_tuple, res_increase, batch_size, mask_threshold)
    valset = valdh.initialize_dataset(valset, shuffle=shuffle)

    # # Bechmarking dataset, use to keep track of prediction progress per best model
    testset = None
    if QUICKSAVE and benchmark_file is not None:
        # WE use this bechmarking set so we can see the prediction progressing over time
        benchmark_set = load_indexes(benchmark_file)
        if load_patches_all_axis: 
            if extended_data_augmentation:
                ph = PatchHandler4D_extended_data_augmentation_optimized(data_dir, patch_size_tuple, res_increase, batch_size, mask_threshold, csv_file=benchmark_file)
            else:
                ph = PatchHandler4D_all_axis(data_dir, patch_size_tuple, res_increase, batch_size, mask_threshold)
        else:
            ph = PatchHandler4D(data_dir, patch_size_tuple, res_increase, batch_size, mask_threshold)
        # No shuffling, so we can save the first batch consistently
        testset = ph.initialize_dataset(benchmark_set, shuffle=False) 

    # ------- Main Network ------
    print(f"4DFlowNet Patch {patch_size_tuple}, lr {initial_learning_rate}, batch {batch_size}")
    network = TrainerController_temporal(patch_size_tuple, res_increase, initial_learning_rate, QUICKSAVE, network_name, n_low_resblock, n_hi_resblock, low_res_block, high_res_block, upsampling_block =  upsampling_block, post_processing_block=post_processing_block, lr_decay_epochs=lr_decay_epochs, include_mag_input=include_mag_input)
    print("Network created, now initializing model directory...")
    network.init_model_dir()

    if restore:
        print(f"Restoring model {model_file}...")
        network.restore_model(model_dir, model_file)
        print("Learning rate", network.optimizer.lr.numpy())
    
    # write into csv file
    print("Write settings into overview csv file..", flush=True)
    write_settings_into_csv_file(overview_csv,network.unique_model_name, os.path.basename(training_file) , os.path.basename(validate_file), os.path.basename(benchmark_file), epochs,batch_size,patch_size_tuple, n_low_resblock, n_hi_resblock,upsampling_block, low_res_block, high_res_block, post_processing_block, sampling, notes)
    

    print(network.unique_model_name, "with", n_low_resblock, "low res blocks and", n_hi_resblock, "high res blocks")
    print("Patch size:", patch_size_tuple, "Res increase:", res_increase)
    print("Initial learning rate:", initial_learning_rate, "Batch size:", batch_size, "Mask threshold:", mask_threshold)
    print("Extended data augmentation:", extended_data_augmentation)
    print("Training file:", training_file)
    print("Validation file:", validate_file, flush=True)
    if testset is not None:
        print("Benchmark file:", benchmark_file)
    print("Shuffling:", shuffle)
    print(f"Training {network.unique_model_name} with {len(trainset)} training patches and {len(valset)} validation patches")
    print(f"Train dataset cardinality: {tf.data.experimental.cardinality(trainset).numpy()}", flush=True)
    print(f"Val dataset cardinality: {tf.data.experimental.cardinality(valset).numpy()}", flush=True)
    
    network.train_network(trainset, valset, n_epoch=epochs, testset=testset)
