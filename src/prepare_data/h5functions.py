import numpy as np
import h5py
import os

def save_to_h5(output_filepath, col_name, dataset, expand_dims = True):
    if expand_dims: dataset = np.expand_dims(dataset, axis=0)

    # convert float64 to float32 to save space
    if dataset.dtype == 'float64':
        dataset = np.array(dataset, dtype='float32')
    
    with h5py.File(output_filepath, 'a') as hf:    
        if col_name not in hf:
            datashape = (None, )
            if (dataset.ndim > 1):
                datashape = (None, ) + dataset.shape[1:]
            hf.create_dataset(col_name, data=dataset, maxshape=datashape)
        else:
            hf[col_name].resize((hf[col_name].shape[0]) + dataset.shape[0], axis = 0)
            hf[col_name][-dataset.shape[0]:] = dataset


def merge_data_to_h5(input_file, toadd_file):
    """
    Add data from merge_file to input_file if keys are not present.
    """
    with h5py.File(input_file, mode='a') as input_h5:
        with h5py.File(toadd_file, mode='r') as toadd_h5:
            for key in toadd_h5.keys():
                if key not in input_h5.keys():
                    print('Adding key', key)
                    dataset = np.array(toadd_h5[key])

                    # convert float64 to float32 to save space
                    if dataset.dtype == 'float64':
                        dataset = np.array(dataset, dtype='float32')
                    datashape = (None, )
                    if (dataset.ndim > 1):
                        datashape = (None, ) + dataset.shape[1:]
                    input_h5.create_dataset(key, data=dataset, maxshape=datashape)

def delete_data_from_h5(h5_file,lst_keys):
    """
    Delete data from keys from h5_file 
    """
    with h5py.File(h5_file, mode='a') as hf:
        for key in lst_keys:
            if key in hf.keys():
                del hf[key]
                print('Deleted key', key)

def create_h5_file(dir, name):
    """
    Create a new h5 file at the specified path and with the given name.
    """
    file_path = os.path.join(dir, name)
    with h5py.File(file_path, 'w') as file:
        pass  # Empty block to create the file
    print(f"Created new h5 file: {file_path}")