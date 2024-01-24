import numpy as np
import os
import h5py
from scipy.integrate import trapz
from matplotlib import pyplot as plt
from utils import prediction_utils

"""
This file contains functions for temporal downsampling
"""


def cartesian_temporal_downsampling(hr, sampling_factor, offset = 0):
    '''
    Downsample by simply taking every n-th frame (sampling factor). Start at offset
    '''
    assert len(hr.shape) == 4, "Input should be 4D"
    assert sampling_factor >= 1, "Sampling factor should be >= 1"

    return hr[offset::sampling_factor, : , :, :]



#TODO : maybe use convolution instead of averaging to increase speed
def temporal_averaging(hr, radius):
    """
    Average the temporal dimension with a given radius
    Ideally, radius should be odd, so that the averaging is symmetric.
    This can be used for temporal downsampling as this is simple average smooting
    """
    assert len(hr.shape) == 4, "Input should be 4D"
    assert radius >= 1, "Radius should be >= 1"

    T = hr.shape[0]
    hr_avg = np.zeros_like(hr)

    # loop through all the frames and save the average in hr_avg                        
    for t in range(T):
        for i in range(t -radius//2, t+radius//2+1):
            
            # use periodical boundary conditions, i.e. after last frame take first frame again and vice verse
            if i >= T :
                i = i%(T)
            
            # sum up all the 3D data 
            hr_avg += np.asarray(hr_avg[i])
    
    # divide by number of frames to take the average
    hr_avg  /= radius

    return hr_avg



def temporal_smoothing_box_function_toeger(hr,t_range, sigma,):
    """
    This is the implementation of the smoothing box function for temporal averaging and is based on the paper:  "Blood flow imaging by optimal matching of computational fluid dynamics to 4D-flow data" by Töger et al. 2020
    Note that temporal boundaries are handled periodically.
    Also, different to suggested in the paper, not the area under the curve is normalized to 1, but the sum of the discrete weights are normalized to 1. 
    t_range: range of the heart cycle, going from 0 to (often) 1000 ms (1s), which is then divided into the number of frames
    Returns the smoothed data.
    """
    assert len(hr.shape) == 4, "Input should be 4D"

    start_t = t_range[0]
    end_t = t_range[-1]
    len_t = end_t - start_t
    hr_avg = np.zeros_like(hr)
    dt = t_range[1] - t_range[0] # temporal resolution

    # extend range to handle periodic boundary conditions
    # extend the boundaries by a quarter fo total time to cover temporal cycle
    extended_boundaries_left = np.arange(start_t, start_t - (len_t / 4), -dt)[::-1] #reverse
    extended_boundaries_left = extended_boundaries_left[:-1] # remove last element as it is already included in t_range
    extended_boundaries_right = np.arange(end_t+dt, end_t+(len_t/4), dt)

    # boundary extension length to be the same
    if len(extended_boundaries_right) > len(extended_boundaries_left):
        extended_boundaries_right = extended_boundaries_right[:-1]
    elif len(extended_boundaries_right) < len(extended_boundaries_left):
        extended_boundaries_left = extended_boundaries_left[1:]

    t_range_extended = np.append(np.append(extended_boundaries_left, t_range), extended_boundaries_right)

    def smoothed_box_fct(t, t0, w, sigma):
            """
            Smoothed box function. With alpha = 1 this is not normalized to 1
            """
            non_normalized = (1/(1+np.exp(-(t-(t0-w/2))/sigma)) - 1/(1+np.exp(-(t-(t0+w/2))/sigma)))
            alpha = 1
            # alpha = 1/integral_trapez(non_normalized, t)
            return alpha * non_normalized

    def integral_trapez(fct, t):
        """
        Calculate the integral of the smoothed box function with trapey formula   
        """
        return trapz(fct, t)

    
    # loop through all the frames and return the smoothed result hr_avg
    for i, t0 in enumerate(t_range):
        weighting =  smoothed_box_fct(t_range_extended, t0, dt, sigma)
        
        # normalize the weighting # note: this is not included in the paper 
        weighting /= np.sum(weighting)

        # add the weighting to the periodic boundaries
        periodic_weighting = np.zeros_like(t_range)
        periodic_weighting = weighting[len(extended_boundaries_left):len(extended_boundaries_left)+len(t_range)] # middle
        periodic_weighting[:len(extended_boundaries_right)] += weighting[-len(extended_boundaries_right):] 
        periodic_weighting[-len(extended_boundaries_left):] += weighting[:len(extended_boundaries_left)]

        # plt.plot(t_range, periodic_weighting)
        # plt.scatter(t0, periodic_weighting[np.where(t_range == t0)])
        # plt.plot(t_range_extended, weighting)
        # plt.scatter(t0, weighting[np.where(t_range_extended == t0)])

        # weight input by the periodic weighting
        hr_avg[i, :, :, :] = np.sum(hr*periodic_weighting[:, None, None, None], axis = 0)
    # plt.show()
    
    print(f"Created temporally smoothed data with sigma = {sigma} in range {start_t} to {end_t} and resolution {dt}")
    return hr_avg



def merge_data_to_h5(input_file, toadd_file):
    """
    Add data from merge_file to input file if keys are not present.
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
    

if __name__ == '__main__':
    # load data
    hr_file = 'data/CARDIAC/M1_2mm_step2_invivoP01_magnitude.h5'

    # save data
    smooth_file_lr = 'data/CARDIAC/M1_2mm_step2_invivoP01_magn_temporalsmoothing_toeger_periodic_LRfct_noise.h5'
    smooth_file_hr = 'data/CARDIAC/M4_2mm_step2_invivoP02_magn_temporalsmoothing_toeger_periodic_HRfct.h5'

    keys =  [ "mag_u", "mag_v", "mag_w", "mask", ] 
    delete_data_from_h5(smooth_file_lr, keys)
    # delete_data_from_h5(smooth_file_hr, keys)
    merge_data_to_h5(smooth_file_lr, hr_file)
    exit()

    with h5py.File(hr_file, mode = 'r' ) as p1:
          hr_u = np.asarray(p1['u']) 
          hr_v = np.asarray(p1['v'])
          hr_w = np.asarray(p1['w'])

   
    t_range = np.linspace(0, 1, hr_u.shape[0])
    smoothing = 0.004

    #-------LR function smoothing-------
    # Note the output will be in same dimension as HR; but the smoothing function is applied on the downsampled data.
    if True: 
        # downsample and then apply smoothing
        hr_u0 = hr_u[::2]
        hr_v0 = hr_v[::2]
        hr_w0 = hr_w[::2]
        hr_u1 = hr_u[1::2]
        hr_v1 = hr_v[1::2]
        hr_w1 = hr_w[1::2]

        t_range0 = t_range[::2]
        t_range1 = t_range[1::2]
        

        hr_u_temporal_smoothing0 = temporal_smoothing_box_function_toeger(hr_u0, t_range0, smoothing)
        hr_v_temporal_smoothing0 = temporal_smoothing_box_function_toeger(hr_v0, t_range0, smoothing)
        hr_w_temporal_smoothing0 = temporal_smoothing_box_function_toeger(hr_w0, t_range0, smoothing)

        hr_u_temporal_smoothing1 = temporal_smoothing_box_function_toeger(hr_u1, t_range1, smoothing)
        hr_v_temporal_smoothing1 = temporal_smoothing_box_function_toeger(hr_v1, t_range1, smoothing)
        hr_w_temporal_smoothing1 = temporal_smoothing_box_function_toeger(hr_w1, t_range1, smoothing)

        u_temp_smoothing = np.zeros_like(hr_u)
        v_temp_smoothing = np.zeros_like(hr_v)
        w_temp_smoothing = np.zeros_like(hr_w)

        u_temp_smoothing[::2] = hr_u_temporal_smoothing0
        u_temp_smoothing[1::2] = hr_u_temporal_smoothing1
        v_temp_smoothing[::2] = hr_v_temporal_smoothing0
        v_temp_smoothing[1::2] = hr_v_temporal_smoothing1
        w_temp_smoothing[::2] = hr_w_temporal_smoothing0
        w_temp_smoothing[1::2] = hr_w_temporal_smoothing1

        if os.path.exists(smooth_file_lr):
            print("STOP - File already exists!")
            exit()
        print(f'saving to {smooth_file_lr}')
        prediction_utils.save_to_h5(smooth_file_lr, 'u', u_temp_smoothing)
        prediction_utils.save_to_h5(smooth_file_lr, 'v', v_temp_smoothing)
        prediction_utils.save_to_h5(smooth_file_lr, 'w', w_temp_smoothing)

        # add to file orginial data such as mask, venc etc.
        merge_data_to_h5(smooth_file_lr, hr_file)

    #-----------HR smoothing------------ 
    if True: 
        u_temp_smoothing = temporal_smoothing_box_function_toeger(hr_u, t_range, smoothing)
        v_temp_smoothing = temporal_smoothing_box_function_toeger(hr_v, t_range, smoothing)
        w_temp_smoothing = temporal_smoothing_box_function_toeger(hr_w, t_range, smoothing)

        if os.path.exists(smooth_file_hr):
            print("STOP - File already exists!")
            exit()
        print(f'saving to {smooth_file_hr}')
        prediction_utils.save_to_h5(smooth_file_hr, 'u', u_temp_smoothing)
        prediction_utils.save_to_h5(smooth_file_hr, 'v', v_temp_smoothing)
        prediction_utils.save_to_h5(smooth_file_hr, 'w', w_temp_smoothing)

        merge_data_to_h5(smooth_file_hr, hr_file)



    