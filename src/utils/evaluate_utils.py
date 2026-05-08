
from utils.colors import *
import os
import sys
import numpy as np
import time
import h5py
import scipy
from scipy.ndimage import binary_erosion
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
import pandas as pd

sys.path.insert(0, '../src')

#---------------LOAD DATA--------------------
def load_lossdata(file):
    print('Loading ', file)
    df_loss = pd.read_csv(file,  on_bad_lines='warn', skiprows = 4, skipfooter = 4, header = 5, engine = 'python')
    return df_loss

def load_vel_interpolation(dict_interpolation,lr,  filename, method, mask, vel_colnames, savefile = True):
    """ Load interpolated velocity fields from file if not existing, calculate and save them to h5 file"""
    
    if not os.path.isfile(filename):
        print("Interpolation file does not exist - calculate interpolation and save files")
        print("Save interpolation files to: ", filename)

        for vel in vel_colnames:
            if method == 'linear':
                dict_interpolation[vel] = temporal_linear_interpolation(lr[vel], gt[vel].shape)
            elif method == 'cubic':
                dict_interpolation[vel] = temporal_cubic_interpolation(lr[vel], gt[vel].shape)
            elif method == 'NN':
                dict_interpolation[vel] = temporal_NN_interpolation(lr[vel], gt[vel].shape)
            elif method == 'sinc':
                dict_interpolation[vel] = temporal_sinc_interpolation_ndarray(lr[vel], gt[vel].shape)
            else:  
                raise ValueError(f'Interpolation method ({method}) not known')
            
            if savefile:
                prediction_utils.save_to_h5(filename, vel , dict_interpolation[vel], compression='gzip')

            dict_interpolation[f'{vel}_fluid'] = np.multiply(dict_interpolation[vel], mask)
    
    else:
        print("Load existing interpolation file")
        with h5py.File(filename, mode = 'r' ) as h5:
            for vel in vel_colnames:
                dict_interpolation[vel] = np.array(h5[vel]).squeeze()
    
                dict_interpolation[f'{vel}_fluid'] = np.multiply(dict_interpolation[vel], mask)
    
    return dict_interpolation

def load_velocity_data(filename, datadict, vel_colnames, load_mask = False):
    """ Load velocity data from h5 file and store in dictionary"""
    vel_colnames_save = ['u', 'v', 'w']
    # load h5 file
    with h5py.File(filename, mode = 'r' ) as h5:
        for vel, v in zip(vel_colnames, vel_colnames_save):
            datadict[v] = np.asarray(h5[vel])
        if load_mask:
            datadict['mask'] = np.asarray(h5['mask']).squeeze()
            datadict["mask"][np.where(h5["mask"] !=0)] = 1

            if len(datadict['mask'].shape) == 3:
                print('Create static temporal mask for model, mask shape before:', datadict["mask"].shape )
                datadict['mask'] = create_dynamic_mask(datadict['mask'], h5['u'].shape[0])

    return datadict


#--------------IMG UTILITIES----------------
# Crop mask to match desired shape * downsample
def crop_gt(gt, desired_shape):
    '''
    This function crops the ground truth to match the desired shape.
    It assumes that the ground truth is a 4D array.'''
    crop = np.array(gt.shape) - np.array(desired_shape)
    if crop[0]:
        gt = gt[1:-1,:,:]
    if crop[1]:
        gt = gt[:,1:-1,:]
    if crop[2]:
        gt = gt[:,:,1:-1]
    if len(crop)>3 and crop[3]:
        gt = gt[:,:,:, 1:-1]
        
    return gt

def random_indices3D(mask, n):
    '''
    This function generates random indices in a 3D mask based on a given threshold.
    It assumes that the mask is a 3D array.
    The function randomly selects 'n' samples from the mask that have values greater than the threshold.
    It returns the x, y, and z indices of the selected samples.
    '''

    assert(len(mask.shape)==3) # Ensure that the mask is 3D

    mask_threshold = 0.9
    sample_pot = np.where(mask > mask_threshold)  # Find indices where mask values are greater than the threshold
    rng = np.random.default_rng()

    # Sample 'n' random samples without replacement
    sample_idx = rng.choice(len(sample_pot[0]), replace=False, size=n)

    # Get the x, y, and z indices of the selected samples
    x_idx = sample_pot[0][sample_idx]
    y_idx = sample_pot[1][sample_idx]
    z_idx = sample_pot[2][sample_idx]
    return x_idx, y_idx, z_idx


def create_dynamic_mask(mask, n_frames):
    '''
    from static mask create dynamic mask of shape (n_frames, h, w, d)
    '''
    assert(len(mask.shape) == 3), " shape: " + str(mask.shape) # shape of mask is assumed to be 3 dimensional
    print('Create static temporal mask.')
    return np.repeat(np.expand_dims(mask, 0), n_frames, axis=0)


def get_fluid_region_points(data, binary_mask):
    '''
    reshapes input such that we get data of form frames, n_fluid_points
    '''
    assert len(binary_mask.shape) == 1 #remove this function from here
    if len(binary_mask.squeeze().shape) ==3:
            binary_mask = create_dynamic_mask(binary_mask, data.shape[0])
        
    points_in_mask = np.where(binary_mask !=0)
    return data[:, points_in_mask[1], points_in_mask[2], points_in_mask[3]].reshape(data.shape[0], -1)

def get_fluid_region_points_frame(data_frame, binary_mask):
    '''
    returns flattened array with all the fluid boundary points in 3D data frame
    '''
    assert len(binary_mask.shape) == 3 # mask should be 3D
    assert len(data_frame.shape) == 3 # data should be 3D
        
    return data_frame[np.where(binary_mask != 0 )].flatten()

def normalize_to_0_1(data):
    """
    Normalize data to 0-1 range
    """
    return (np.array(data, dtype=float)- np.min(data))/(np.max(data)-np.min(data))

def check_and_normalize(img):
        if img.dtype == np.uint8:
                return np.asarray(img, dtype=float)/255

        return (img - np.min(img))/(np.max(img) - np.min(img))

def get_2Dslice(data, frame, axis, slice_idx):
    '''
    Returns 2D slice from 4D data with given time frame, axis and index
    '''
    if len(data.squeeze().shape) == 3:
        frame = 0
        print("Only one frame available: take first frame.")
        if len(data.shape) == 3:
            data = np.expand_dims(data, 0)
        
    if axis == 0 :
        return data[frame, slice_idx, :, :]
    elif axis == 1:
        return data[frame, :, slice_idx, :]
    elif axis == 2:
        return data[frame, :, :, slice_idx]
    else: 
        print("Invalid axis! Axis must be 0, 1 or 2")

def get_indices(frames, axis, slice_idx):
    '''
    Returns indices for 4D data with given time frames, axis and index
    '''
    if axis == 0 :
        return np.index_exp[frames, slice_idx, :, :]
    elif axis == 1:
        return np.index_exp[frames, :, slice_idx, :]
    elif axis == 2:
        return np.index_exp[frames, :, :, slice_idx]
    else: 
        print("Invalid axis! Axis must be 0, 1 or 2")

def get_spatial_indices(axis, slice_idx):
    '''
    Returns indices for 3D data with given axis and index
    '''
    if axis == 0 :
        return np.index_exp[slice_idx, :, :]
    elif axis == 1:
        return np.index_exp[:, slice_idx, :]
    elif axis == 2:
        return np.index_exp[:, :, slice_idx]
    else: 
        print("Invalid axis! Axis must be 0, 1 or 2")

def crop_center(img,croph,cropw):
    '''
    Crop center of image given size of the new image
    '''
    # from https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image
    x,y = img.shape
    starth = y//2-(croph//2)
    startw = x//2-(cropw//2)    
    return img[starth:starth+cropw,startw:startw+cropw]

def get_boundaries(binary_mask):
    '''
    returns boundary and core mask given a binary mask. 
    Note that mask values should be 0 and 1
    '''

    if (len(binary_mask.shape)==3):
        print("Create boundary mask for 3D data")
        core_mask = binary_erosion(binary_mask)
        boundary_mask = binary_mask - core_mask
        return boundary_mask, core_mask
    
    core_mask       = np.zeros_like(binary_mask)
    boundary_mask   = np.zeros_like(binary_mask)

    for t in range(binary_mask.shape[0]):
        core_mask[t, :, :, :] = binary_erosion(binary_mask[t, :, :, :])
        boundary_mask[t, :, :, :] = binary_mask[t, :, :, :] - core_mask[t, :, :, :]

        
    assert(np.linalg.norm(binary_mask - (boundary_mask + core_mask))== 0 ) # check that there is no overlap between core and boundary mask
    return boundary_mask, core_mask


# ---------------EVALUATION METRICS---------- 

def calculate_relative_error_np(u_pred, v_pred, w_pred, u_hi, v_hi, w_hi, binary_mask):
    '''
    Relative error calculation for numpy arrays as in training
    '''
    # if epsilon is set to 0, we will get nan and inf
    epsilon = 1e-5

    u_diff = np.square(u_pred - u_hi)
    v_diff = np.square(v_pred - v_hi)
    w_diff = np.square(w_pred - w_hi)

    diff_speed = np.sqrt(u_diff + v_diff + w_diff)
    actual_speed = np.sqrt(np.square(u_hi) + np.square(v_hi) + np.square(w_hi)) 

    # actual speed can be 0, resulting in inf
    relative_speed_loss = diff_speed / (actual_speed + epsilon)
    
    # Make sure the range is between 0 and 1
    relative_speed_loss = np.clip(relative_speed_loss, 0., 1.)

    # Apply correction, only use the diff speed if actual speed is zero
    condition = np.not_equal(actual_speed, np.array(0.))
    corrected_speed_loss = np.where(condition, relative_speed_loss, diff_speed)

    multiplier = 1e4 # round it so we don't get any infinitesimal number
    corrected_speed_loss = np.round(corrected_speed_loss * multiplier) / multiplier
    # print(corrected_speed_loss)
    
    # Apply mask
    # binary_mask_condition = (mask > threshold)
    binary_mask_condition = np.equal(binary_mask, 1.0)          
    corrected_speed_loss = np.where(binary_mask_condition, corrected_speed_loss, np.zeros_like(corrected_speed_loss))
    
    # Calculate the mean from the total non zero accuracy, divided by the masked area
    # reduce first to the 'batch' axis
    mean_err = np.sum(corrected_speed_loss, axis=(1,2,3)) / (np.sum(binary_mask, axis=(0,1,2)) + 1) 

    # now take the actual mean
    # mean_err = tf.reduce_mean(mean_err) * 100 # in percentage
    mean_err = mean_err * 100

    return mean_err

def calculate_relative_error_normalized(u_pred, v_pred, w_pred, u_hi, v_hi, w_hi, binary_mask):
    '''
    Calculate relative error with tanh as normalization
    '''

    # if epsilon is set to 0, we will get nan and inf
    epsilon = 1e-5 

    if len(binary_mask.squeeze().shape) ==3:
        print('Create temporal mask to calculate relative error')
        binary_mask = create_dynamic_mask(binary_mask, u_hi.shape[0])

    u_diff = np.square(u_pred - u_hi)
    v_diff = np.square(v_pred - v_hi)
    w_diff = np.square(w_pred - w_hi)

    diff_speed = np.sqrt(u_diff + v_diff + w_diff)
    actual_speed = np.sqrt(np.square(u_hi) + np.square(v_hi) + np.square(w_hi)) 

    print("max/min before tanh", np.max(diff_speed / (actual_speed + epsilon)), np.min(diff_speed / (actual_speed + epsilon)))

    # actual speed can be 0, resulting in inf
    #relative_speed_loss = np.arctan(diff_speed / (actual_speed + epsilon))
    relative_speed_loss = np.tanh(diff_speed / (actual_speed + epsilon))
    print("max/min after tanh", np.max(relative_speed_loss), np.min(relative_speed_loss))
    # Make sure the range is between 0 and 1
    #relative_speed_loss = np.clip(relative_speed_loss, 0., 1.)

    # Apply correction, only use the diff speed if actual speed is zero
    condition = np.not_equal(actual_speed, np.array(0.)) # chnages from condition = np.not_equal(actual_speed, np.array(tf.constant(0.)))
    corrected_speed_loss = np.where(condition, relative_speed_loss, diff_speed)

    multiplier = 1e4 # round it so we don't get any infinitesimal number
    corrected_speed_loss = np.round(corrected_speed_loss * multiplier) / multiplier
    # print(corrected_speed_loss)
    
    # Apply mask
    # binary_mask_condition = (mask > threshold)
    binary_mask_condition = np.equal(binary_mask, 1.0)          
    corrected_speed_loss = np.where(binary_mask_condition, corrected_speed_loss, np.zeros_like(corrected_speed_loss))
    # print(found_indexes)
    # Calculate the mean from the total non zero accuracy, divided by the masked area
    # reduce first to the 'batch' axis
    mean_err = np.sum(corrected_speed_loss, axis=(1,2,3)) / (np.sum(binary_mask, axis=(1, 2, 3)) + 1) 

    # now take the actual mean
    # mean_err = tf.reduce_mean(mean_err) * 100 # in percentage
    mean_err = mean_err * 100

    return mean_err


def cosine_similarity(u_hr, v_hr, w_hr, u_sr, v_sr, w_sr, eps =1e-10):
    """
    cosine similarity calculation. 1 if simlar direction, 0 if orthogonal, -1 if opposite direction
    """

    return (u_hr*u_sr + v_hr*v_sr + w_hr*w_sr)/(l2_norm(u_hr, v_hr, w_hr)* (l2_norm(u_sr, v_sr, w_sr) )+ eps)

def l2_norm(u, v, w):
    return np.sqrt(u**2 + v**2 + w**2)

def calculate_rmse(pred,gt, binary_mask, return_std= False, mask_threshold = 0.5):
    '''
    Calculate root mean squared error between prediction and ground truth for each frame
    i.e. rmse(t) = sqrt((pred - gt)**2/N), where N number of point in fluid region
    If return_std is set to true, the standard deviation of (pred - gt)**2 is returned as well
    '''
    if len(pred.shape)==3: pred = np.expand_dims(pred, 0)
    if len(gt.shape)==3:  gt = np.expand_dims(gt, 0)
    
    if len(binary_mask.squeeze().shape) ==3:
        print('Create temporal mask for RMSE caculation', binary_mask.shape, pred.shape, gt.shape )
        binary_mask = create_dynamic_mask(binary_mask, pred.shape[0])
        print('Reshaped to', binary_mask.shape)
    
    binary_mask[np.where(binary_mask > mask_threshold)] = 1
    binary_mask[np.where(binary_mask <= mask_threshold)] = 0
    
    rmse = np.sqrt(np.mean((pred - gt)**2, axis = (1, 2, 3), where=binary_mask.astype(bool)))
    
    if return_std:
        
        var = np.std((pred - gt)**2, axis = (1, 2, 3), where=binary_mask.astype(bool))
        return rmse, var
    return rmse

def calculate_pointwise_error(u_pred, v_pred, w_pred, u_hi, v_hi, w_hi, binary_mask):
    '''
    Returns a relative pointswise error and a dictionary with the absolute difference between prediction and ground truth
    '''
    # if epsilon is set to 0, we will get nan and inf
    epsilon = 1e-5
    if len(binary_mask.squeeze().shape) ==3:
        binary_mask = create_dynamic_mask(binary_mask, u_hi.shape[0])

    u_diff = np.square(u_pred - u_hi)
    v_diff = np.square(v_pred - v_hi)
    w_diff = np.square(w_pred - w_hi)

    diff_speed = np.sqrt(u_diff + v_diff + w_diff)
    actual_speed = np.sqrt(np.square(u_hi) + np.square(v_hi) + np.square(w_hi)) 

    # actual speed can be 0, resulting in inf
    relative_speed_loss = diff_speed / (actual_speed + epsilon)
    
    # Make sure the range is between 0 and 1
    #relative_speed_loss = np.clip(relative_speed_loss, 0., 1.)

    idx_mask = np.where(binary_mask == 0)
    relative_speed_loss[:,idx_mask[1], idx_mask[2], idx_mask[3]] = 0

    error_absolut = {} 
    error_absolut["u"] = np.sqrt(u_diff)
    error_absolut["v"] = np.sqrt(v_diff)
    error_absolut["w"] = np.sqrt(w_diff)
    error_absolut["speed"] = np.abs(np.sqrt(np.square(u_pred) + np.square(v_pred) + np.square(w_pred)) - actual_speed)

    return relative_speed_loss, error_absolut

def calculate_mean_speed(u_hi, v_hi, w_hi, binary_mask):
    '''
    Calculate mean speed of given values. Assumption: Values are in m/sec and mean speed returned in cm/sec
    Important: Set values of u, v, w outside of fluid region to zero 
    '''
    if len(binary_mask.squeeze().shape) ==3:
        binary_mask = create_dynamic_mask(binary_mask, u_hi.shape[0])


    u_hi = np.multiply(u_hi, binary_mask)
    v_hi = np.multiply(v_hi, binary_mask)
    w_hi = np.multiply(w_hi, binary_mask)

    speed = np.sqrt(np.square(u_hi) + np.square(v_hi) + np.square(w_hi))
    mean_speed = np.sum(speed, axis=(1,2,3)) / (np.sum(binary_mask, axis=(1, 2, 3)) + 1) *100
    return mean_speed

def sigmoid(x):
    '''
    Sigmoid function
    '''
    return 1 / (1 + np.exp(-x))

def signaltonoise_fluid_region(data, mask):
    assert len(data.shape) == 3 # look at three dimensional data
    norm_data = normalize_to_0_1(data)
    return signaltonoise(norm_data[np.where(mask==1)], norm_data[np.where(mask ==0)], axis=0)

def signaltonoise(fluid_region, non_fluid_region, axis=0, ddof=0):
    '''
    source: https://stackoverflow.com/questions/63177236/how-to-calculate-signal-to-noise-ratio-using-python
    '''
    m  = fluid_region.mean(axis)
    sd = non_fluid_region.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def signaltonoise_db(a, axis=0, ddof=0):
    return 20*np.log10(np.abs(signaltonoise(a, axis, ddof)))


def compare_mask_and_velocitymask(u_hi, v_hi, w_hi, binary_mask):
    '''
    Compares the given binary mask with created mask to the nonzero values of u, v and w and returns the overlap mask
    '''

    # create binary mask for u, v and w
    overlap_mask= np.zeros_like(u_hi)
    overlap_mask[np.where(u_hi != 0)] = 1
    overlap_mask[np.where(v_hi != 0)] = 1
    overlap_mask[np.where(w_hi != 0)] = 1

    mask = overlap_mask.copy()    
    extended_mask = np.repeat(binary_mask, 3, axis=0)

    # 0: overlap of zero values, 1: overlap of nonzero values, 2: created mask has value and binary mask odes not, 3: binary mask has value and created mask does not
    overlap_mask[np.where((extended_mask == 0) & (overlap_mask == 1))] = 2
    overlap_mask[np.where((extended_mask == 1) & (overlap_mask == 0))] = 3
    
    return overlap_mask, mask[0].squeeze()

def calculate_k_R2( pred, gt, binary_mask):
    '''
    Calculate r^2 and k in fluid region with line y = kx+m
    Note that this takes a 3D data frame as input, i.e. it only considers a single frame
    '''
    assert len(pred.shape) == 3 # this should be a 3D data frame

    # extract values within fluid region for prediction (SR) and ground truth (HR)
    sr_vals = get_fluid_region_points_frame(pred,binary_mask)
    hr_vals = get_fluid_region_points_frame(gt,binary_mask )

    # calculate linear regression parameters with scipy
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(hr_vals, sr_vals)
    return slope,  r_value**2

def calculate_k_R2_timeseries(pred, gt, binary_mask):

    '''
    Calculate r^2 and k in fluid region with line y = kx+m
    Note that this takes a 4D data frame as input, i.e. it considers multiple frames
    '''
    assert len(pred.shape) == 4 # this should be a 4D data frame
    N_frames = pred.shape[0]
    k = np.zeros(N_frames)
    r2 = np.zeros(N_frames)

    for t in range(N_frames):
        # extract values within fluid region for prediction (SR) and ground truth (HR)
        sr_vals = get_fluid_region_points_frame(pred[t], binary_mask[t])
        hr_vals = get_fluid_region_points_frame(gt[t], binary_mask[t])

        # calculate linear regression parameters with scipy
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(hr_vals, sr_vals)

        k[t] = slope
        r2[t] = r_value**2
    return k, r2


def calculate_flow_profile(velocity, binary_mask,spacing = [2.0, 2.0, 2.0]):
    assert len(velocity.shape) == 3
    assert len(binary_mask.shape) == 2
    # assert velocity.shape == binary_mask.shape # check that the shapes are the same
    area = np.sum(binary_mask)*spacing[0]*spacing[1]*spacing[2]
    mean_vel = np.mean(velocity, where=binary_mask.astype(bool), axis=(1, 2))
    return mean_vel*area


#---------------PLOTTING-------------------

def bland_altman_plot(pred_vel, gt_vel, mask, timepoint, p=0.1, y_lim = None, centered_ylim = False,  set_background = True, ax=None, fontsize=15, save_as=None):
       
    if mask.ndim ==3:
        mask_comp = mask
    elif mask.ndim == 4:
        mask_comp = mask[timepoint]
    else:
        print("Mask should be 3D or 4D")
    

    # Apply the mask to select only fluid region
    pred_masked = pred_vel[timepoint][np.where(mask_comp > 0.5)].flatten()
    gt_masked   = gt_vel[timepoint][np.where(mask_comp > 0.5)].flatten()

    # Compute mean and difference
    mean_values = (pred_masked + gt_masked) / 2
    diff_values = gt_masked-pred_masked
    
    # Mean and ±1.96 SD lines
    mean_diff = np.mean(diff_values)
    std_diff = np.std(diff_values)
    upper_limit = mean_diff + 1.96 * std_diff
    lower_limit = mean_diff - 1.96 * std_diff

    random_selection = np.random.randint(0, len(mean_values), size=int(len(mean_values) * p))
    mean_values_plot = mean_values[random_selection]
    diff_values_plot = diff_values[random_selection]

    # Plot on provided ax or create a new figure if ax is None
    if ax is None:
        ax = plt.gca()
        single_plot = True
        y_limits_max = 0
    else:
        single_plot = False
        _, y_limits_max = ax.get_ylim()
    
    # get y limits
    if set_background:
        if y_lim is None:
            if centered_ylim:
                # get max
                y_lim_up = np.max(np.abs(diff_values_plot)) * 1.1
                ax.set_ylim(-y_lim_up, y_lim_up)
            else:
                # Set y-limits to ensure space for the shading
                ax.set_ylim(np.min(diff_values_plot) * 1.1, np.maximum(np.max(diff_values_plot) * 1.1, y_limits_max))
        else:
            y_lim_down = y_lim[0]
            y_lim_up = y_lim[1]
            ax.set_ylim(y_lim_down, y_lim_up)

        # Shading the area above the mean difference line in light gray
        ax.axhspan(mean_diff, ax.get_ylim()[1], color='lightgray', alpha=0.5)
        print(f'Y lim is {ax.get_ylim()} and is set by given ylim {y_lim} and is centered {centered_ylim}')
    # Scatter plot
    ax.scatter(mean_values_plot, diff_values_plot, s=12, color='black')

    # Plot lines and labels
    ax.axhline(mean_diff, color='red', linestyle='--', label="Mean Difference")
    ax.axhline(upper_limit, color='gray', linestyle='--')
    ax.axhline(lower_limit, color='gray', linestyle='--')

    # Text annotations for ±1.96 SD
    x_left, x_right = ax.get_xlim()
    ax.text(x_right * 0.9, upper_limit, f'{upper_limit:.2f}', color='gray', ha='right', va='bottom', fontsize=fontsize)
    ax.text(x_right * 0.9, lower_limit * 1.05, f'{lower_limit:.2f}', color='gray', ha='right', va='top', fontsize=fontsize)

    # Labels, ticks, and save
    ax.set_xlabel(r'0.5(V$_{HR}$ + V$_{SR}$) [m/s]', fontsize=fontsize)
    if single_plot: ax.set_ylabel(r'V$_{HR}$ - V$_{SR}$ [m/s]', fontsize=fontsize)
    ax.locator_params(axis='x', nbins=3, tight=True)
    ax.locator_params(axis='y', nbins=3, tight=True)
    # ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    # ax.tick_params(axis='both', direction='in', length=4, width=0.5, labelsize=8, pad=2)
    # ax.minorticks_on()
    # ax.tick_params(which='minor', length=2, width=0.5)
    # ax.tick_params(axis='both', labelsize=fontsize//2, colo)
    ax.tick_params(axis='both', which='minor', color='lightgray', labelsize=fontsize//2 + 2, labelcolor='gray')
    ax.tick_params(axis='both', which='major', color='gray', labelcolor='gray', labelsize=fontsize//2 +2)
    for spine in ax.spines.values():
        spine.set_color("gray")       # Change the spine color to gray
        spine.set_linewidth(0.5) 
    if save_as is not None:
        plt.savefig(save_as, transparent=True)



def set_axis_properties(ax, xlabel='', ylabel='', title='', xlim=None, ylim=None, 
                        fontsize=18, tick_color='gray', label_color='black', tick_number=3, add_grid=False, col_index = 0, set_figure_box = False):
    """Configures common axis properties with options for tick, label colors, and a grid line at y=1."""
    ax.set_xlabel(xlabel, fontsize=fontsize, color=label_color)
    ax.set_title(title, fontsize=fontsize, color=label_color)
    
    # Set y-axis label only if it's the first column in the subplot row
    if ylabel and col_index == 0:
        ax.set_ylabel(ylabel, fontsize=fontsize, color=label_color)

    # Tick settings
    ax.tick_params(axis='both', which='major', labelsize=fontsize//2 + 2, color=tick_color, labelcolor=tick_color)
    ax.locator_params(axis='x', nbins=tick_number)
    ax.locator_params(axis='y', nbins=tick_number)
    
    # Grid line for k plot at y=1
    if add_grid:
        # ax.axhline(y=1, color='gray', linestyle='--', linewidth=1)
        ax.grid(True, which='major', color='lightgray', linestyle='--')

    # Set axis limits if provided
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    if set_figure_box:
        for spine in ax.spines.values():
            spine.set_color("gray")       # Change the spine color to gray
            spine.set_linewidth(0.5)


def plot_regression_points_new(ax, hr_vals, sr_vals, all_hr, all_sr, abs_max, direction, color='black', show_text=False, fontsize=18):
    """Plots regression line and points for a single axis."""
    x_range = np.linspace(-abs_max, abs_max, 100)
    corr_line, text = get_corr_line_and_r2(all_hr, all_sr, x_range)
    
    ax.plot(x_range, x_range, color='grey', label='diagonal')
    ax.plot(x_range, corr_line, 'k--', color = KI_colors['Plum'])
    ax.scatter(hr_vals, sr_vals, s=30, c=color, label='core voxels')
    set_axis_properties(ax, xlabel=r"V$_{HR}$ [m/s]", ylabel=r"V$_{SR}$ [m/s]", title=direction, xlim=(-abs_max, abs_max), ylim=(-abs_max, abs_max), fontsize=fontsize, set_figure_box=True)
    
    if show_text:
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=fontsize, verticalalignment='top')

def plot_k_r2_values(ax, t_range, k_vals, r2_vals, peak_frame, min_val, max_val, k_label, r2_label, color_k='black', color_r2='purple', fontsize=18, only_k = False):
    """Plots k and R2 values over time on a single axis."""
    ax.set_ylim([min_val, max_val])
    ax.scatter([t_range[peak_frame]], [k_vals[peak_frame]], color='darkgrey', label='peak synthesized frame')
    if not only_k:
        ax.plot(t_range, r2_vals,  label=r2_label, color=color_r2)
        ax.scatter([t_range[peak_frame]], [r2_vals[peak_frame]], color='darkgrey')
    ax.plot(t_range, k_vals,'--', label=k_label, color=color_k)
    col_index = 1 if ('x' in k_label.lower() or 'u' in k_label.lower()) else 0
    if only_k:
        set_axis_properties(ax, xlabel='time [s]', ylabel=k_label, fontsize=fontsize, add_grid=True, col_index=col_index, xlim=(t_range[0], t_range[-1])) #ylabel=r'$k/R^2$',
    else:
        set_axis_properties(ax, xlabel='time [s]', fontsize=fontsize, add_grid=True, col_index=col_index, set_figure_box=True) #ylabel=r'$k/R^2$',
        # ax.legend(loc='lower right', fontsize=fontsize)


def plot_correlation_nobounds(gt, prediction, frame_idx,color_points = 'black',show_text = False, save_as = None, figsize = (7, 7), fontsize = 16, show_title = True):
    '''
    Plot correlation plot between ground truth and prediction at a given frame
    '''
    # set percentage of how many random points are used
    p = 0.1
    mask_threshold = 0.5

    mask = np.asarray(gt['mask']).squeeze()

    # if mask static make dynamic mask 
    if len(mask.shape) == 3:
        mask = create_dynamic_mask(mask, prediction['u'].shape[0])
    
    # threshold mask
    mask[np.where(mask > mask_threshold)] = 1 

    idx_core = np.where(mask[frame_idx] == 1)

    # get random indices for core to plot a subset of the points
    x_idx, y_idx, z_idx = random_indices3D((mask)[frame_idx], n=int(p*np.count_nonzero(mask[frame_idx])))

    # Get velocity values in all directions
    # HR
    hr_u = np.asarray(gt['u'][frame_idx])
    hr_u_rnd = hr_u[x_idx, y_idx, z_idx]
    hr_v = np.asarray(gt['v'][frame_idx])
    hr_v_rnd = hr_v[x_idx, y_idx, z_idx]
    hr_w = np.asarray(gt['w'][frame_idx])
    hr_w_rnd = hr_w[x_idx, y_idx, z_idx]

    # SR 
    sr_u = np.asarray(prediction['u'][frame_idx])
    sr_u_rnd = sr_u[x_idx, y_idx, z_idx]
    sr_v = np.asarray(prediction['v'][frame_idx])
    sr_v_rnd = sr_v[x_idx, y_idx, z_idx]
    sr_w = np.asarray(prediction['w'][frame_idx])
    sr_w_rnd = sr_w[x_idx, y_idx, z_idx]

    def plot_regression_points(hr_vals, sr_vals, all_hr, all_sr, direction = 'u'):
        N = 100
        # make sure that the range is the same for all plots and make square range
        x_range = np.linspace(-abs_max, abs_max, N)
        
        corr_line, text = get_corr_line_and_r2(all_hr, all_sr, x_range)

        # plot linear correlation line and parms
        if show_text:
            plt.gca().text(0.05, 0.95, text,transform=plt.gca().transAxes, fontsize=fontsize, verticalalignment='top')
        plt.plot(x_range, x_range, color= 'grey', label = 'diagonal line')
        plt.plot(x_range, corr_line, 'k--')
        plt.scatter(hr_vals, sr_vals, s=30, c=[color_points], label = 'core voxels')
        
        if show_title:
            plt.title(direction, fontsize=fontsize)
        plt.xlabel(r"V$_{HR}$ [m/s]", fontsize=fontsize)
        plt.ylabel(r"V$_{SR}$ [m/s]", fontsize=fontsize)
        # lgnd = plt.legend(loc = 'lower right', markerscale=2.0, fontsize=10)
        plt.ylim(-abs_max, abs_max)
        plt.xlim(-abs_max, abs_max)
        plt.locator_params(axis='y', nbins=3)
        plt.locator_params(axis='x', nbins=3)
        plt.tick_params(axis='y', labelsize = fontsize//2+2, labelcolor='gray', color = 'gray')
        plt.tick_params(axis='x', labelsize = fontsize//2+2, labelcolor='gray', color = 'gray')

        #set tickcolor to gray
        # plt.tick_params(axis='both', which='major', color='gray', labelcolor='gray')

        #maek box gray
        for spine in plt.gca().spines.values():
            spine.set_color("gray")
            spine.set_linewidth(0.5)
        plt.tight_layout()
        # lgnd.legendHandles[1]._sizes = [30]
        # lgnd.legendHandles[2]._sizes = [30]

    
    print("Plotting correlation lines...")
    min_vals = np.min([np.min(sr_u_rnd), np.min(sr_v_rnd), np.min(sr_w_rnd)])
    max_vals = np.max([np.max(sr_u_rnd), np.max(sr_v_rnd), np.max(sr_w_rnd)])
    abs_max  = np.max([np.abs(min_vals), np.abs(max_vals)])

    plt.close()

    # plot regression line for Vx, Vy and Vz
    plt.figure(figsize=figsize)
    plot_regression_points(hr_u_rnd, sr_u_rnd, hr_u[idx_core], sr_u[idx_core],  direction=r'$V_x$')
    plt.tight_layout()
    if save_as is not None: plt.savefig(f"{save_as}_LRXplot.png")
    
    plt.clf()
    plt.figure(figsize=figsize)
    plot_regression_points(hr_v_rnd, sr_v_rnd, hr_v[idx_core], sr_v[idx_core], direction=r'$V_y$')
    plt.tight_layout()
    if save_as is not None: plt.savefig(f"{save_as}_LRYplot.png")

    plt.clf()
    plt.figure(figsize=figsize)
    plot_regression_points(hr_w_rnd, sr_w_rnd, hr_w[idx_core], sr_w[idx_core],  direction=r'$V_z$')
    plt.tight_layout()
    if save_as is not None: plt.savefig(f"{save_as}_LRZplot.png")

    plt.clf()
    save_subplots = True
    
    plt.close()
    # plot Vx, Vy and Vz in subplots
    if save_subplots: 
        fig = plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plot_regression_points(hr_u_rnd, sr_u_rnd, hr_u[idx_core], sr_u[idx_core], direction=r'$V_x$')
        plt.subplot(1, 3, 2)
        plot_regression_points(hr_v_rnd, sr_v_rnd, hr_v[idx_core], sr_v[idx_core], direction=r'$V_y$')
        plt.subplot(1, 3, 3)
        plot_regression_points(hr_w_rnd, sr_w_rnd, hr_w[idx_core], sr_w[idx_core],  direction=r'$V_z$')
        plt.tight_layout()
        if save_as is not None: plt.savefig(f"{save_as}_all_LRXYZ_subplots.png", transparent=True)
        
    return fig


def plot_correlation_nobounds_new(gt, prediction, frame_idx, color_points='black', show_text=False, save_as=None, figsize=(12, 4), fontsize=16):
    '''
    Plot correlation plot between ground truth and prediction at a given frame
    '''
    # Parameters for random sampling and mask threshold
    p = 0.1
    mask_threshold = 0.5

    # Create mask if needed
    mask = np.asarray(gt['mask']).squeeze()
    if len(mask.shape) == 3:
        mask = create_dynamic_mask(mask, prediction['u'].shape[0])
    mask[mask > mask_threshold] = 1
    idx_core = np.where(mask[frame_idx] == 1)

    # Get random indices for plotting a subset of points
    x_idx, y_idx, z_idx = random_indices3D(mask[frame_idx], n=int(p * np.count_nonzero(mask[frame_idx])))

    # Extract velocity components
    hr_velocities = {dir: np.asarray(gt[dir][frame_idx])[x_idx, y_idx, z_idx] for dir in ['u', 'v', 'w']}
    sr_velocities = {dir: np.asarray(prediction[dir][frame_idx])[x_idx, y_idx, z_idx] for dir in ['u', 'v', 'w']}
    
    abs_max = max(abs(np.min(list(sr_velocities.values()))), abs(np.max(list(sr_velocities.values()))))

    # Define function to plot each subplot with consistent formatting
    def plot_regression_points(ax, hr_vals, sr_vals, all_hr, all_sr, direction):
        x_range = np.linspace(-abs_max, abs_max, 100)
        corr_line, text = get_corr_line_and_r2(all_hr, all_sr, x_range)
        
        ax.plot(x_range, x_range, color='gray', label='diagonal line')  # Diagonal line
        ax.plot(x_range, corr_line, 'k--')  # Regression line
        ax.scatter(hr_vals, sr_vals, s=30, c=color_points, label='core voxels')

        ax.set_title(direction, fontsize=fontsize)
        ax.set_xlim(-abs_max, abs_max)
        ax.set_ylim(-abs_max, abs_max)
        ax.locator_params(axis='y', nbins=3)
        ax.locator_params(axis='x', nbins=3)
        ax.set_xlabel(r"V$_{HR}$ [m/s]", fontsize=fontsize)
        ax.spines['top'].set_color('gray')
        ax.spines['bottom'].set_color('gray')
        ax.spines['left'].set_color('gray')
        ax.spines['right'].set_color('gray')
        ax.tick_params(axis='both', which='minor', color='lightgray', labelsize=fontsize//2, labelcolor='gray')
        ax.tick_params(axis='both', which='major', color='gray', labelcolor='gray', labelsize=fontsize//2)
        ax.tick_params(axis='y', labelleft=True)
        
        if show_text:
            ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=fontsize, verticalalignment='top')

    # Create a single figure with subplots for each velocity component
    fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=True)
    directions = ['u', 'v', 'w']
    labels = [r'$V_x$', r'$V_y$', r'$V_z$']
    
    # Plot for each component
    for i, direction in enumerate(directions):
        plot_regression_points(
            axs[i],
            hr_velocities[direction],
            sr_velocities[direction],
            np.asarray(gt[direction][frame_idx])[idx_core],
            np.asarray(prediction[direction][frame_idx])[idx_core],
            labels[i]
        )
        if i == 0:
            axs[i].set_ylabel(r"V$_{SR}$ [m/s]", fontsize=fontsize)
    
    
    # Adjust layout and save figure
    plt.tight_layout()
    # plt.tight_layout(rect=[0.05, 0.05, 1, 1])  # Adjust to avoid overlapping with shared labels
    if save_as:
        plt.savefig(f"{save_as}_all_LRXYZ_subplots.png", transparent=True)

    plt.show()
    return fig


def get_corr_line_and_r2(hr_vals, sr_vals, x_range):
        '''
        Returns correlation line and text for plot
        '''
        z = np.polyfit(hr_vals, sr_vals, 1)
        corr_line = np.poly1d(z)(x_range)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(hr_vals, sr_vals)
        text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r_value**2:0.4f}$"
        
        return corr_line, text


def plot_correlation(gt, prediction, bounds, frame_idx,color_b = KI_colors['Plum'], save_as = None):
    '''
    Plot correlation plot between ground truth and prediction at a given frame
    '''
    # set percentage of how many random points are used
    p = 0.1
    mask_threshold = 0.6

    mask = np.asarray(gt['mask']).squeeze()

    # if mask static make dynamic mask 
    if len(mask.shape) == 3:
        mask = create_dynamic_mask(mask, prediction['u'].shape[0])
    
    # threshold mask
    mask[np.where(mask > mask_threshold)] = 1 

    # get indices of core and boundary
    idx_core = np.where((mask[frame_idx]-bounds[frame_idx]) == 1)
    idx_bounds = np.where(bounds[frame_idx] == 1)

    # get random indices for core and boundary to plot a subset of the points
    # core (subtract bounds from mask such that mask only contains core points)
    x_idx, y_idx, z_idx = random_indices3D((mask-bounds)[frame_idx], n=int(p*np.count_nonzero(mask[frame_idx])))
    # boundary 
    x_idx_b, y_idx_b, z_idx_b = random_indices3D(bounds[frame_idx], n=int(p*np.count_nonzero(bounds[frame_idx])))
    
    # Get velocity values in all directions
    # HR
    hr_u = np.asarray(gt['u'][frame_idx])
    hr_u_core = hr_u[x_idx, y_idx, z_idx]
    hr_u_bounds = hr_u[x_idx_b, y_idx_b, z_idx_b]
    hr_v = np.asarray(gt['v'][frame_idx])
    hr_v_core = hr_v[x_idx, y_idx, z_idx]
    hr_v_bounds = hr_v[x_idx_b, y_idx_b, z_idx_b]
    hr_w = np.asarray(gt['w'][frame_idx])
    hr_w_core = hr_w[x_idx, y_idx, z_idx]
    hr_w_bounds = hr_w[x_idx_b, y_idx_b, z_idx_b]

    # SR 
    sr_u = np.asarray(prediction['u'][frame_idx])
    sr_u_vals = sr_u[x_idx, y_idx, z_idx]
    sr_u_bounds = sr_u[x_idx_b, y_idx_b, z_idx_b]
    sr_v = np.asarray(prediction['v'][frame_idx])
    sr_v_vals = sr_v[x_idx, y_idx, z_idx]
    sr_v_bounds = sr_v[x_idx_b, y_idx_b, z_idx_b]
    sr_w = np.asarray(prediction['w'][frame_idx])
    sr_w_vals = sr_w[x_idx, y_idx, z_idx]
    sr_w_bounds = sr_w[x_idx_b, y_idx_b, z_idx_b]

    def plot_regression_points(hr_vals, sr_vals, hr_vals_bounds, sr_vals_bounds,all_hr, all_sr, all_hr_bounds, all_sr_bounds, direction = 'u'):
        N = 100
        # make sure that the range is the same for all plots and make square range
        x_range = np.linspace(-abs_max, abs_max, N)
        
        corr_line, text = get_corr_line_and_r2(all_hr, all_sr, x_range)
        corr_line_bounds, text_bounds = get_corr_line_and_r2(all_hr_bounds, all_sr_bounds, x_range)

        # plot linear correlation line and parms
        plt.gca().text(0.05, 0.95, text,transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.gca().text(0.05, 0.82, text_bounds,transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', color=color_b)
        plt.plot(x_range, x_range, color= 'grey', label = 'diagonal line')
        plt.plot(x_range, corr_line_bounds, '--', color = color_b)
        plt.plot(x_range, corr_line, 'k--')
        plt.scatter(hr_vals, sr_vals, s=1.2, c=["black"], label = 'core voxels')
        plt.scatter(hr_vals_bounds, sr_vals_bounds, s=1.2, c=[color_b], label = 'boundary voxels')
        
        plt.title(direction)
        plt.xlabel("V HR (m/s)")
        plt.ylabel("V prediction (m/s)")
        lgnd = plt.legend(loc = 'lower right', markerscale=2.0, fontsize=10)
        plt.ylim(-abs_max, abs_max)
        plt.xlim(-abs_max, abs_max)
        plt.locator_params(axis='y', nbins=3)
        plt.locator_params(axis='x', nbins=3)
        lgnd.legendHandles[1]._sizes = [30]
        lgnd.legendHandles[2]._sizes = [30]

    
    print("Plotting correlation lines...")

    min_vals = np.min([np.min(sr_u_vals), np.min(sr_v_vals), np.min(sr_w_vals)])
    max_vals = np.max([np.max(sr_u_vals), np.max(sr_v_vals), np.max(sr_w_vals)])
    abs_max = np.max([np.abs(min_vals), np.abs(max_vals)])
    print('min/max/abs max', min_vals, max_vals, abs_max)

    plt.clf()

    # plot regression line for Vx, Vy and Vz
    plt.figure(figsize=(5, 5))
    plot_regression_points(hr_u_core, sr_u_vals, hr_u_bounds, sr_u_bounds,hr_u[idx_core], sr_u[idx_core], hr_u[idx_bounds], sr_u[idx_bounds],direction=r'$V_x$')
    plt.tight_layout()
    if save_as is not None: plt.savefig(f"{save_as}_LRXplot.svg")
    
    plt.clf()
    plt.figure(figsize=(5, 5))
    plot_regression_points(hr_v_core, sr_v_vals, hr_v_bounds, sr_v_bounds,hr_v[idx_core], sr_v[idx_core], hr_v[idx_bounds], sr_v[idx_bounds],direction=r'$V_y$')
    plt.tight_layout()
    if save_as is not None: plt.savefig(f"{save_as}_LRYplot.svg")

    plt.clf()
    plt.figure(figsize=(5, 5))
    plot_regression_points(hr_w_core, sr_w_vals, hr_w_bounds, sr_w_bounds,hr_w[idx_core], sr_w[idx_core], hr_w[idx_bounds], sr_w[idx_bounds], direction=r'$V_z$')
    plt.tight_layout()
    if save_as is not None: plt.savefig(f"{save_as}_LRZplot.svg")

    plt.clf()
    save_subplots = True

    # plot Vx, Vy and Vz in subplots
    if save_subplots: 
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plot_regression_points(hr_u_core, sr_u_vals, hr_u_bounds, sr_u_bounds,hr_u[idx_core], sr_u[idx_core], hr_u[idx_bounds], sr_u[idx_bounds],direction=r'$V_x$')
        plt.subplot(1, 3, 2)
        plot_regression_points(hr_v_core, sr_v_vals, hr_v_bounds, sr_v_bounds,hr_v[idx_core], sr_v[idx_core], hr_v[idx_bounds], sr_v[idx_bounds],direction=r'$V_y$')
        plt.subplot(1, 3, 3)
        plot_regression_points(hr_w_core, sr_w_vals, hr_w_bounds, sr_w_bounds,hr_w[idx_core], sr_w[idx_core], hr_w[idx_bounds], sr_w[idx_bounds], direction=r'$V_z$')
        plt.tight_layout()
        if save_as is not None: plt.savefig(f"{save_as}_LRXYZ_subplots.pdf")
    

def show_temporal_development_line(gt, lr, pred, mask, axis, indices, save_as = "Temporal_development.png"):
    mask[np.where(mask !=0)] = 1
    gt = np.multiply(gt, mask)
    lr = np.multiply(lr, mask)
    pred = np.multiply(pred, mask)

    def get_line(data):
        #returns line in 4D data over all time steps
        x,y = indices
        if axis == 1:
            return data[:, :, x, y]
        elif axis ==2:
            return data[:, x, :, y]
        elif axis ==3:
            return data[:, x,  y, :]
        else:
            print("Invalid axis: Please choose axis 1, 2, 3")

    prediction      = get_line(pred).transpose()
    ground_truth    = get_line(gt).transpose()
    low_resolution  = get_line(lr).transpose()
    print('prediction shape', prediction.shape)

    min_v = np.min([np.min(prediction), np.min(ground_truth), np.min(low_resolution)])
    max_v = np.max([np.max(prediction), np.max(ground_truth), np.max(low_resolution)])


    plt.subplot(1, 3, 1)
    plt.imshow(low_resolution, vmin = min_v, vmax = max_v, cmap='jet')
    plt.title("LR")
    plt.xlabel('t')
    plt.yticks([])

    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth, vmin = min_v, vmax = max_v, cmap='jet')
    plt.title("GT")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 3, 3)
    plt.imshow(prediction, vmin = min_v, vmax = max_v, cmap='jet')
    plt.title("SR")
    plt.xticks([])
    plt.yticks([])

    plt.savefig(save_as,bbox_inches='tight')


def show_quiver( u, v, w, mask,frame,save_as = "Quiver_3DFlow.png"):
    x_len, y_len, z_len = u.shape
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Make the grid
    x, y, z = np.meshgrid(np.arange(y_len),np.arange(x_len),np.arange(z_len))
    
    set_to_zero = 0.9
    if len(mask.shape) ==3: mask = create_dynamic_mask(mask, u.shape[0])

    x_idx, y_idx, z_idx = random_indices3D(mask[frame], int(np.count_nonzero(mask[frame])*set_to_zero))
    u[x_idx, y_idx, z_idx] = 0
    v[x_idx, y_idx, z_idx] = 0
    w[x_idx, y_idx, z_idx] = 0
    
    cropx = cropy = cropz = 10
    startx = x_len//2-(cropx//2)
    starty = y_len//2-(cropy//2)    
    startz = z_len//2-(cropz//2)
    u = u[startx:startx+cropx, starty:starty+cropy,startz:startz+cropz] 
    v = v[startx:startx+cropx, starty:starty+cropy,startz:startz+cropz] 
    w = w[startx:startx+cropx, starty:starty+cropy,startz:startz+cropz] 

    x = x[startx:startx+cropx, starty:starty+cropy,startz:startz+cropz] 
    y = y[startx:startx+cropx, starty:starty+cropy,startz:startz+cropz] 
    z = z[startx:startx+cropx, starty:starty+cropy,startz:startz+cropz] 

    ax.quiver(x, y, z, u, v, w, length=0.3, normalize=True, color=plt.cm.viridis([200, 50, 100, 200, 200, 50, 50, 100, 100]))
    fig.savefig(f'{save_as}_frame{frame}.png')
    plt.show()
    plt.clf()

def make_3D_quiver_plot(data,mask,  frame, set_to_zero=0.9):
        
        u_quiver = data['u'][frame].copy() 
        v_quiver = data['v'][frame].copy() 
        w_quiver = data['w'][frame].copy() 

        x_len, y_len, z_len = data['u'].shape[1::]

        # ev = np.array([1, 0, 0])
        # angles = np.arccos(np.dot(ev, [u, v, w]) / (np.linalg.norm(ev) * np.linalg.norm(v2)))
        # angles = np.multiply

        # Make the grid
        x, y, z = np.meshgrid(np.arange(y_len),np.arange(x_len),np.arange(z_len))
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # Erase (i.e. set zo zero) multiple points to increase visibility
        x_idx, y_idx, z_idx = random_indices3D(mask[frame], int(np.count_nonzero(mask[frame])*set_to_zero))
        u_quiver[x_idx, y_idx, z_idx] = 0
        v_quiver[x_idx, y_idx, z_idx] = 0
        w_quiver[x_idx, y_idx, z_idx] = 0

        # crop image from middle
        cropx = cropy = cropz = 25
        startx = x_len//2-(cropx//2)
        starty = y_len//2-(cropy//2)    
        startz = z_len//2-(cropz//2)
        u_quiver = u_quiver[startx:startx+cropx, starty:starty+cropy,startz:startz+cropz] 
        v_quiver = v_quiver[startx:startx+cropx, starty:starty+cropy,startz:startz+cropz] 
        w_quiver = w_quiver[startx:startx+cropx, starty:starty+cropy,startz:startz+cropz] 

        x =x[startx:startx+cropx, starty:starty+cropy,startz:startz+cropz] 
        y =y[startx:startx+cropx, starty:starty+cropy,startz:startz+cropz] 
        z =z[startx:startx+cropx, starty:starty+cropy,startz:startz+cropz]

        u_new = u_quiver[np.where(u_quiver != 0)]
        v_new = v_quiver[np.where(u_quiver != 0)]
        w_new = w_quiver[np.where(u_quiver != 0)]

        x_new = x[np.where(u_quiver!=0)] 
        y_new = y[np.where(u_quiver!=0)] 
        z_new = z[np.where(u_quiver!=0)]

        u = u_new.ravel()
        v = v_new.ravel()
        w = w_new.ravel()

        # Color by magnitude
        c = np.sqrt(u_new**2+ v_new**2+ w_new**2) #np.arctan2(w_new, u_new)
        # Flatten and normalize

        c = (c.ravel() - c.min()) / c.ptp()
        # Repeat for each body line and two head lines
        c = np.concatenate((c, np.repeat(c, 2)))
        # Colormap
        c = plt.cm.jet(c)


        ax.quiver(x_new, y_new, z_new, u_new, v_new, w_new, length=10, normalize=False,  pivot='middle', color = c)
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.zlabel('z')
        # plt.show()


def plot_rmse(gt, pred, comparison_lst, name_comparison,save_as,colors_comp = None,  figsize = (5, 5)):
    '''
    Plot RMSE for Vx, Vy and Vz within and outside the fluid region
    '''

    reverse_mask = 1 - gt['mask'] 

    fig = plt.figure(figsize = figsize)
    plt.plot(calculate_rmse(pred['u'], gt['u'], gt['mask']), label = r'$V_x$ fluid region', color = KI_colors['Grey'])
    plt.plot(calculate_rmse(pred['v'], gt['v'], gt['mask']), label = r'$V_y$ fluid region', color = KI_colors['LightBlue'])
    plt.plot(calculate_rmse(pred['w'], gt['w'], gt['mask']), label = r'$V_z$ fluid region', color = KI_colors['Plum'])

    plt.plot(calculate_rmse(pred['speed'], gt['speed'], gt['mask']), label = 'speed')
    plt.plot(calculate_rmse(pred['u'], gt['u'], reverse_mask), label = r'$V_x$ non-fluid region',linestyle = '--',  color = KI_colors['Grey'])
    plt.plot(calculate_rmse(pred['v'], gt['v'], reverse_mask), label = r'$V_y$ non-fluid region',linestyle = '--',  color = KI_colors['LightBlue'])
    plt.plot(calculate_rmse(pred['w'], gt['w'], reverse_mask), label = r'$V_z$ non-fluid region',linestyle = '--',  color = KI_colors['Plum'])

    for i, (comp_res, name) in enumerate(zip(comparison_lst, name_comparison)):
        if colors_comp is not None: 
            color = colors_comp[i]
        else:
            color = None
        for vel, plt_name in zip(['u', 'v', 'w'], [r'$V_x$', r'$V_y$', r'$V_z$']):
            RMSE_comp = calculate_rmse(comp_res[vel],  gt[vel], gt['mask'] )
            plt.plot(RMSE_comp, label= rf'{plt_name}_{name}', color = color)


    plt.ylabel('RMSE')
    plt.xlabel('Frame')
    plt.title('RMSE ')
    plt.legend(loc = 'upper left')

    plt.tight_layout()
    plt.savefig(save_as,bbox_inches='tight')
    return fig

def plot_relative_error(gt, pred, comparison_lst, name_comparison, save_as, colors_comp = None,  figsize = (10, 3)):

  
    fig = plt.figure(figsize=figsize)
    rel_error = calculate_relative_error_normalized(pred['u'], pred['v'], pred['w'], gt['u'], gt['v'], gt['w'], gt['mask'])

    plt.plot(rel_error, label = 'SR', color = KI_colors['Blue'])

    for i, (comp_res, name) in enumerate(zip(comparison_lst, name_comparison)):
        if colors_comp is not None: 
            color = colors_comp[i]
        else:
            color = None
        RE_comp = calculate_relative_error_normalized(comp_res['u'], comp_res['v'], comp_res['w'], gt['u'], gt['v'], gt['w'], gt['mask'] )
        plt.plot(RE_comp, label= name, color = color)

    # plt.plot(50*np.ones(len(rel_error)), 'k:')
    plt.xlabel("Frame", fontsize = 14)
    plt.title("Relative error")
    plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
    plt.ylabel("Relative error (%)", fontsize = 14)
    #plt.ylim((0, 50))
    plt.legend(loc = 'upper left', fontsize = 14)
    plt.savefig(save_as,bbox_inches='tight')
    return fig


def plot_mean_speed(gt, pred, lr, comparison_lst, name_comparison,save_as,colors_comp = None,  figsize= (10, 3)):
    fig = plt.figure(figsize=figsize)

    gt_meanspeed    = calculate_mean_speed(gt['u'], gt['v'], gt['w'], gt['mask'])
    lr_meanspeed    = calculate_mean_speed(lr['u'], lr['v'], lr['w'], lr['mask'])
    pred_meanspeed  = calculate_mean_speed(pred['u'], pred['v'], pred['w'], gt['mask'])

    t_hr_range = np.linspace(0, 1, len(gt_meanspeed), endpoint=True)
    t_lr_range = np.linspace(0, 1, len(lr_meanspeed), endpoint=True)
    t_sr_range = np.linspace(0, 1, len(pred_meanspeed), endpoint=True)

    plt.plot(t_hr_range, gt_meanspeed, '.-',label ='High resolution', color = 'black')
    plt.plot(t_sr_range, pred_meanspeed,'.-', label= '4DFlowNet', color = KI_colors['Blue'])
    plt.plot(t_lr_range,  lr_meanspeed,'.-',  label = 'Low resolution', color = KI_colors['Green'])
    for i, (comp_res, name) in enumerate(zip(comparison_lst, name_comparison)):
        if colors_comp is not None: 
            color = colors_comp[i]
        else:
            color = None
        comp_meanspeed = calculate_mean_speed(comp_res['u'], comp_res['v'],gt['mask'] )
        plt.plot(comp_meanspeed, label= name, color = color)

    plt.xlabel("Frame")
    plt.ylabel("Mean speed (cm/s)")
    plt.legend(loc = 'upper left')
    plt.title('Mean speed')
    plt.savefig(save_as,bbox_inches='tight')
    return fig



def plot_qual_comparison_peak(u_hr, v_hr, w_hr, u_sr, v_sr, w_sr, mask_cube, abserror_cube, comparison_lst, comparison_names, timepoint, min_v, max_v, include_error=False, figsize=(10, 10), save_as=None, colormap = 'viridis', fontsize=16):
    
    print(f"Plotting qualitative comparison of flow at timepoint {timepoint}...")

    cmap = colormap

    # Determine the number of columns and rows
    N_columns = 2 + len(comparison_lst)
    N_rows = 3
    if include_error:
        N_columns += 1

    fig, axes = plt.subplots(nrows=N_rows, ncols=N_columns, constrained_layout=True, figsize=figsize)

    # Set dynamic min and max values for color normalization if not provided
    if min_v is None or max_v is None:
        if len(mask_cube.shape) == 2:
            mask_cube = np.repeat(mask_cube[np.newaxis, ...], u_hr.shape[0])
            min_v = np.quantile(u_hr.flatten(), 0.01)
            max_v = np.quantile(u_hr.flatten(), 0.99)
        else:
            min_v = np.quantile(u_hr[mask_cube != 0].flatten(), 0.01)
            max_v = np.quantile(u_hr[mask_cube != 0].flatten(), 0.99)

    if include_error:
        min_rel_error = np.min(np.array(abserror_cube))
        max_rel_error = np.max(np.array(abserror_cube))

    
    for j, col_name in enumerate(['HR', 'SR'] + comparison_names):
        # Assign the correct data based on the column (HR, SR, or comparison)
        if col_name == 'HR':
            img_data_u, img_data_v, img_data_w = u_hr, v_hr, w_hr
        elif col_name == 'SR':
            img_data_u, img_data_v, img_data_w = u_sr, v_sr, w_sr
        else:
            img_data_u, img_data_v, img_data_w = comparison_lst[j - 2]

        # Plot u, v, w for each column (HR, SR, comparison)
        for i, (img_data, label) in enumerate(zip([img_data_u, img_data_v, img_data_w], [r'$V_x$', r'$V_y$', r'$V_z$'])):
            ax = axes[i, j]
            im = ax.imshow(img_data, vmin=min_v, vmax=max_v, cmap=cmap)
            if j == 0:
                ax.set_ylabel(label, fontsize=fontsize)
            if i == 0:
                ax.set_title(col_name, fontsize=fontsize)
            ax.set_xticks([])
            ax.set_yticks([])

    # Plot absolute error if included
    if include_error:
        for i in range(N_rows):
            ax = axes[i, -1]
            err_img = ax.imshow(abserror_cube[i], vmin=min_rel_error, vmax=max_rel_error, cmap=cmap)
            if i == 0:
                ax.set_title("Abs. Error", fontsize=fontsize)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.colorbar(err_img, ax=axes[:, -1], aspect=15, pad=0.01, label='Abs. error [m/s]')

    # Add the main color bar for the velocity
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), aspect=15, pad=0.01, label='Velocity [m/s]')
    # set ticks size 
    cbar.ax.tick_params(labelsize=fontsize//2)
    cbar.set_label('Velocity [m/s]', fontsize=fontsize//2 + 4)	
    # Set number of ticks
    cbar.ax.locator_params(nbins=3)
    # plt.tight_layout()
    print(f'Qualitative comparison saved under {save_as}')
    if save_as is not None:
        plt.savefig(save_as, bbox_inches='tight', transparent=True)


def plot_spatial_comparison(low_res, ground_truth, prediction, frame_idx = 9, axis=1, slice_idx = 50):

    if slice_idx% 2 != 0 : print("Slice index should be even!")

    patch = [40, 40]

    vel_colnames = ['u', 'v', 'w', 'div_x']#, 'divergence_y', 'divergence_z']
    vel_plotnames = ['Vx', r'Vy', r'Vz']
    n = 1

    #calculate divergence
    ground_truth['div_x'], ground_truth['div_y'], ground_truth['div_z'] = np.asarray(calculate_divergence(ground_truth['u'], ground_truth['v'], ground_truth['w']))
    low_res['div_x'], low_res['div_y'], low_res['div_z'] = np.asarray(calculate_divergence(low_res['u'], low_res['v'], low_res['w']))
    prediction['div_x'], prediction['div_y'], prediction['div_z'] = np.asarray(calculate_divergence(prediction['u'], prediction['v'], prediction['w']))


    for i, vel in enumerate(vel_colnames):
        slice_lr = get_2Dslice(low_res[vel], frame_idx, axis, slice_idx//2)
        slice_gt = get_2Dslice(ground_truth[vel], frame_idx, axis, slice_idx)
        slice_sr = get_2Dslice(prediction[vel], frame_idx, axis, slice_idx)

        slice_lr = crop_center(slice_lr, patch[0]//2, patch[1]//2)
        slice_gt = crop_center(slice_gt, patch[0], patch[1])
        slice_sr = crop_center(slice_sr, patch[0], patch[1])

        max_v = np.max(np.stack((np.resize(slice_lr, slice_gt.shape), slice_gt, slice_sr)))
        min_v = np.min(np.stack((np.resize(slice_lr, slice_gt.shape), slice_gt, slice_sr)))
        
        plt.subplot(len(vel_colnames), 4, n)
        plt.imshow(slice_lr, vmin = min_v, vmax = max_v, cmap='jet')
        if i == 0: plt.title("LR")
        plt.xticks([])
        plt.yticks([])
        plt.ylabel(vel)

        plt.subplot(len(vel_colnames), 4, n+1)
        plt.imshow(slice_gt, vmin = min_v, vmax = max_v, cmap='jet')
        if i == 0: plt.title("HR")
        plt.xticks([])
        plt.yticks([])

        plt.subplot(len(vel_colnames), 4, n+2)
        plt.imshow(slice_sr, vmin = min_v, vmax = max_v, cmap='jet')
        if i == 0: plt.title("4DFlowNet")
        plt.xticks([])
        plt.yticks([])

        #TODO real linear interpolation
        plt.subplot(len(vel_colnames), 4, n+3)
        plt.imshow(slice_lr, vmin = min_v, vmax = max_v, cmap='jet', interpolation='bilinear')
        if i == 0: plt.title("bilinear")
        plt.xticks([])
        plt.yticks([])
        
        # plt.subplot(len(vel_colnames), 5, n+4)
        # plt.imshow(slice_lr, vmin = min_v, vmax = max_v, cmap='jet', interpolation='bicubic')
        # if i == 0: plt.title("bicubic")
        # plt.xticks([])
        # plt.yticks([])

        plt.colorbar()
        n+=4

    #fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("4DFlowNet/results/plots/Comparison_prediction.png")


#TODO:merge with other function? 
def comparison_plot_slices_over_time(gt_cube,lr_cube, mask_cube, comparison_lst, comparison_name, timepoints, axis, idx,min_v, max_v, save_as = "Qualitative_Frame_comparison.png", figsize=(10,10)):
    """ Qualitative comparison of different network models over timeframe for one velocity direction"""

    def row_based_idx(num_rows, num_cols, idx):
        return np.arange(1, num_rows*num_cols + 1).reshape((num_rows, num_cols)).transpose().flatten()[idx-1]
    
    T = 2 + len(comparison_lst)
    N = len(timepoints)

    fig, axes = plt.subplots(nrows=T, ncols=N, constrained_layout=True, figsize=figsize)

    i = 1
    idxs = get_indices(timepoints, axis, idx)
    gt_cube = gt_cube[idxs]
    mask_cube = mask_cube[idxs]
    
    # find same range to plot velocity images to
    min_v = np.quantile(gt_cube[np.where(mask_cube !=0)].flatten(), 0.01)
    max_v = np.quantile(gt_cube[np.where(mask_cube !=0)].flatten(), 0.99)

        
    for j,t in enumerate(timepoints):
        gt_slice = gt_cube[j]

        # low resolution
        lr_slice = np.zeros_like(gt_slice)
            
        plt.subplot(T, N, row_based_idx(T, N, i))
        if t%2 == 0:
            lr_slice = lr_cube[get_indices(t//2, axis=axis, slice_idx=idx )]
            plt.imshow(lr_slice, vmin = min_v, vmax = max_v, cmap='viridis', aspect='auto')
            if i == 1: plt.ylabel("LR")
            plt.xticks([])
            plt.yticks([])
        plt.title('frame '+ str(t))
        plt.xticks([])
        plt.yticks([])
        i +=1

        # ground truth
        plt.subplot(T, N, row_based_idx(T, N, i))
        plt.imshow(gt_slice, vmin = min_v, vmax = max_v, cmap='viridis', aspect='auto')
        if i == 2: plt.ylabel("HR")
        plt.xticks([])
        plt.yticks([])
        i +=1	

        # plot model predictions
        for comp, name in zip(comparison_lst, comparison_name):
            
            plt.subplot(T, N, row_based_idx(T, N, i))
            im = plt.imshow(comp[idxs][j], vmin = min_v, vmax = max_v, cmap='viridis', aspect='auto')
            if i-1 == (i-1)%T: plt.ylabel(name)
            plt.xticks([])
            plt.yticks([])
            i +=1
        
    fig.colorbar(im, ax=axes.ravel().tolist(), aspect = 50, label = 'velocity (m/s)')
    plt.savefig(save_as,bbox_inches='tight' )


# adapted with LR is 'not acquired'
def plot_qual_comparsion(gt_cube,lr_cube,  pred_cube,mask_cube, abserror_cube, comparison_lst, comparison_name, timepoints, 
                         min_v, max_v, include_error = False,  figsize = (10, 10), save_as = "Qualitative_frame_seq.png", fontsize_lr = 8, colormap = 'viridis', center_vmin_vmax = False, aspect_colorbar = 15):
    def row_based_idx(num_rows, num_cols, idx):
        return np.arange(1, num_rows*num_cols + 1).reshape((num_rows, num_cols)).transpose().flatten()[idx-1]

    print(f"Plotting qualitative comparison of timepoints {timepoints}...")

    # gt_cube = np.abs(gt_cube)
    # lr_cube = np.abs(lr_cube)
    # pred_cube = np.abs(pred_cube)
    # for i in range(len(comparison_lst)):
    #     comparison_lst[i] = np.abs(comparison_lst[i]) 

    ups_factor = 2
    cmap = colormap
    fontsize = 16

    T = 3 + len(comparison_lst)
    N = len(timepoints)
    if include_error: T += 1

    # fig = plt.figure(figsize=figsize)
    fig, axes = plt.subplots(nrows=T, ncols=N, constrained_layout=True, figsize=figsize, facecolor='w')
    if min_v is None or max_v is None:
        if len(mask_cube.shape) == 2:
            print(mask_cube.shape)
            mask_cube = np.repeat(mask_cube[np.newaxis, ...], gt_cube.shape[0])
            
            min_v = np.quantile(gt_cube.flatten(), 0.01)
            max_v = np.quantile(gt_cube.flatten(), 0.99)
        else:
            min_v = np.quantile(gt_cube[np.where(mask_cube != 0)].flatten(), 0.01)
            max_v = np.quantile(gt_cube[np.where(mask_cube != 0)].flatten(), 0.99)
    
    if center_vmin_vmax:
        max_v = np.max([abs(min_v), abs(max_v)])
        min_v = -max_v

    if include_error:
        min_rel_error = np.min(np.array(abserror_cube))
        max_rel_error = np.max(np.array(abserror_cube))

    dsize=(gt_cube.shape[1],gt_cube.shape[2])
    bar_width=1
    text = 'Not acquired'
    empty_data = np.full(dsize, fill_value=0.75, dtype=float)
    for i in range(-dsize[0], dsize[1], bar_width*2):
        empty_data += 0.5*np.eye(dsize[0], dsize[1], k=i, dtype=float)


    img_cnt = 1
    for j,t in enumerate(timepoints):

        plt.subplot(T, N, row_based_idx(T, N, img_cnt))
        if t%ups_factor == 0:
            lr_slice = lr_cube[j//2]
            plt.imshow(lr_slice, vmin = min_v, vmax = max_v, cmap=cmap,)
            if img_cnt == 1: plt.ylabel("LR", fontsize = fontsize)

        else:
            plt.imshow(empty_data, cmap='gray',vmin=0, vmax=1)
            plt.text(dsize[1] / 2, dsize[0] / 2, text, color='black', fontsize=fontsize_lr, ha='center', va='center', multialignment='center', fontweight='bold')
            if img_cnt == 1: plt.ylabel("LR", fontsize = fontsize)
 
        # plt.title('frame '+ str(t))
        # plt.xticks([])
        # plt.yticks([])
        plt.axis('off')
        img_cnt +=1
        plt.subplot(T, N, row_based_idx(T, N, img_cnt))
        plt.imshow(gt_cube[j, :, :], vmin = min_v, vmax = max_v, cmap=cmap,)
        if img_cnt == 2: plt.ylabel("HR", fontsize = fontsize)
        # plt.xticks([])
        # plt.yticks([])
        plt.axis('off')
        img_cnt +=1
        plt.subplot(T, N, row_based_idx(T, N, img_cnt))
        im = plt.imshow(pred_cube[j, :, :], vmin = min_v, vmax = max_v, cmap=cmap,)
        if img_cnt == 3: plt.ylabel("SR", fontsize = fontsize)
        # plt.xticks([])
        # plt.yticks([])
        plt.axis('off')


        for comp, name in zip(comparison_lst, comparison_name):
            img_cnt +=1
            plt.subplot(T, N, row_based_idx(T, N, img_cnt))
            im = plt.imshow(comp[j, :, :], vmin = min_v, vmax = max_v, cmap=cmap, )
            if img_cnt-1 == (img_cnt-1)%T: plt.ylabel(name, fontsize = fontsize)
            # plt.xticks([])
            # plt.yticks([])
            plt.axis('off')
        

        img_cnt +=1
        if include_error:
            plt.subplot(T, N, row_based_idx(T, N, img_cnt))
            err_img = plt.imshow(abserror_cube[j, :, :],vmin=min_rel_error, vmax=max_rel_error, cmap=cmap,)
            if img_cnt-1 == (img_cnt-1)%T: plt.ylabel("abs. error", fontsize = fontsize)
            # plt.xticks([])
            # plt.yticks([])
            plt.axis('off')
            if t == timepoints[-1]:
                fig.colorbar(err_img, ax = axes[-1], aspect = aspect_colorbar,pad=0.01, label = 'abs. error [m/s]')

            img_cnt +=1

    if include_error:
        cbar = fig.colorbar(im, ax=axes.ravel()[:-N].tolist(), aspect = aspect_colorbar,pad=0.01, label = 'velocity [m/s]')
    else:
        # cbar = fig.colorbar(im, ax=axes.ravel().tolist(), aspect = 23,pad=0.01, label = 'velocity [m/s]')
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), aspect = aspect_colorbar,pad=0.01, label = 'velocity [m/s]')

    cbar.set_label('velocity [m/s]', fontsize=fontsize)#//2+ 4
    cbar.locator = ticker.MaxNLocator(nbins=4)  # Set the maximum number of ticks
    cbar.update_ticks()
    cbar.outline.set_visible(False)
    # cbar.ax.tick_params(labelsize=fontsize//2 + 2)
    cbar.ax.tick_params(labelsize=fontsize//2 + 4)
    print(f'Qualitative comparison saved under {save_as}')
    plt.savefig(save_as,bbox_inches='tight', transparent=True)


def plot_slices_over_time(gt_cube,lr_cube,  mask_cube, rel_error_cube, comparison_lst, comparison_name, timepoints, axis, idx,min_v, max_v,exclude_rel_error = True, save_as = "Frame_comparison.png", figsize = (30,20)):
    def row_based_idx(num_rows, num_cols, idx):
        return np.arange(1, num_rows*num_cols + 1).reshape((num_rows, num_cols)).transpose().flatten()[idx-1]

    
    T = 3 + len(comparison_lst)  #len(timepoints)
    N = len(timepoints)  #4 + len(comparison_lst)
    print(T, N)
    if exclude_rel_error: T -=1
    print(T, N)

    # fig = plt.figure(figsize=(10,10))
    fig, axes = plt.subplots(nrows=T, ncols=N, constrained_layout=True, figsize=figsize)

    i = 1
    idxs = get_indices(timepoints, axis, idx)
    gt_cube = gt_cube[idxs]
    mask_cube = mask_cube[idxs]
    
    # pred_cube = pred_cube[idxs]
    #lr = lr[idxs]

    min_v = np.quantile(gt_cube[np.where(mask_cube !=0)].flatten(), 0.01)
    max_v = np.quantile(gt_cube[np.where(mask_cube !=0)].flatten(), 0.99)

    if not exclude_rel_error:
        rel_error_slices =[get_2Dslice(rel_error_cube, t, axis, idx) for t in timepoints]
        min_rel_error = np.min(np.array(rel_error_slices))
        max_rel_error = np.max(np.array(rel_error_slices))
    for j,t in enumerate(timepoints):
        
        gt_slice = gt_cube[j]
        # pred_slice = pred_cube[j]

        lr_slice = np.zeros_like(gt_slice)
        if t%2 == 0: lr_slice = get_2Dslice(lr_cube, t//2, axis=axis, slice_idx=idx )
        plt.subplot(T, N, row_based_idx(T, N, i))

        if t%2 == 0:
            plt.imshow(lr_slice, vmin = min_v, vmax = max_v, cmap='viridis', aspect='auto')
            if i == 1: plt.ylabel("LR")
            plt.xticks([])
            plt.yticks([])
            
        plt.title('frame '+ str(t))
        plt.xticks([])
        plt.yticks([])        

        i +=1
        plt.subplot(T, N, row_based_idx(T, N, i))
        plt.imshow(gt_slice, vmin = min_v, vmax = max_v, cmap='viridis', aspect='auto')
        if i == 2: plt.ylabel("HR")
        plt.xticks([])
        plt.yticks([])

        # i +=1
        # plt.subplot(T, N, row_based_idx(T, N, i))
        # plt.imshow(pred_slice, vmin = min_v, vmax = max_v, cmap='viridis',aspect='auto')
        # if i == 3: plt.ylabel("4DFlowNet")
        # plt.xticks([])
        # plt.yticks([])


        for comp, name in zip(comparison_lst, comparison_name):
            i +=1
            plt.subplot(T, N, row_based_idx(T, N, i))
            im = plt.imshow(get_2Dslice(comp,t, axis=axis, slice_idx=idx), vmin = min_v, vmax = max_v, cmap='viridis', aspect='auto')
            if i-1 == (i-1)%T: plt.ylabel(name)
            plt.xticks([])
            plt.yticks([])

        if not exclude_rel_error:
            i +=1
            plt.subplot(T, N, row_based_idx(T, N, i))
            re_img = plt.imshow(get_2Dslice(rel_error_cube, t, axis, idx),vmin=min_rel_error, vmax=max_rel_error, cmap='viridis',aspect='auto')
            if i-1 == (i-1)%T: plt.ylabel("abs. error")
            plt.xticks([])
            plt.yticks([])
            if t == timepoints[-1]:
                plt.colorbar(re_img, ax = axes[-1], aspect = 10, label = 'abs. error ')

        i +=1

    fig.colorbar(im, ax=axes.ravel().tolist(), aspect = 60, label = 'velocity (m/s)')
    plt.savefig(save_as,bbox_inches='tight', transparent=True)
    # plt.tight_layout()

def plot_slices_over_time1(gt_cube,lr_cube,  mask_cube, rel_error_cube, comparison_lst, comparison_name, timepoints, axis, idx,min_v, max_v,exclude_rel_error = True, save_as = "Frame_comparison.png", figsize = (30,20)):
    def row_based_idx(num_rows, num_cols, idx):
        return np.arange(1, num_rows*num_cols + 1).reshape((num_rows, num_cols)).transpose().flatten()[idx-1]

    
    T = 3 + len(comparison_lst)#len(timepoints)
    N = len(timepoints)#4 + len(comparison_lst)
    print(T, N)
    if exclude_rel_error: T -=1
    print(T, N)

    # fig = plt.figure(figsize=(10,10))
    fig, axes = plt.subplots(nrows=T, ncols=N, constrained_layout=True, figsize=figsize)

    i = 1
    idxs = get_indices(timepoints, axis, idx)
    gt_cube = gt_cube[idxs]
    mask_cube = mask_cube[idxs]
    
    # pred_cube = pred_cube[idxs]
    #lr = lr[idxs]

    # min_v = np.quantile(gt_cube[np.where(mask_cube !=0)].flatten(), 0.01)
    # max_v = np.quantile(gt_cube[np.where(mask_cube !=0)].flatten(), 0.99)
    if not exclude_rel_error:
        rel_error_slices =[get_2Dslice(rel_error_cube, t, axis, idx) for t in timepoints]
        min_rel_error = np.min(np.array(rel_error_slices))
        max_rel_error = np.max(np.array(rel_error_slices))
    for j,t in enumerate(timepoints):
        
        gt_slice = gt_cube[j]
        # pred_slice = pred_cube[j]

        lr_slice = np.zeros_like(gt_slice)
        if t%2 == 0: lr_slice = get_2Dslice(lr_cube, t//2, axis=axis, slice_idx=idx )
        plt.subplot(T, N, row_based_idx(T, N, i))

        if t%2 == 0:
            plt.imshow(lr_slice, vmin = min_v, vmax = max_v, cmap='viridis', aspect='auto')
            if i == 1: plt.ylabel("LR")
            plt.xticks([])
            plt.yticks([])
            
        plt.title('frame '+ str(t))
        plt.xticks([])
        plt.yticks([])
        # plt.axis('off')
        

        i +=1
        plt.subplot(T, N, row_based_idx(T, N, i))
        plt.imshow(gt_slice, vmin = min_v, vmax = max_v, cmap='viridis', aspect='auto')
        if i == 2: plt.ylabel("HR")
        plt.xticks([])
        plt.yticks([])

        # i +=1
        # plt.subplot(T, N, row_based_idx(T, N, i))
        # plt.imshow(pred_slice, vmin = min_v, vmax = max_v, cmap='viridis',aspect='auto')
        # if i == 3: plt.ylabel("4DFlowNet")
        # plt.xticks([])
        # plt.yticks([])


        for comp, name in zip(comparison_lst, comparison_name):
            i +=1
            plt.subplot(T, N, row_based_idx(T, N, i))
            im = plt.imshow(get_2Dslice(comp,t, axis=axis, slice_idx=idx), vmin = min_v, vmax = max_v, cmap='viridis', aspect='auto')
            if i-1 == (i-1)%T: plt.ylabel(name)
            plt.xticks([])
            plt.yticks([])

        if not exclude_rel_error:
            i +=1
            plt.subplot(T, N, row_based_idx(T, N, i))
            re_img = plt.imshow(get_2Dslice(rel_error_cube, t, axis, idx),vmin=min_rel_error, vmax=max_rel_error, cmap='viridis',aspect='auto')
            if i-1 == (i-1)%T: plt.ylabel("abs. error")
            plt.xticks([])
            plt.yticks([])
            if t == timepoints[-1]:
                plt.colorbar(re_img, ax = axes[-1], aspect = 10, label = 'abs. error ')

        
        i +=1
    # plt.tight_layout()
    # plt.subplots_adjust(wspace=0, hspace=0)
    fig.colorbar(im, ax=axes.ravel().tolist(), aspect = 40, label = 'velocity (m/s)')
    plt.savefig(save_as,bbox_inches='tight' )

def show_timeframes(gt,lr,  pred,mask, rel_error, comparison_lst, comparison_name, timepoints, axis, idx,min_v, max_v,save_as = "Frame_comparison.png"):
    '''
    Plots a series of frames next to eachother to compare 
    '''
    plt.clf()
    T = len(timepoints)
    N = 3 + len(comparison_lst)
    i = 1
    for j,t in enumerate(timepoints):
        
        gt_slice = get_2Dslice(gt, t,  axis=axis, slice_idx=idx )
        pred_slice = get_2Dslice(pred, t, axis=axis, slice_idx=idx )

        lr_slice = np.zeros_like(gt_slice)
        if t%2 == 0: lr_slice = get_2Dslice(lr, t//2, axis=axis, slice_idx=idx )
        
        # min_v = np.min([np.min(pred_slice ), np.min(gt_slice), np.min(lr_slice)])
        # max_v = np.max([np.max(pred_slice), np.max(gt_slice), np.max(lr_slice)])  

        plt.subplot(T, N, i)

        if t%2 == 0:
            plt.imshow(lr_slice, vmin = min_v, vmax = max_v, cmap='viridis', aspect='auto')
            if i == 1: plt.title("LR")
            plt.xticks([])
            plt.yticks([])
            plt.ylabel('frame = '+ str(t))
            
        else:
            #plt.imshow(lr_slice, vmin = min_v, vmax = max_v, cmap='jet', aspect='auto')
            plt.axis('off')
        

        i +=1
        plt.subplot(T, N, i)
        plt.imshow(gt_slice, vmin = min_v, vmax = max_v, cmap='viridis', aspect='auto')
        if i == 2: plt.title("GT")
        plt.xticks([])
        plt.yticks([])

        i +=1
        plt.subplot(T, N, i)
        plt.imshow(pred_slice, vmin = min_v, vmax = max_v, cmap='viridis',aspect='auto')
        if i == 3: plt.title("4DFlowNet")
        plt.xticks([])
        plt.yticks([])
        for comp, name in zip(comparison_lst, comparison_name):
            i +=1
            plt.subplot(T, N, i)
            plt.imshow(get_2Dslice(comp,t, axis=axis, slice_idx=idx), vmin = min_v, vmax = max_v, cmap='viridis', aspect='auto')
            if i-1 == (i-1)%N: plt.title(name)
            plt.xticks([])
            plt.yticks([])

        i +=1

    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    #plt.tight_layout()
    plt.savefig(save_as,bbox_inches='tight')
    
    if False:
        plt.clf()
        mask[np.where(mask !=0)] = 1
        gt = np.multiply(gt, mask)
        lr = np.multiply(lr, mask)
        pred = np.multiply(pred, mask)

        N = 4
        i = 1
        for j,t in enumerate(timepoints):
            
            gt_slice = get_2Dslice(gt, t,  axis=axis, slice_idx=idx )
            pred_slice = get_2Dslice(pred, t, axis=axis, slice_idx=idx )
            err_slice = get_2Dslice(rel_error, t, axis=axis, slice_idx=idx )
            #dt_slice = get_slice(dt, t, axis=axis, slice_idx=idx )
            #print("shape dt:", dt.shape, dt_slice.shape, gt_slice.shape )

            lr_slice = np.zeros_like(gt_slice)
            if t%2 == 0: lr_slice = get_2Dslice(lr, t//2, axis= axis, slice_idx= idx )

            #min_v = np.min([np.min(pred_slice ), np.min(gt_slice), np.min(lr_slice)])
            #max_v = np.max([np.max(pred_slice), np.max(gt_slice), np.max(lr_slice)])  

            plt.subplot(T, N, i)
            if t%2 == 0:
                plt.imshow(lr_slice, vmin = min_v, vmax = max_v, cmap='viridis')
                if i == 1: plt.title("LR")
                plt.xticks([])
                plt.yticks([])
                plt.ylabel('frame = '+ str(t))
            else:
                plt.axis('off')

            i += 1
            plt.subplot(T, N, i)
            plt.imshow(gt_slice, vmin = min_v, vmax = max_v, cmap='viridis')
            if i == 2: plt.title("GT")
            plt.xticks([])
            plt.yticks([])

            i += 1
            plt.subplot(T, N, i)
            plt.imshow(pred_slice, vmin = min_v, vmax = max_v, cmap='viridis')
            if i == 3: plt.title("SR")
            plt.xticks([])
            plt.yticks([])

            i += 1
            plt.subplot(T, N, i)
            plt.imshow(err_slice, cmap='jet')
            if i == 4: plt.title("Relative error")
            plt.xticks([])
            plt.yticks([])

            # i +=1
            # plt.subplot(T, 5, i)
            # plt.imshow(dt_slice, cmap='jet')
            # if i == 5: plt.title("|dt|")
            # plt.xticks([])
            # plt.yticks([])

            # plt.colorbar()
            

            
            i +=1
            

        save_under = save_as[:-4]+ "_fluidregion.png"
        print("save with only fluid region visible", save_under)
        #plt.tight_layout()
        plt.subplots_adjust(hspace=0, wspace=0)
        plt.savefig(save_under,bbox_inches='tight')
        #plt.clf()

#-------- merge this functions---- above



def calculate_temporal_derivative(data, timestep=1):
    '''
    Calculate difference between two time frames and each voxel
    i.e. for u: dt u(t) = |u(t+1) - u(t)| / timestep
    '''

    n_frames = data.shape[0]
    dt =  np.zeros_like(data)
    for t in range(n_frames-1):
        dt[t, :, :, :] = (data[t+timestep, :, :, :] - data[t, :, :, :])/timestep

    dt = np.abs(dt) 
    
    return dt

def velocity_through_plane(idx_plane, data, normal, order_normal= [0, 1, 2]):
    ''' Returns the velocity through a plane, i.e. in direction of the normal, using projection on normal vector
    params:
        idx_plane: tuple of indices for the plane
        data: 4D data dictionary with keys u,v and w
        normal: normal vector of the plane
        order_normal: order of the normal vector, i.e. correspondece between cartesian plane and u, v, w (similar to paraview)
    '''
    N_frames = data['u'].shape[0]
    vx_in_plane = data['u'][:, idx_plane[0], idx_plane[1], idx_plane[2]].reshape(N_frames, -1)
    vy_in_plane = data['v'][:, idx_plane[0], idx_plane[1], idx_plane[2]].reshape(N_frames, -1)
    vz_in_plane = data['w'][:, idx_plane[0], idx_plane[1], idx_plane[2]].reshape(N_frames, -1)
    return vx_in_plane*normal[order_normal[0]]+ vy_in_plane*normal[order_normal[1]]+ vz_in_plane*normal[order_normal[2]]


def rmse_plane(idx_intersection_plane_fluid,normal,  data,gt,order_normal = [0, 1, 2],   label = '', color = 'black'):

    V_plane_pred    = velocity_through_plane(idx_intersection_plane_fluid, data, normal, order_normal = order_normal)
    V_plane_gt      = velocity_through_plane(idx_intersection_plane_fluid, gt, normal, order_normal = order_normal)

    rmse = np.sqrt(np.mean((V_plane_pred-V_plane_gt)**2, axis = 1))
    plt.plot(rmse,'.-', label = label, color = color)
    plt.xlabel('frame')
    plt.ylabel('RMSE')

def plot_max_speed_plane(idx_intersection_plane_fluid, data, normal, frames,order_normal = [0, 1, 2], label = '', color = 'black'):
    N_frames = data['u'].shape[0]

    # Velocity through plane
    V_plane = velocity_through_plane(idx_intersection_plane_fluid, data, normal, order_normal = order_normal)
    max_speed= np.max(V_plane, axis = 1)*100
    min_speed = np.min(V_plane, axis = 1)*100

    abs_max_vel = np.max(np.abs(V_plane), axis = 1)*100
    positive_mask = max_speed >= 0

    res = np.zeros_like(abs_max_vel)
    res[positive_mask] = abs_max_vel[positive_mask]
    res[~positive_mask] = -abs_max_vel[~positive_mask]
    

    if N_frames != frames:
        plt.plot(range(frames)[::2], max_speed,'--', color = color, label = f'{label} max')
        plt.plot(range(frames)[::2], min_speed,':', color = color, label = f'{label} min')
    else:
        plt.plot(max_speed,'--', label = f'{label} max', color = color)
        plt.plot(min_speed,':', label = f'{label} min', color = color)
    plt.xlabel('frame')
    plt.ylabel('velocity (cm/s)')
    plt.title('Velocity through plane')

def plot_mean_speed_plane(idx_intersection_plane_fluid, data, frames,normal, order_normal = [0, 1, 2], label = '', color = 'black'):
    N_frames = data['u'].shape[0]

    #Velocity through plane
    V_plane = velocity_through_plane(idx_intersection_plane_fluid, data, normal, order_normal = order_normal)
    mean_speed= np.mean(V_plane, axis = 1)*100
    if N_frames != frames:
        plt.plot(range(frames)[::2], mean_speed,'.-', color = 'yellowgreen', label = label)
    else:
        plt.plot(mean_speed,'.-', label = label, color = color)
    plt.xlabel('frame')
    plt.ylabel('Mean velocity (cm/s)')
    plt.title('Velocity through plane')
    return mean_speed


def plot_line_speed(x_line,frame, data, normal, points_in_plane, label = '', color = 'black'):
    plane_slice = points_in_plane[x_line, :, :]
    
    idx_line_p = np.where(plane_slice == 1)
    # Get points
    line_points_idx = np.index_exp[frame, x_line, idx_line_p[0], idx_line_p[1]]

    V_line = data[line_points_idx]*100#np.sqrt(data['u'][line_points_idx]**2+ data['v'][line_points_idx]**2+ data['w'][line_points_idx]**2) *100#(data['u'][line_points_idx]*normal[0]+ data['v'][line_points_idx]*normal[1]+ data['w'][line_points_idx]*normal[2]) *100
    #project
    plt.plot(V_line, label = label, color = color)
    plt.xlabel('voxel number')
    plt.ylabel('V (cm/s)')
    plt.title(f'Speed on line in frame {frame}')



def create_temporal_comparison_gif(lr, hr, pred, vel, save_as):

    v_lr = lr[vel]
    v_hr = hr[vel]
    v_pred = pred[vel]

    v_NN = temporal_NN_interpolation(v_lr,v_hr.shape )

    combined_image = np.concatenate((v_NN, v_hr, v_pred), axis = 3)
    idx = 30

    generate_gif_volume(combined_image[:,idx, :, : ], axis = 0, save_as = save_as)


def plot_k_r2_vals(gt, pred, bounds, peak_flow_frame,color_b = KI_colors['Plum'] , save_as= 'K_R2_values'):
    vel_colnames = ['u', 'v', 'w']
    vel_plotname = [r'$V_x$', r'$V_y$', r'$V_z$']


    print('Peak flow frame:', peak_flow_frame)
    frames = gt['u'].shape[0]
    k, r2, k_bounds,r2_bounds  = np.zeros(3*frames), np.zeros(3*frames), np.zeros(3*frames), np.zeros(3*frames)
    bounds_mask = bounds.copy()
    core_mask = gt['mask'] - bounds_mask

    plt.figure(figsize=(8, 8))
    #calculate k values in core and boundary region
    for i, vel in enumerate(vel_colnames):
        for t in range(frames):
            k[t+i*frames], r2[t+i*frames]  = calculate_k_R2( pred[vel][t], gt[vel][t], core_mask[t])
            k_bounds[t+i*frames], r2_bounds[t+i*frames]  = calculate_k_R2( pred[vel][t], gt[vel][t], bounds[t])

    min_val = np.minimum(0.05, np.minimum(np.min(k_bounds), np.min(r2_bounds)))
    max_val = np.maximum(np.max(k), np.max(r2))
    #make subplots
    for i, (vel, title) in enumerate(zip(vel_colnames, vel_plotname)):
        plt.subplot(2, 3, i+1)
        plt.plot(range(frames), k[i*frames:i*frames+frames] , label = 'k core', color = 'black')
        plt.plot(range(frames), k_bounds[i*frames:i*frames+frames] ,'--',  label = 'k boundary', color = color_b)
        plt.plot(np.ones(frames), 'k:')
        plt.ylim([min_val, np.maximum(max_val, 1.05)])
        plt.title(title)
        plt.xlabel('frames')
        plt.ylabel('k')
        plt.locator_params(axis='y', nbins=3)
        plt.locator_params(axis='x', nbins=3)
        plt.scatter(np.ones(2)*peak_flow_frame, [k[i*frames+peak_flow_frame],k_bounds[i*frames+peak_flow_frame]] , label = 'peak flow frame', color = KI_colors['Grey'])
        plt.legend()
        print(f'Average k vals core {np.average(k[i*frames:i*frames+frames])}')
        print(f'Average k vals boundary {np.average(k_bounds[i*frames:i*frames+frames])}')
        print(f'Min k vals core {np.min(k[i*frames:i*frames+frames])}')
        print(f'Min k vals boundary {np.min(k_bounds[i*frames:i*frames+frames])}')
        print(f'Max k vals core {np.max(k[i*frames:i*frames+frames])}')
        print(f'Max k vals boundary {np.max(k_bounds[i*frames:i*frames+frames])}')
    for i, (vel, title) in enumerate(zip(vel_colnames, vel_plotname)):
        plt.subplot(2, 3, i+4)
        print(f'Average R2 vals core {np.average(r2[i*frames:i*frames+frames])}')
        print(f'Average R2 vals boundary {np.average(r2_bounds[i*frames:i*frames+frames])}')
        plt.plot(range(frames), r2[i*frames:i*frames+frames] ,label = r'$R^2$ core', color = 'black')
        plt.plot(range(frames), r2_bounds[i*frames:i*frames+frames] ,'--', label = r'$R^2$ boundary', color = color_b)
        plt.plot(np.ones(frames), 'k:')
        plt.ylim([min_val, np.maximum(max_val, 1.05)])
        plt.title(title)
        plt.xlabel('frames')
        plt.ylabel(r'$R^2$')
        plt.scatter(np.ones(2)*peak_flow_frame, [r2[i*frames+peak_flow_frame], r2_bounds[i*frames+peak_flow_frame]] , label = 'peak flow frame', color = KI_colors['Grey'])
        plt.locator_params(axis='y', nbins=3)
        plt.locator_params(axis='x', nbins=3)
        plt.legend()

    plt.tight_layout()
    plt.savefig(f'{save_as}_VXYZ.svg')

    #save each plot separately
    plt.figure(figsize=(5, 5))
    for i, (vel, title) in enumerate(zip(vel_colnames, vel_plotname)):
        plt.clf()
        plt.plot(range(frames), k[i*frames:i*frames+frames] , label = 'k core', color = 'black')
        plt.plot(range(frames), k_bounds[i*frames:i*frames+frames] ,'--',  label = 'k boundary', color = color_b)
        plt.plot(np.ones(frames), 'k:')
        plt.ylim([min_val, np.maximum(max_val, 1.05)])
        plt.title(title)
        plt.xlabel('frames')
        plt.ylabel('k')
        plt.locator_params(axis='y', nbins=3)
        plt.locator_params(axis='x', nbins=3)
        plt.scatter(np.ones(2)*peak_flow_frame, [k[i*frames+peak_flow_frame],k_bounds[i*frames+peak_flow_frame]] , label = 'peak flow frame', color = KI_colors['Grey'])
        plt.legend()
        plt.savefig(f'{save_as}_k_{vel}.svg', bbox_inches='tight')
    for i, (vel, title) in enumerate(zip(vel_colnames, vel_plotname)):
        plt.clf()
        plt.plot(range(frames), r2[i*frames:i*frames+frames] ,label = r'$R^2$ core', color = 'black')
        plt.plot(range(frames), r2_bounds[i*frames:i*frames+frames] ,'--', label = r'$R^2$ boundary', color = color_b)
        plt.plot(np.ones(frames), 'k:')
        plt.ylim([min_val, np.maximum(max_val, 1.05)])
        plt.title(title)
        plt.xlabel('frames')
        plt.ylabel(r'$R^2$')
        plt.locator_params(axis='y', nbins=3)
        plt.locator_params(axis='x', nbins=3)
        plt.scatter(np.ones(2)*peak_flow_frame, [r2[i*frames+peak_flow_frame], r2_bounds[i*frames+peak_flow_frame]] , label = 'peak flow frame', color = KI_colors['Grey'])
        plt.legend()
        plt.savefig(f'{save_as}_R2_{vel}.svg', bbox_inches='tight')


def plot_k_r2_vals_nobounds(k, r2, peak_flow_frame, figsize = (15,5),exclude_tbounds = False,  save_as= None):
    print('Plot k and r2 values with peak flow frame', peak_flow_frame, ' ..')

    vel_colnames = ['u', 'v', 'w']
    vel_plotname = [r'$V_x$', r'$V_y$', r'$V_z$']
    k_legendname = [r'k$_x$', r'k$_y$', r'k$_z$']
    R2_legendname = [r'R$_x^2$',r'R$_y^2$',r'R$_z^2$']

    fontsize = 18
    frames = k.shape[1]

    t_range = range(frames)
    if exclude_tbounds:
        print('Exclude temporal boundaries..')
        t_range = t_range[1:-1]
        k = k[:, 1:-1]
        r2 = r2[:, 1:-1]
        idx_peak_flow_frame = peak_flow_frame -1
    else:
        idx_peak_flow_frame = peak_flow_frame

    min_val = np.minimum(0.45, np.minimum(np.min(k), np.min(r2)))
    max_val = np.maximum(1.05, np.maximum(np.max(k), np.max(r2)))

    # Create subplots
    # plt.subplots_adjust(wspace=0.3)
    fig1, axs = plt.subplots(1, 3, figsize=figsize)#, sharey=True)

    for i, (vel, title) in enumerate(zip(vel_colnames, vel_plotname)):
        # Plot k values on the primary y-axis
        
        axs[i].set_ylim([min_val, max_val])
        axs[i].set_title(title, fontsize = fontsize)
        axs[i].set_xlabel('frame', fontsize=fontsize)
        axs[i].set_ylabel(r'k/$R^2$', fontsize=fontsize)
        axs[i].locator_params(axis='y', nbins=3)
        axs[i].locator_params(axis='x', nbins=3)
        axs[i].tick_params(axis='y', labelsize = fontsize)
        axs[i].tick_params(axis='x', labelsize = fontsize)
        print('Peak flow frame:', peak_flow_frame, t_range[idx_peak_flow_frame])
        # k-values
        axs[i].plot(t_range, k[i, :], label=k_legendname[i], color='black')
        axs[i].scatter(np.ones(1)*t_range[idx_peak_flow_frame], [k[i, idx_peak_flow_frame]], color=KI_colors['Grey'])
        # R2 values 
        axs[i].plot(t_range, r2[i, :], '--', label=R2_legendname[i], color=KI_colors['Plum'])
        axs[i].scatter(np.ones(1)*t_range[idx_peak_flow_frame], [r2[i, idx_peak_flow_frame]],  color=KI_colors['Grey']) #label='peak flow frame',


        axs[i].plot(np.ones(frames), 'k:')
        text_frame = len(t_range) -7  # Middle of the plot range
        axs[i].text(text_frame, 1.0, '1.0', verticalalignment='bottom', horizontalalignment='left', fontsize=fontsize, color='black')
        axs[i].legend(loc='lower right', fontsize=fontsize)

    plt.tight_layout()
    if save_as is not None: plt.savefig(f'{save_as}_VXYZ.png')

    # save separate
    for i, (vel, title) in enumerate(zip(vel_colnames, vel_plotname)):

        # Create subplots
        fig2, ax = plt.subplots(figsize=(5, 5))

        ax.set_ylim([min_val, max_val])
        ax.set_title(title, fontsize = fontsize)
        ax.set_xlabel('frame', fontsize=fontsize)
        ax.set_ylabel(r'k/$R^2$', fontsize=fontsize)
        ax.locator_params(axis='y', nbins=3)
        ax.locator_params(axis='x', nbins=3)
        ax.tick_params(axis='y', labelsize = fontsize)
        ax.tick_params(axis='x', labelsize = fontsize)
        
        # k-values
        ax.plot(t_range, k[i, :], label='k', color='black')
        ax.scatter(np.ones(1)*t_range[idx_peak_flow_frame], [k[i, peak_flow_frame]], color=KI_colors['Grey'])
        # R2 values 
        ax.plot(t_range, r2[i, :], '--', label=r'$R^2$', color=KI_colors['Plum'])
        ax.scatter(np.ones(1)*t_range[idx_peak_flow_frame], [r2[i, peak_flow_frame]], label='peak flow frame', color=KI_colors['Grey'])
        ax.plot(np.ones(frames), 'k:')
        ax.text(text_frame, 1.0, '1.0', verticalalignment='bottom', horizontalalignment='left', fontsize=fontsize, color='black')
        ax.legend(loc='lower right')
        if save_as is not None: plt.savefig(f'{save_as}_{vel}.svg')

        # Close the plot to avoid memory leaks
        plt.close()
    
    return fig1, axs

def calculate_and_plot_k_r2_vals_nobounds(gt, pred, mask, peak_flow_frame, figsize = (8,8),exclude_tbounds = False,  save_as= None):

    vel_colnames = ['u', 'v', 'w']
    frames = gt['u'].shape[0]
    #calculate k values in core and boundary region
    k, r2 = np.zeros((3, frames)), np.zeros((3, frames))
    for i, vel in enumerate(vel_colnames):
        for t in range(frames):
            k[i, t], r2[i, t]  = calculate_k_R2( pred[vel][t], gt[vel][t], mask[t])

    plot_k_r2_vals_nobounds(k, r2, peak_flow_frame, figsize = figsize,exclude_tbounds = exclude_tbounds,  save_as= save_as)

    return k, r2


def combined_correlation_k_r2_plots(gt, prediction, frame_idx, k, r2, peak_flow_frame, color_points='black', show_text=False, save_as=None, figsize=(15, 10), exclude_tbounds=False, t_s_range = [0, 1]):
    '''
    Combine correlation plots and k/r2 value plots into a single subplot of size (2, 3).
    '''
    fontsize = 18
    p = 0.1
    mask_threshold = 0.6
    plot_time_in_sec = True

    # Handle mask
    mask = np.asarray(gt['mask']).squeeze()
    if len(mask.shape) == 3:
        mask = create_dynamic_mask(mask, prediction['u'].shape[0])
    mask[np.where(mask > mask_threshold)] = 1 

    # Random indices for core and boundary points
    idx_core = np.where(mask[frame_idx] == 1)
    x_idx, y_idx, z_idx = random_indices3D(mask[frame_idx], n=int(p * np.count_nonzero(mask[frame_idx])))

    # Velocity values
    hr_u = np.asarray(gt['u'][frame_idx])
    hr_v = np.asarray(gt['v'][frame_idx])
    hr_w = np.asarray(gt['w'][frame_idx])

    sr_u = np.asarray(prediction['u'][frame_idx])
    sr_v = np.asarray(prediction['v'][frame_idx])
    sr_w = np.asarray(prediction['w'][frame_idx])

    # Core values for correlation plots
    hr_u_core, hr_v_core, hr_w_core = hr_u[x_idx, y_idx, z_idx], hr_v[x_idx, y_idx, z_idx], hr_w[x_idx, y_idx, z_idx]
    sr_u_core, sr_v_core, sr_w_core = sr_u[x_idx, y_idx, z_idx], sr_v[x_idx, y_idx, z_idx], sr_w[x_idx, y_idx, z_idx]
    
    def plot_regression_points(hr_vals, sr_vals, all_hr, all_sr, direction='u'):
        N = 100
        x_range = np.linspace(-abs_max, abs_max, N)
        
        corr_line, text = get_corr_line_and_r2(all_hr, all_sr, x_range)

        if show_text:
            plt.gca().text(0.05, 0.95, text, transform=plt.gca().transAxes, fontsize=fontsize, verticalalignment='top')
        plt.plot(x_range, x_range, color='grey', label='diagonal line')
        plt.plot(x_range, corr_line, 'k--')
        plt.scatter(hr_vals, sr_vals, s=30, c=[color_points], label='core voxels')
        plt.title(direction, fontsize=fontsize+2)
        plt.xlabel("V HR [m/s]", fontsize=fontsize)
        plt.ylabel("V SR [m/s]", fontsize=fontsize)
        plt.ylim(-abs_max, abs_max)
        plt.xlim(-abs_max, abs_max)
        plt.locator_params(axis='y', nbins=3)
        plt.locator_params(axis='x', nbins=3)
        plt.tick_params(axis='y', labelsize=fontsize)
        plt.tick_params(axis='x', labelsize=fontsize)

    vel_colnames = ['u', 'v', 'w']
    vel_plotname = [r'$V_x$', r'$V_y$', r'$V_z$']
    k_legendname = [r'k$_x$', r'k$_y$', r'k$_z$']
    R2_legendname = [r'R$_x^2$',r'R$_y^2$',r'R$_z^2$']

    frames = k.shape[1]

    t_range = range(frames)
    if plot_time_in_sec:
        t_range = np.linspace(t_s_range[0], t_s_range[1], frames)
    if exclude_tbounds:
        t_range = t_range[1:-1]
        k = k[:, 1:-1]
        r2 = r2[:, 1:-1]
        idx_peak_flow_frame = peak_flow_frame - 1
    else:
        idx_peak_flow_frame = peak_flow_frame
    

    min_val = np.minimum(0.45, np.minimum(np.min(k), np.min(r2)))
    max_val = np.maximum(1.05, np.maximum(np.max(k), np.max(r2)))

    # Create the combined subplot
    fig, axs = plt.subplots(2, 3, figsize=figsize)

    # Adjust vertical spacing
    # plt.subplots_adjust(hspace=1.0)

    # First row - correlation plots
    plot_pairs = [(hr_u_core, sr_u_core, hr_u, sr_u, r'V$_x$'),
                  (hr_v_core, sr_v_core, hr_v, sr_v, r'V$_y$'),
                  (hr_w_core, sr_w_core, hr_w, sr_w, r'V$_z$')]
    
    abs_max = max(np.abs(np.hstack([hr_u_core, hr_v_core, hr_w_core])).max(), np.abs(np.hstack([sr_u_core, sr_v_core, sr_w_core])).max())

    for i, (hr_vals, sr_vals, all_hr, all_sr, direction) in enumerate(plot_pairs):
        plt.sca(axs[0, i])
        plot_regression_points(hr_vals, sr_vals, all_hr[idx_core], all_sr[idx_core], direction=direction)

    # Second row - k and R² value plots
    for i, (vel, title) in enumerate(zip(vel_colnames, vel_plotname)):
        axs[1, i].set_ylim([min_val, max_val])
        # axs[1, i].set_title(title, fontsize=fontsize)
        if plot_time_in_sec:
            axs[1, i].set_xlabel('time [s]', fontsize=fontsize)
        else:
            axs[1, i].set_xlabel('frame', fontsize=fontsize)
        axs[1, i].set_ylabel(r'k/$R^2$', fontsize=fontsize)
        axs[1, i].locator_params(axis='y', nbins=3)
        axs[1, i].locator_params(axis='x', nbins=3)
        axs[1, i].tick_params(axis='y', labelsize=fontsize)
        axs[1, i].tick_params(axis='x', labelsize=fontsize)

        # k-values
        axs[1, i].plot(t_range, k[i, :], label=k_legendname[i], color='black')
        axs[1, i].scatter(np.ones(1) * t_range[idx_peak_flow_frame], [k[i, idx_peak_flow_frame]], color='grey')
        # R² values
        axs[1, i].plot(t_range, r2[i, :], '--', label=R2_legendname[i], color=KI_colors['Plum'])
        axs[1, i].scatter(np.ones(1) * t_range[idx_peak_flow_frame], [r2[i, idx_peak_flow_frame]], color='grey')

        axs[1, i].plot(t_range, np.ones(frames), 'k:')
        text_frame = t_range[-1] - 0.1*t_range[-1]
        axs[1, i].text(text_frame, 1.0, '1.0', verticalalignment='bottom', horizontalalignment='left', fontsize=fontsize-2, color='black')
        axs[1, i].legend(loc='lower right', fontsize=fontsize)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    if save_as is not None:
        print(f'Saving combined correlation and k/r2 figure in {save_as}_combined_plot.svg')
        plt.savefig(f'{save_as}_combined_plot.svg')

    plt.show()

    return fig, axs




def animate_data_over_time_gif(spatial_idx, data,  min_v, max_v, save_as = 'Animate_',fps =10,  colormap = 'viridis', show_colorbar = False):

    print('Create animation plotting data over time..')

    fig = plt.figure(frameon=False)
    im1 = plt.imshow(data[0, spatial_idx[0], spatial_idx[1], spatial_idx[2]],interpolation='none', vmin=min_v, vmax=max_v, cmap = colormap)
    plt.axis('off')
    if show_colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05, ) 
        cb = plt.colorbar(im1, cax=cax)
        cb.set_label(label='V [m/s]',size=15,weight='bold')
        cb.ax.locator_params(nbins=3)
        # tick_locator = ticker.MaxNLocator(nbins=3)
        # cb.locator = tick_locator
        # cb.update_ticks()
    plt.tight_layout()

    # initialization function: plot the background of each frame
    def init():
        im1.set_data(np.random.random((5,5)))
        return [im1]

    # animation function is called sequentially
    def animate(i):
        im1.set_array(data[i, spatial_idx[0], spatial_idx[1], spatial_idx[2]])
        return [im1]

    anim = animation.FuncAnimation(fig,animate, init_func=init,
                                frames = data.shape[0],
                                interval = 100, repeat = False) # in ms)
    anim.save(f'{save_as}_{fps}fps.gif', fps=fps)

def animate_comparison_gif(lr_data, hr_data, sr_data,spatial_idx, min_v, max_v, save_as='Comparison_', fps=10, colormap='viridis'):
    """
    Create a synchronized animation comparing LR, HR, and SR data in a 3x3 grid (vx, vy, vz rows; LR, HR, SR columns).
    
    Parameters:
        lr_data (dict): Low-resolution data dictionary with keys ['u', 'v', 'w'].
        hr_data (dict): High-resolution data dictionary with keys ['u', 'v', 'w'].
        sr_data (dict): Super-resolution data dictionary with keys ['u', 'v', 'w'].
        min_v (float): Minimum value for color scale.
        max_v (float): Maximum value for color scale.
        save_as (str): Prefix for the saved GIF file.
        fps (int): Frames per second for the animation.
        colormap (str): Colormap to use for the plots.
    """
    print('Creating comparison animation with dictionary input...')

    # Determine frame synchronization
    lr_factor = int(hr_data['u'].shape[0] / lr_data['u'].shape[0])  # Ratio of HR/SR to LR frames
    lr_frames = {key: np.repeat(data, lr_factor, axis=0) for key, data in lr_data.items()}  # Synchronize LR frames

    # Set up the figure
    fig, axes = plt.subplots(3, 3, figsize=(5.5, 5), gridspec_kw={'height_ratios': [1, 1, 1]})
    ims = []

    components = ['u', 'v', 'w']  # Velocity components vx, vy, vz
    titles = ['LR', 'HR', 'SR']   # Column titles
    row_labels = [r'$V_x$', r'$V_y$', r'$V_z$']  # Row labels

    # Initialize subplots for each component and resolution
    for row, (comp, row_label) in enumerate(zip(components, row_labels)):
        for col, (data_dict, title) in enumerate(zip([lr_frames, hr_data, sr_data], titles)):
            ax = axes[row, col]
            if col == 0:
                print('Row label:', row_label, col)
                # ax.set_ylabel(row_label, fontsize=14, labelpad=15)
                # fig.text(0.05, 0.88 - row * 0.28, row_label, fontsize=14, ha='center', va='center', rotation=0)
                fig.text(0.02, 0.77 - row * 0.255, row_label, fontsize=14, ha='center', va='center', rotation=0)
            if row == 0:
                ax.set_title(title, fontsize=14)
            ax.axis('off')
            im = ax.imshow(data_dict[comp][0, spatial_idx[0], spatial_idx[1], spatial_idx[2]], interpolation='none', vmin=min_v, vmax=max_v, cmap=colormap)
            
            ims.append(im)
    # Adjust layout to make space for the colorbar
    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15, hspace=0.1, wspace=0.1)

    # Add a horizontal colorbar below the grid
    cbar_ax = fig.add_axes([0.05, 0.13, 0.9, 0.02])  # Adjust colorbar position
    cb = fig.colorbar(ims[-1], cax=cbar_ax, orientation='horizontal')
    cb.set_label('velocity [m/s]', fontsize=14)
    cb.ax.tick_params(labelsize=10)

    def animate(i):
        # Update each subplot with new frame data
        for idx, (data_dict, im) in enumerate(zip([lr_frames, hr_data, sr_data] * 3, ims)):
            row = idx // 3
            comp = components[row]
            im.set_array(data_dict[comp][i, spatial_idx[0], spatial_idx[1], spatial_idx[2]])
        return ims

    anim = animation.FuncAnimation(fig, animate, frames=hr_data['u'].shape[0], interval=1000 / fps, blit=False)

    # Save the animation as a GIF
    anim.save(f'{save_as}_{fps}fps.gif', fps=fps)
    plt.close(fig)

def animate_comparison_with_error_gif(
        lr_data, hr_data, sr_data, spatial_idx, min_v, max_v, min_err, max_err, 
        save_as='Comparison_Error_', fps=10, colormap='viridis'):
    """
    Create a synchronized animation comparing LR, HR, SR, and their absolute error in a 3x4 grid.
    
    Parameters:
        lr_data (dict): Low-resolution data dictionary with keys ['u', 'v', 'w'].
        hr_data (dict): High-resolution data dictionary with keys ['u', 'v', 'w'].
        sr_data (dict): Super-resolution data dictionary with keys ['u', 'v', 'w'].
        min_v (float): Minimum value for velocity color scale.
        max_v (float): Maximum value for velocity color scale.
        min_err (float): Minimum value for error color scale.
        max_err (float): Maximum value for error color scale.
        save_as (str): Prefix for the saved GIF file.
        fps (int): Frames per second for the animation.
        colormap (str): Colormap to use for the plots.
    """
    print('Creating comparison animation with error column...')

    t_range = np.linspace(0, 1, hr_data['u'].shape[0])  # Time range in seconds
    # Determine frame synchronization
    lr_factor = int(hr_data['u'].shape[0] / lr_data['u'].shape[0])  # Ratio of HR/SR to LR frames
    lr_frames = {key: np.repeat(data, lr_factor, axis=0) for key, data in lr_data.items()}  # Synchronize LR frames

    # Set up the figure
    fig, axes = plt.subplots(3, 4, figsize=(7.5, 5.5), gridspec_kw={'height_ratios': [1, 1, 1]})
    ims = []

    components = ['u', 'v', 'w']  # Velocity components vx, vy, vz
    titles = ['LR', 'HR', 'SR', 'Abs. Error']  # Column titles
    row_labels = [r'$V_x$', r'$V_y$', r'$V_z$']  # Row labels

    # Initialize subplots for each component and resolution
    for row, (comp, row_label) in enumerate(zip(components, row_labels)):
        for col, title in enumerate(titles):
            ax = axes[row, col]
            if col == 0:
                fig.text(0.02, 0.77 - row * 0.25, row_label, fontsize=14, ha='center', va='center', rotation=0)
            if row == 0:
                ax.set_title(title, fontsize=14)
            ax.axis('off')

            if col < 3:  # LR, HR, SR columns
                data_dict = [lr_frames, hr_data, sr_data][col]
                im = ax.imshow(data_dict[comp][0, spatial_idx[0], spatial_idx[1], spatial_idx[2]], 
                               interpolation='none', vmin=min_v, vmax=max_v, cmap=colormap)
            else:  # Error column
                error = np.abs(hr_data[comp] - sr_data[comp])
                im = ax.imshow(error[0, spatial_idx[0], spatial_idx[1], spatial_idx[2]], 
                               interpolation='none', vmin=min_err, vmax=max_err, cmap=colormap)
            
            ims.append(im)

    # Adjust layout to make space for the colorbars
    fig.subplots_adjust(left=0.05, right=0.92, top=0.9, bottom=0.15, hspace=0.1, wspace=0.1)

    # Add horizontal colorbar for velocity below the entire grid (spanning LR, HR, SR columns)
    cbar_ax_v = fig.add_axes([0.05, 0.13, 0.645, 0.02])  # Position for velocity colorbar
    cb_v = fig.colorbar(ims[-2], cax=cbar_ax_v, orientation='horizontal')  # Use the last velocity plot for colorbar
    cb_v.set_label('velocity [m/s]', fontsize=14)
    cb_v.ax.tick_params(labelsize=10)

    # Add horizontal colorbar for absolute error, only below the error column
    cbar_ax_e = fig.add_axes([0.72, 0.13, 0.198, 0.02])  # Position for error colorbar, right of error column
    cb_e = fig.colorbar(ims[-1], cax=cbar_ax_e, orientation='horizontal')  # Use the error plot for colorbar
    cb_e.set_label('error [m/s]', fontsize=14)
    cb_e.ax.tick_params(labelsize=10)
    time_text = axes[2, 3].text(0.95, 0.05, f'0.0s', transform=axes[2, 3].transAxes,
                            fontsize=14, color='white', ha='right', va='bottom')

    def animate(i):
        # Update each subplot with new frame data
        for idx, im in enumerate(ims):
            col = idx % 4
            row = idx // 4
            comp = components[row]
            
            if col < 3:  # Update LR, HR, SR columns
                data_dict = [lr_frames, hr_data, sr_data][col]
                im.set_array(data_dict[comp][i, spatial_idx[0], spatial_idx[1], spatial_idx[2]])
            else:  # Update Error column
                error = np.abs(hr_data[comp] - sr_data[comp])
                im.set_array(error[i, spatial_idx[0], spatial_idx[1], spatial_idx[2]])
         # Calculate the time in seconds and update the text in the last plot (Error plot)
        
        # axes[2, 3].text(0.95, 0.05, f'{t_range[i]:.1f}s', transform=axes[2, 3].transAxes,
        #                 fontsize=12, color='white', ha='right', va='bottom')
        time_text.set_text(f'{t_range[i]:.2f}s')
        return ims

    anim = animation.FuncAnimation(fig, animate, frames=hr_data['u'].shape[0], interval=1000 / fps, blit=False)

    # Save the animation as a GIF
    anim.save(f'{save_as}_{fps}fps.gif', fps=fps)
    plt.close(fig)


#TODO extend to time 
def generate_gif_volume(img3D, axis = 0, save_as = "animation"):
    # check that input is 3 dimensional suc that normalization is correct
    img3D = img3D.squeeze()
    assert len(img3D.shape) == 3


    img3D = check_and_normalize(img3D)

    if axis == 0:
            frames = [Image.fromarray(img3D[i, :, :]*255) for i in range(img3D.shape[0])]
    elif axis ==1:
            frames = [Image.fromarray(img3D[:, i, :]*255) for i in range(img3D.shape[1])]
    elif axis == 2:
            frames = [Image.fromarray(img3D[:, :, i]*255) for i in range(img3D.shape[2])]
    else: 
        print("Invalid axis input.")
    
    frame_one = frames[0]
    frame_one.save(save_as+".gif", format="GIF", append_images=frames,save_all=True, duration=500, loop=0) #/home/pcallmer/Temporal4DFlowNet/results/plots
    

def compare_peak_flow_pixel(gt,lr, model_names, set_names, labels, colors,name_comparison,patch_size, show_avg, show_pixel, use_dynamical_mask = False):

    plt.figure(figsize=(7, 5))
    def show_peak_flow_pixel(x, pred_data, label, color, line_style = '-'):
        '''Plot peak flow vosel in time and also averages cube around it '''
    
        if show_pixel: # show only flow of peak flow voxel 
            plt.plot(x, pred_data['speed'][:, idx_max[1], idx_max[2], idx_max[3]]*100,line_style, label = f'{label} pixel', color = color)

        if show_avg: # show average flow of  region around peak flow voxel. Regsion is depending on patch size
            plt.plot(x, np.average(pred_data['speed']  [:, idx_max[1]-patch_size:idx_max[1]+patch_size+1, idx_max[2]-patch_size:idx_max[2]+patch_size+1, idx_max[3]-patch_size:idx_max[3]+patch_size+1], axis = (1, 2, 3))*100,line_style,label = f'{label} avg', color = color)
    
    # get voxel with maximum flow
    idx_max = np.unravel_index(np.argmax(gt['speed']), shape = gt['speed'].shape)
    x = np.arange(gt['speed'].shape[0])

    show_peak_flow_pixel(x, gt, label = 'gt', color='black')
    show_peak_flow_pixel(x[::2 ], lr, label= 'low res',color='yellowgreen',line_style='-o')


    for m_name, s_name, label, color in zip(model_names, set_names, labels, colors):
        pred = load_velocity_data(f'{result_dir}/Temporal4DFlowNet_{m_name}/{s_name}set_result_model{data_model}_2mm_step{step}_{m_name[-4::]}_temporal.h5', {}, ['u_combined', 'v_combined', 'w_combined'], load_mask = False)
        show_peak_flow_pixel(x, pred, label, color, line_style='--')

    if show_avg:
        plt.title(f"Speed at pixel {idx_max[1::]} with average of number of voxels: {(2*patch_size+1)**3}")
    else:
         plt.title(f"Speed at pixel {idx_max[1::]}")
    plt.ylabel('Speed (cm/s)')
    plt.xlabel('Frame')
    plt.legend()

    plt.savefig(f'{eval_dir}/{name_comparison}_peak_flow_voxel_speed.png')
    plt.show()




# ------------------- INTERPOLATION FUNCTIONS---------------------------


def temporal_linear_interpolation(lr, hr_shape):
    '''
    Linear interpolation in time, from (t, h, w, d) to (2t, h, w, d)
    Be aware that if the hr shape is twice as high the last frame will be set to zero, since it it not in-between slices.
    Using a equidistant grid, leading to taking the average of two consequtive frames
    '''

    # only temporal resolution increases 
    downsampling_factor = lr.shape[0]/hr_shape[0]

    t_hr = np.linspace(0, lr[0]-downsampling_factor,  hr_shape[0])
    
    tg, xg, yg ,zg = np.mgrid[0:hr_shape[0], 0:hr_shape[1], 0:hr_shape[2], 0:hr_shape[3]]
    coord = np.array([tg.flatten(), xg.flatten(), yg.flatten() ,zg.flatten()])

    interpolate = scipy.ndimage.map_coordinates(lr,coord, mode='constant').reshape(hr_shape)
    return interpolate


def temporal_NN_interpolation(lr, hr_shape):
    '''
    Nearest neighbor interpolation in time, from (t, h, w, d) to (2t, h, w, d)
    For an equidistant grid this is only the doubling of the temporal resolution
    '''
    t_lr = np.arange(0, lr.shape[0])
    x_lr = np.arange(0, lr.shape[1])
    y_lr = np.arange(0, lr.shape[2])
    z_lr = np.arange(0, lr.shape[3])

    t_hr = np.linspace(0, lr.shape[0]-0.5,  hr_shape[0])
    
    # tg, xg, yg ,zg = np.mgrid(t_hr, x_lr, y_lr, z_lr, indexing='ij', sparse=True)

    tg, xg, yg ,zg = np.mgrid[0:hr_shape[0], 0:hr_shape[1], 0:hr_shape[2], 0:hr_shape[3]]
    coord = np.array([tg.flatten(), xg.flatten(), yg.flatten() ,zg.flatten()])

    interpolate = scipy.ndimage.map_coordinates(lr,coord, mode='constant').reshape(hr_shape)

    return interpolate

#TODO test this
def temporal_cubic_interpolation(lr, hr_shape):
    '''
    Cubic interpolation in time , from (t, h, w, d) to (2t, h, w, d)
    '''

    t_lr = np.arange(0, lr.shape[0])
    x_lr = np.arange(0, lr.shape[1])
    y_lr = np.arange(0, lr.shape[2])
    z_lr = np.arange(0, lr.shape[3])

    t_hr = np.linspace(0, lr.shape[0]-0.5,  hr_shape[0])
    
    
    tg, xg, yg ,zg = np.mgrid[0:hr_shape[0], 0:hr_shape[1], 0:hr_shape[2], 0:hr_shape[3]]
    coord = np.array([tg.flatten(), xg.flatten(), yg.flatten() ,zg.flatten()])

    interpolate = scipy.ndimage.map_coordinates(lr,coord, mode='constant').reshape(hr_shape)
    # interp = RegularGridInterpolator((t_lr, x_lr, y_lr, z_lr), lr, method='cubic', bounds_error=False, fill_value=0)
    # interpolate = interp((tg, xg, yg ,zg))

    return interpolate


def temporal_sinc_interpolation_ndarray(data, s_range, e_range):
    """
    Interpolate the data in the temporal direction using sinc interpolation
    data is nd array: expect T x X x Y x Z
    s_range is range of the sample points
    e_range is evaluation range
    """
    dt = s_range[1] - s_range[0]

    sinc_matrix =  np.sinc((e_range - s_range[:, None])/dt).transpose()

    #tensordot product
    sinc_interp = np.tensordot(sinc_matrix, data, axes = ([1], [0]))
    
    return np.asarray(sinc_interp)

def temporal_linear_interpolation_np(lr, hr_shape):
    """
    Linear interpolation in time, from (t, h, w, d) to (2t, h, w, d)
    """
    factor = int(hr_shape[0] / lr.shape[0])
    T, x, y, z = hr_shape
    interpolate = np.zeros((hr_shape))
    interpolate[::factor, :, :, :] = lr
    for t in range(0, T-factor, factor):
        interpolate[1+t, :, :, :] = (interpolate[t, :, :, :] + interpolate[1+t+1, :, :, :])/factor

    interpolate[-1, :, :, :] = interpolate[-2, :, :, :] 

    return interpolate



def spatial3D_NN_interpolation(lr, hr_shape, method = 'nearest'):
    assert len(lr.shape) == 3
    assert len(hr_shape) == 3

    x_hr = np.linspace(0, lr.shape[0],  int(hr_shape[0]))
    y_hr = np.linspace(0, lr.shape[1],  int(hr_shape[1]))
    z_hr = np.linspace(0, lr.shape[2],  int(hr_shape[2]))
    
    xg, yg ,zg = np.meshgrid(x_hr, y_hr, z_hr, indexing='ij', sparse=True)
    
    interpolate = scipy.ndimage.map_coordinates(lr, np.array([xg, yg, zg]), mode=method)

    return interpolate


