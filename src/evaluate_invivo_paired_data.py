import numpy as np
import time
import h5py
import pandas as pd

import argparse
import matplotlib.pyplot as plt
from utils.evaluate_utils import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import defaultdict
from prepare_data.h5functions import save_to_h5
import matplotlib
import scipy
from utils.colors import *
import matplotlib.animation as animation
import matplotlib.cm as cm
from data_specifics.INVIVOII_params import *

plt.rcParams['figure.figsize'] = [10, 8]

def plot_plane_flows(u_hr, v_hr, w_hr, u_lr, v_lr, w_lr, u_sr, v_sr, w_sr, 
                     cp_plane, plane_normal, idxs_nonflow_area, 
                     order_normal = [0, 1, 2], 
                     show_mask = True, save_as = None, 
                    thickness=2, factor_plane_normal = [1, 1, 1], lr_hb_duration=1, hr_hb_duration=1):

    print("Plot plane flow..")

    plane_normal /= np.linalg.norm(plane_normal)

    # calculate the plane
    d = -np.dot(cp_plane, plane_normal)
    xx, yy = np.meshgrid(np.arange(0, u_hr.shape[1]), np.arange(0, u_hr.shape[2]))
    zz = (-plane_normal[0] * xx - plane_normal[1] * yy - d) * 1. / plane_normal[2] # adapt to oder of normal ? 
    
    # Initialize the volume region
    volume_region = np.zeros_like(mask)
    
    # Iterate over a range of values to create a thickness around the plane
    for t in range(-thickness, thickness + 1):
        zz_t = zz + t  # Shift plane points by t voxels
        zz_t[np.where(zz_t < 0)] = 0
        zz_t[np.where(zz_t >= u_hr.shape[3])] = u_hr.shape[3] - 1
        
        # Get points within the volume
        points_in_thick_plane = np.zeros_like(mask)
        points_in_thick_plane[xx.flatten().astype(int), yy.flatten().astype(int), zz_t.flatten().astype(int)] = 1

        # Add these points to the volume region
        volume_region += points_in_thick_plane
    # Ensure the region values are binary (1 if inside the volume, 0 if outside)
    volume_region = np.clip(volume_region, 0, 1)

    # Restrict the volume to fluid region
    volume_core = volume_region.copy()
    volume_core[np.where(mask == 0)] = 0

    # Adjust to different models
    volume_selected_region = volume_core.copy()
    for idx_nonflow_area in idxs_nonflow_area:
        volume_selected_region[idx_nonflow_area] = 0

    print('----------------')

    # plot only the selected region in the thin slice plane
    print(f'Number of points in the plane: {np.sum(volume_selected_region)}')

    if show_mask:
        # Always adjust to different models
        zz[np.where(zz < 0)] = 0
        zz[np.where(zz >= u_hr.shape[3])] = u_hr.shape[3] - 1
        points_in_plane = np.zeros_like(mask)
        points_in_plane[xx.flatten().astype(int), yy.flatten().astype(int), zz.flatten().astype(int)] = 1
        idx_plane                = np.where(points_in_plane>0)
        points_MV = points_in_plane.copy()
        points_MV[np.where(mask==0)] = 0
        for idx_nonflow_area in idxs_nonflow_area:
            points_MV[idx_nonflow_area] = 0
        img_mask = mask[idx_plane].reshape(xx.shape[1], -1)
        img_MV_mask = points_MV[idx_plane].reshape(xx.shape[1], -1)
        plt.imshow(mask[idx_plane].reshape(xx.shape[1], -1))
        plt.imshow(img_MV_mask+img_mask)
        if save_as is not None:
            plt.savefig(f'{eval_dir}/{os.path.basename(save_as).replace("vel", "mask_of_plane")}')
        plt.show()

    if True: 
        #-----plot MV 2; 3D plot with plane and intersection----- 
        boundary_mask, _ = get_boundaries(mask)

        x_bounds, y_bounds, z_bounds = np.where(boundary_mask==1)
        x_MV, y_MV, z_MV = np.where(volume_selected_region==1)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        # ax.plot_surface(xx, yy, zz, alpha = 0.33, color = KI_colors['Grey']) # plot plane
        ax.scatter3D(x_bounds, y_bounds, z_bounds, s= 3, alpha = 0.1) #plot boundary points
        ax.scatter3D(cp_plane[0], cp_plane[1], cp_plane[2],'x', color = 'red') #show point in plane
        ax.scatter3D(cp_plane[0], cp_plane[1] ,cp_plane[2] , s = 3, color = 'black') # plot normal point
        ax.scatter3D(x_MV, y_MV, z_MV, alpha = 0.2, s = 3, color = 'red') #plot MV points
        plt.xlabel('x')
        plt.ylabel('y')
        ax.set_zlabel('z')
        ax.set_zlim(0, hr_u.shape[3])
        plt.title(f'3D plot of plane and intersection, in total {np.sum(volume_selected_region)} points')
        plt.savefig(f'{eval_dir}/{os.path.basename(save_as).replace("vel", "3D_plot")}')
        plt.show()

    if True:
        # inpt and idx plane
        zz[np.where(zz < 0)] = 0
        zz[np.where(zz >= u_hr.shape[3])] = u_hr.shape[3] - 1
        points_in_plane = np.zeros_like(mask)
        points_in_plane[xx.flatten().astype(int), yy.flatten().astype(int), zz.flatten().astype(int)] = 1
        idx_plane                = np.where(points_in_plane>0)
        lr_vel   = velocity_through_plane_paired_invivo(idx_plane, u_lr, v_lr, w_lr, plane_normal, order_normal = order_normal).reshape(u_lr.shape[0], xx.shape[1], -1)
        hr_vel   = velocity_through_plane_paired_invivo(idx_plane, u_hr, v_hr, w_hr, plane_normal, order_normal = order_normal).reshape(u_hr.shape[0], xx.shape[1], -1)
        pred_vel = velocity_through_plane_paired_invivo(idx_plane, u_sr, v_sr, w_sr, plane_normal, order_normal = order_normal).reshape(u_sr.shape[0], xx.shape[1], -1)

        #-----plot MV 1; Qualitave plot----- 
        # idx_crop = np.index_exp[:, 17:37, 5:25]
        # idx_crop2 = np.index_exp[17:37, 5:25]
        idx_crop = np.index_exp[:, :, :]

        # crop to important region
        lr_vel_crop = lr_vel[idx_crop]
        hr_vel_crop = hr_vel[idx_crop ]
        pred_vel_crop = pred_vel[idx_crop]

        timepoints_hr = [6, 7, 8, 9]
        timepoints_lr = [3, 4, 5, 6]
        plot_qual_comparsion(hr_vel_crop[timepoints_hr[0]:timepoints_hr[-1]+1], lr_vel_crop[timepoints_lr[0]:timepoints_lr[-1]+1], pred_vel_crop[timepoints_hr[0]:timepoints_hr[-1]+1], img_MV_mask,None,  [], [], 
                         min_v=None, max_v = None,  timepoints = timepoints_hr,figsize=(8, 5),  save_as = save_as.replace("vel", "qualitative_timeseries_LRHRSR"))
        plt.show()
    
    plt.close('all')
    #-----plot MV 3; Plot Flow profile within mask----- 

    # project velocities on plane normal
    vel_hr_plane_projection = vel_projection_plane(u_hr, v_hr, w_hr, plane_normal, volume_selected_region, order_normal = order_normal, factor_plane_normal = factor_plane_normal)
    vel_sr_plane_projection = vel_projection_plane(u_sr, v_sr, w_sr, plane_normal, volume_selected_region, order_normal = order_normal, factor_plane_normal = factor_plane_normal)
    vel_lr_plane_projection = vel_projection_plane(u_lr, v_lr, w_lr, plane_normal, volume_selected_region, order_normal = order_normal, factor_plane_normal = factor_plane_normal)

    # plot mean velocities over time:
    t_range_in_s_hr = np.linspace(0, hr_hb_duration, vel_hr_plane_projection.shape[0])
    t_range_in_s_lr = np.linspace(0, lr_hb_duration, vel_lr_plane_projection.shape[0])
    t_range_in_s_sr = np.linspace(0, lr_hb_duration, vel_sr_plane_projection.shape[0])
    plt.figure(figsize=(6, 3))
    plt.plot(t_range_in_s_hr, vel_hr_plane_projection,'-o', label='HR', color='black', markersize = 3)
    plt.plot(t_range_in_s_lr, vel_lr_plane_projection,'-o', label='LR', color='forestgreen', markersize = 3)
    plt.plot(t_range_in_s_sr, vel_sr_plane_projection, '--o', label='SR', color=KI_colors['Plum'], markersize = 3)
    plt.legend(fontsize=16)
    plt.xlabel('time [s]', fontsize=16)
    plt.ylabel('Velocity (m/s)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize = 16)
    plt.locator_params(axis='y', nbins=3) 
    if save_as is not None:
        plt.savefig(save_as, bbox_inches='tight')#, transparent=True)
    plt.show()
    plt.close('all')
    return vel_hr_plane_projection, vel_sr_plane_projection, vel_lr_plane_projection

def velocity_through_plane_paired_invivo(idx_plane, u, v, w, plane_normal, order_normal = [0, 1, 2]):

    N_frames = u.shape[0]
    vx_in_plane = u[:, idx_plane[0], idx_plane[1], idx_plane[2]].reshape(N_frames, -1)
    vy_in_plane = v[:, idx_plane[0], idx_plane[1], idx_plane[2]].reshape(N_frames, -1)
    vz_in_plane = w[:, idx_plane[0], idx_plane[1], idx_plane[2]].reshape(N_frames, -1)
    return vx_in_plane*plane_normal[order_normal[0]]+ vy_in_plane*plane_normal[order_normal[1]]+ vz_in_plane*plane_normal[order_normal[2]]
    

def vel_projection_plane(u, v, w, plane_normal, mask_volume, order_normal = [0, 1, 2], factor_plane_normal = [1, 1, 1] ):
    assert mask_volume.shape == u.shape[1:], 'Mask and velocity field must have the same shape'
    assert mask_volume.ndim == 3, 'Mask must be 3D'

    idx_x, idx_y, idx_z = np.where(mask_volume)
    idx_timeseries = np.index_exp[:, idx_x, idx_y, idx_z]

    # project velocities on plane normal
    vel_plane_projection = np.mean((u[idx_timeseries]*plane_normal[order_normal[0]]*factor_plane_normal[0]+
                                    v[idx_timeseries]*plane_normal[order_normal[1]]* factor_plane_normal[1]+
                                    w[idx_timeseries]*plane_normal[order_normal[2]]* factor_plane_normal[2]).reshape(u.shape[0], -1),
                                        where = mask_volume[idx_x, idx_y, idx_z].astype(bool), axis = 1)

    return vel_plane_projection


def plot_correlation_paired_invivo(v_hr, v_sr, mask, frame_idx, p=0.1, fontsize=16, save_as=None, title_y='SR', direction='x', ax=None):
    """
    Plot correlation plot between high-resolution (v_hr) and super-resolution (v_sr) data for a specified frame.
    
    Parameters:
        v_hr: High-resolution data.
        v_sr: Super-resolution data.
        mask: Mask for valid data points.
        frame_idx: Frame index for plotting.
        p: Proportion of random samples to take from mask.
        fontsize: Font size for labels and title.
        save_as: File path to save the plot.
        title_y: Title for the Y-axis.
        direction: Title for the plot (X or Y direction).
        ax: Optional Matplotlib axes object for plotting (for subplots).
    """
    
    # Set up random sampling and shared figure settings
    mask = np.asarray(mask).squeeze()
    if v_hr.shape[0] != v_sr.shape[0]:
        frame_idx_sr = frame_idx // int(v_hr.shape[0] / v_sr.shape[0])
    else:
        frame_idx_sr = frame_idx
    
    # Determine core and random sample points
    idx_core = np.where(mask == 1)
    x_idx, y_idx, z_idx = random_indices3D(mask, n=int(p * np.count_nonzero(mask)))
    
    # Extract random samples and compute limits
    hr_samples = v_hr[frame_idx, x_idx, y_idx, z_idx]
    sr_samples = v_sr[frame_idx_sr, x_idx, y_idx, z_idx]
    abs_max = max(abs(hr_samples).max(), abs(sr_samples).max())
    
    # Generate correlation line data
    x_range = np.linspace(-abs_max, abs_max, 100)
    corr_line, text = get_corr_line_and_r2(v_hr[frame_idx][idx_core], v_sr[frame_idx_sr][idx_core], x_range)

    # Create a new figure and axes if none provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
        single_plot = True
    else:
        single_plot = False
    
    # Plotting
    ax.plot(x_range, x_range, color='gray', label='Diagonal Line')   # Diagonal
    ax.plot(x_range, corr_line, 'k--', label='Fit Line')             # Regression
    
    # Scatter plot
    ax.scatter(hr_samples, sr_samples, s=30, color='black', label='Core Voxels')
    
    # Set labels and title
    # ax.set_title(rf"{direction}", fontsize=fontsize)  # Ensure proper formatting for direction
    ax.set_xlabel(r"$V_{HR}$ [m/s]", fontsize=fontsize)        # Consistent formatting for HR
    if single_plot:
        ax.set_ylabel(rf"$V_{{{title_y}}}$ [m/s]", fontsize=fontsize)  # Ensure proper formatting for title_y
    
    # Customize limits, ticks, and spines
    ax.set_xlim(-abs_max, abs_max)
    ax.set_ylim(-abs_max, abs_max)
    ax.locator_params(axis='both', nbins=3)
    ax.tick_params(axis='both', which='minor', color='lightgray', labelsize=fontsize//2+2, labelcolor='gray')
    ax.tick_params(axis='both', which='major', color='gray', labelcolor='gray', labelsize=fontsize//2+2)
    for spine in ax.spines.values():
        spine.set_color('gray')
        spine.set_linewidth(0.5)

    # Add text for R² and other fit metrics if needed
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=fontsize, verticalalignment='top')
    
    # Adjust layout and save if specified
    if ax is None:  # Only apply tight layout if a new figure was created
        plt.tight_layout()
        if save_as:
            plt.savefig(save_as, transparent=True)
        plt.show()

    return ax  # Return the axes object instead of the figure



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="My script description")
    parser.add_argument("--model", type=str, help="Optional argument to pass the name of the model")
    args = parser.parse_args()

    # Define directories and filenames
    if args.model is not None:
        network_model = args.model
    else:
        network_model = '20250502-1741' 
    # for one network evluation on multiple invivo datasets
    
    # set directories 
    # input_dir = 'data/paired_invivo/'
    lr_dir = 'Temporal4DFlowNet/data/paired_invivo'
    hr_dir = 'Temporal4DFlowNet/data/paired_invivo'
    sr_dir = 'Temporal4DFlowNet/results/in_vivo/paired_data'
    
    show_animation = False
    show_qual_timeseries = True
    show_plane_velocities = True
    show_bland_altman = True
    show_qual_peak_frames = True
    tabulate_results = True
    include_interpolation = False
    # plot_mean_speed = True
    # plot_correlation = True
    mode_transformed = False

    
    eval_dir = f'Temporal4DFlowNet/results/in_vivo/paired_data/plots/IEEETMI2/{network_model}'
    if mode_transformed:
        eval_dir = f'{eval_dir}_transformed'
    os.makedirs(eval_dir, exist_ok=True)

    volunteers = ['v3', 'v4', 'v5', 'v6', 'v7']#['v3', 'v5', 'v7']#['v3', 'v4', 'v5', 'v6', 'v7'] # 

    results_aorta_lv = {}
    results_aorta = {}
    results_lv = {}
   
    
     #            idx_cube = np.index_exp[time_points[0]:time_points[-1]+1, 67,  50:90, 30:70]
            # idx_cube_lr = np.index_exp[time_points[0]//2:time_points[-1]//2+1, 67,  50:90, 30:70]


    results_planes_aorta = {}

    for volunteer in volunteers:
        print(f'Evaluate volunteer {volunteer}...')

        hr_filename = f'{volunteer}_wholeheart_25mm_20ms.h5'
        lr_filename = f'{volunteer}_wholeheart_25mm_40ms.h5'
        if mode_transformed:
            hr_filename = f'{volunteer}_wholeheart_25mm_20ms_transformed.h5'
            lr_filename = f'{volunteer}_wholeheart_25mm_40ms_transformed.h5'
        sr_filename = f'{lr_filename[:-3]}/{lr_filename[:-3]}_{network_model}.h5'
        print(sr_filename, network_model)

        os.makedirs(eval_dir, exist_ok=True)

        with h5py.File(f'{lr_dir}/{lr_filename}', 'r') as f:
            print(f.keys())
            lr_u = np.array(f['u'])/100 #m/s
            lr_v = np.array(f['v'])/100 #m/s
            lr_w = np.array(f['w'])/100 #m/s
            mask = np.array(f['mask'])
            # mask = np.array(f['mask_smooth'])
            mask_aorta = np.array(f['mask_aorta_smooth'])
            print('mask shape:', mask.shape)
            
            mask_lv = np.array(f['mask_LV_smooth'])
            if 'hb_duration' in f:
                lr_hb_duration = np.array(f['hb_duration']).astype(float)/1000
            else:
                print('No hb_duration in low resolution file, setting to 1s')
                lr_hb_duration = 1.0 #s, default value if not present in file

        with h5py.File(f'{hr_dir}/{hr_filename}', 'r') as f:
            hr_u = np.array(f['u'])/100 #m/s
            hr_v = np.array(f['v'])/100 #m/s
            hr_w = np.array(f['w'])/100 #m/s
            if 'hb_duration' in f:
                hr_hb_duration = np.array(f['hb_duration']).astype(float)/1000
            else:
                print('No hb_duration in high resolution file, setting to 1s')
                hr_hb_duration = 1.0
            if 'dx' in f:
                dx = np.array(f['dx'])
                if dx.shape[0] == 1:
                    dx = dx[0]
            else:
                dx = 2.5

        print(f" sr filename and sr dir {sr_filename}, {sr_dir}")
        with h5py.File(f'{sr_dir}/{sr_filename}', 'r') as f:
            sr_u = np.array(f['u_combined'])/100 #m/s
            sr_v = np.array(f['v_combined'])/100 #m/s
            sr_w = np.array(f['w_combined'])/100 #m/s

        T_lr = lr_u.shape[0]
        T_hr = hr_u.shape[0]
        print('hb_duration', lr_hb_duration, hr_hb_duration)

        if not mode_transformed:
            volunteer_plane_normal_ascending = volunteer_plane_normal_ascending_orig
            volunteer_plane_normal_descending = volunteer_plane_normal_descending_orig
            volunteer_plot_settings = volunteer_plot_settings_orig
        else:
            volunteer_plane_normal_ascending = volunteer_plane_normal_ascending_transformed
            volunteer_plane_normal_descending = volunteer_plane_normal_descending_transformed
            volunteer_plot_settings = volunteer_plot_settings_transformed


        if include_interpolation:
            # interpolate low resolution data to high resolution
            interpolate_linear_u = temporal_linear_interpolation_np(lr_u, hr_u.shape)
            interpolate_linear_v = temporal_linear_interpolation_np(lr_v, hr_v.shape)
            interpolate_linear_w = temporal_linear_interpolation_np(lr_w, hr_w.shape)

            # load sinc interpolation
            hr_range = np.linspace(0,1,  hr_u.shape[0])
            ups_factor = hr_u.shape[0]//lr_u.shape[0]
            lr_range = hr_range[::ups_factor] # downsamplie like this to get exact same evaluation points

            interpolate_sinc_u = temporal_sinc_interpolation_ndarray(lr_u, lr_range, hr_range)
            interpolate_sinc_v = temporal_sinc_interpolation_ndarray(lr_v, lr_range, hr_range)
            interpolate_sinc_w = temporal_sinc_interpolation_ndarray(lr_w, lr_range, hr_range)

        hr_mean_speed = np.mean(np.sqrt(hr_u**2 + hr_v**2 + hr_w**2), axis=(1, 2, 3), where=mask.astype(bool))
        T_peak_flow_frame = np.argmax(hr_mean_speed)
        synthesized_peak_flow_frame = T_peak_flow_frame.copy() 
        # take next frame if peak flow frame included in lr data
        if synthesized_peak_flow_frame % 2 == 0: 
            if hr_mean_speed[synthesized_peak_flow_frame-1] > hr_mean_speed[synthesized_peak_flow_frame+1]:
                synthesized_peak_flow_frame -=1
            else:
                synthesized_peak_flow_frame +=1


        #------ whole heart evaluation (aorta + lv) ------
        if tabulate_results:
            # calculate rmse 
            rmse_u = np.sqrt(np.mean((hr_u - sr_u)**2, axis=(1, 2, 3), where=mask.astype(bool)))
            rmse_v = np.sqrt(np.mean((hr_v - sr_v)**2, axis=(1, 2, 3), where=mask.astype(bool)))
            rmse_w = np.sqrt(np.mean((hr_w - sr_w)**2, axis=(1, 2, 3), where=mask.astype(bool)))

            # calculate k and r2 values for each direction over time

            k_u_srhr, r2_u_srhr = calculate_k_R2_timeseries(sr_u, hr_u, np.repeat(mask[np.newaxis, ...], T_hr, axis=0))
            k_v_srhr, r2_v_srhr = calculate_k_R2_timeseries(sr_v, hr_v, np.repeat(mask[np.newaxis, ...], T_hr, axis=0))
            k_w_srhr, r2_w_srhr = calculate_k_R2_timeseries(sr_w, hr_w, np.repeat(mask[np.newaxis, ...], T_hr, axis=0))

            results_aorta_lv_volunteer = {}
            results_aorta_lv_volunteer['Volunteer'] = volunteer
            results_aorta_lv_volunteer['RMSE_u'] = rmse_u
            results_aorta_lv_volunteer['RMSE_v'] = rmse_v
            results_aorta_lv_volunteer['RMSE_w'] = rmse_w
            results_aorta_lv_volunteer['k_u'] = k_u_srhr
            results_aorta_lv_volunteer['k_v'] = k_v_srhr
            results_aorta_lv_volunteer['k_w'] = k_w_srhr
            results_aorta_lv_volunteer['r2_u'] = r2_u_srhr
            results_aorta_lv_volunteer['r2_v'] = r2_v_srhr
            results_aorta_lv_volunteer['r2_w'] = r2_w_srhr


            #------ aorta evaluation ------
            # calculate rmse
            rmse_u_aorta = np.sqrt(np.mean((hr_u - sr_u)**2, axis=(1, 2, 3), where=mask_aorta.astype(bool)))
            rmse_v_aorta = np.sqrt(np.mean((hr_v - sr_v)**2, axis=(1, 2, 3), where=mask_aorta.astype(bool)))
            rmse_w_aorta = np.sqrt(np.mean((hr_w - sr_w)**2, axis=(1, 2, 3), where=mask_aorta.astype(bool)))

            # calculate k and r2 values for each direction over time

            k_u_srhr_aorta, r2_u_srhr_aorta = calculate_k_R2_timeseries(sr_u, hr_u, np.repeat(mask_aorta[np.newaxis, ...], T_hr, axis=0))
            k_v_srhr_aorta, r2_v_srhr_aorta = calculate_k_R2_timeseries(sr_v, hr_v, np.repeat(mask_aorta[np.newaxis, ...], T_hr, axis=0))
            k_w_srhr_aorta, r2_w_srhr_aorta = calculate_k_R2_timeseries(sr_w, hr_w, np.repeat(mask_aorta[np.newaxis, ...], T_hr, axis=0))

            results_aorta_volunteer = {}
            results_aorta_volunteer['Volunteer'] = volunteer
            results_aorta_volunteer['RMSE_u'] = rmse_u_aorta
            results_aorta_volunteer['RMSE_v'] = rmse_v_aorta
            results_aorta_volunteer['RMSE_w'] = rmse_w_aorta
            results_aorta_volunteer['k_u'] = k_u_srhr_aorta
            results_aorta_volunteer['k_v'] = k_v_srhr_aorta
            results_aorta_volunteer['k_w'] = k_w_srhr_aorta
            results_aorta_volunteer['r2_u'] = r2_u_srhr_aorta
            results_aorta_volunteer['r2_v'] = r2_v_srhr_aorta
            results_aorta_volunteer['r2_w'] = r2_w_srhr_aorta

            #------ left ventricle evaluation ------
            # calculate rmse
            rmse_u_lv = np.sqrt(np.mean((hr_u - sr_u)**2, axis=(1, 2, 3), where=mask_lv.astype(bool)))
            rmse_v_lv = np.sqrt(np.mean((hr_v - sr_v)**2, axis=(1, 2, 3), where=mask_lv.astype(bool)))
            rmse_w_lv = np.sqrt(np.mean((hr_w - sr_w)**2, axis=(1, 2, 3), where=mask_lv.astype(bool)))

            # calculate k and r2 values for each direction over time
            
            k_u_srhr_lv, r2_u_srhr_lv = calculate_k_R2_timeseries(sr_u, hr_u, np.repeat(mask_lv[np.newaxis, ...], T_hr, axis=0))
            k_v_srhr_lv, r2_v_srhr_lv = calculate_k_R2_timeseries(sr_v, hr_v, np.repeat(mask_lv[np.newaxis, ...], T_hr, axis=0))
            k_w_srhr_lv, r2_w_srhr_lv = calculate_k_R2_timeseries(sr_w, hr_w, np.repeat(mask_lv[np.newaxis, ...], T_hr, axis=0))

            results_lv_volunteer = {}
            results_lv_volunteer['Volunteer'] = volunteer
            results_lv_volunteer['RMSE_u'] = rmse_u_lv
            results_lv_volunteer['RMSE_v'] = rmse_v_lv
            results_lv_volunteer['RMSE_w'] = rmse_w_lv
            results_lv_volunteer['k_u'] = k_u_srhr_lv
            results_lv_volunteer['k_v'] = k_v_srhr_lv
            results_lv_volunteer['k_w'] = k_w_srhr_lv
            results_lv_volunteer['r2_u'] = r2_u_srhr_lv
            results_lv_volunteer['r2_v'] = r2_v_srhr_lv
            results_lv_volunteer['r2_w'] = r2_w_srhr_lv

            # save results in dictionary
            results_aorta_lv[volunteer] = results_aorta_lv_volunteer
            results_aorta[volunteer] = results_aorta_volunteer
            results_lv[volunteer] = results_lv_volunteer

        # plot desecning and asceing aorta
        T_lr = lr_u.shape[0]
        T_hr = hr_u.shape[0]
        
        print('shapes u:', lr_u.shape, hr_u.shape, sr_u.shape)
        print('shapes v:', lr_v.shape, hr_v.shape, sr_v.shape)
        print('shapes w:', lr_w.shape, hr_w.shape, sr_w.shape)

        if show_plane_velocities:
            plane_normal_ascending = np.array(volunteer_plane_normal_ascending[f'{volunteer}_normal'])
            cp_plane_ascending = np.array(volunteer_plane_normal_ascending[f'{volunteer}_origin'])/dx

            plane_normal_descending = np.array(volunteer_plane_normal_descending[f'{volunteer}_normal'])
            cp_plane_descending = np.array(volunteer_plane_normal_descending[f'{volunteer}_origin'])/dx

            print('Evaluating ascending aorta..')
            # plot ascending aorta
            asc_hr, asc_sr, asc_lr = plot_plane_flows(hr_u, hr_v, hr_w, lr_u, lr_v, lr_w, sr_u, sr_v, sr_w, 
                            cp_plane_ascending, plane_normal_ascending, volunteer_plot_settings[volunteer]['idxs_nonflow_area_ascending'], order_normal = volunteer_plot_settings[volunteer]['order_normal'], 
                            thickness=volunteer_plot_settings[volunteer]['thickness_ascending'], factor_plane_normal = volunteer_plot_settings[volunteer]['factor_plane_normal'], 
                            save_as = f'{eval_dir}/{volunteer}_ascending_aorta_vel.png', lr_hb_duration=lr_hb_duration, hr_hb_duration=hr_hb_duration)

            # plot descending aorta
            print('Evaluating descending aorta..')
            desc_hr, desc_sr, desc_lr = plot_plane_flows(hr_u, hr_v, hr_w, lr_u, lr_v, lr_w, sr_u, sr_v, sr_w,
                            cp_plane_descending, plane_normal_descending, volunteer_plot_settings[volunteer]['idxs_nonflow_area_descending'], order_normal = volunteer_plot_settings[volunteer]['order_normal'],  
                            thickness=volunteer_plot_settings[volunteer]['thickness_descending'], factor_plane_normal = volunteer_plot_settings[volunteer]['factor_plane_normal'], 
                            save_as = f'{eval_dir}/{volunteer}_descending_aorta_vel.png', lr_hb_duration=lr_hb_duration, hr_hb_duration=hr_hb_duration)
            results_planes_aorta[volunteer] = {
                'ascending': {
                    'hr': asc_hr,
                    'sr': asc_sr,
                    'lr': asc_lr,
                },
                'descending': {
                    'hr': desc_hr,
                    'sr': desc_sr,
                    'lr': desc_lr,
                },
                'lr_hb_duration': lr_hb_duration,
                'hr_hb_duration': hr_hb_duration,
            }        

        if show_qual_timeseries:
            # time_points = [4, 5, 6, 7, 8,]
            time_points = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            idx_z = 67
            if mode_transformed:
                idx_z = hr_u.shape[1]-idx_z-1 # for transformed data, idx_z is flipped
            idx_cube = np.index_exp[time_points[0]:time_points[-1]+1, idx_z,  50:90, 30:70]
            idx_cube_lr = np.index_exp[time_points[0]//2:time_points[-1]//2+1, idx_z,  50:90, 30:70]
            # idx_cube = np.index_exp[time_points[0]:time_points[-1]+1, 67,  45:95, 20:70]
            # idx_cube_lr = np.index_exp[time_points[0]//2:time_points[-1]//2+1, 67,  45:95, 20:70]
            idx_mask = np.index_exp[59, 45:95, 20:70]

            # without interpolation
            plot_qual_comparsion(hr_u[idx_cube], lr_u[idx_cube_lr], sr_u[idx_cube], mask[idx_mask], None, [], [], time_points, None, None, include_error = False,  
                                figsize = (13, 3.5), save_as = f"{eval_dir}/{volunteer}_qual_comparison_u_frames{time_points[0]}-{time_points[-1]}.png", colormap='viridis')
            plot_qual_comparsion(hr_v[idx_cube], lr_v[idx_cube_lr], sr_v[idx_cube], mask[idx_mask], None, [], [], time_points, None, None, include_error = False,  
                                figsize = (13, 3.5), save_as = f"{eval_dir}/{volunteer}_qual_comparison_v_frames{time_points[0]}-{time_points[-1]}.png", colormap='viridis')
            plot_qual_comparsion(hr_w[idx_cube], lr_w[idx_cube_lr], sr_w[idx_cube], mask[idx_mask], None, [], [], time_points, None, None, include_error = False,  
                                figsize = (13, 3.5), save_as = f"{eval_dir}/{volunteer}_qual_comparison_w_frames{time_points[0]}-{time_points[-1]}.png", colormap='viridis')

            
            for time_point in time_points:
                if time_point % 2 != 0:
                    plt.figure(figsize=(4, 4))
                    plot_correlation_paired_invivo(hr_u, sr_u, mask, time_point, p=0.1, fontsize = 16, direction = r'$V_x$', save_as = f'{eval_dir}/{volunteer}_correlation_frame{time_point}_u.png')
                    plt.savefig(f'{eval_dir}/{volunteer}_correlation_frame{time_point}_Vx.png', bbox_inches='tight', transparent=True)
                    plt.close('all')
                    plot_correlation_paired_invivo(hr_v, sr_v, mask, time_point, p=0.1, fontsize = 16, direction = r'$V_y$', save_as = f'{eval_dir}/{volunteer}_correlation_frame{time_point}_v.png')
                    plt.savefig(f'{eval_dir}/{volunteer}_correlation_frame{time_point}_Vy.png', bbox_inches='tight', transparent=True)
                    plt.close('all')
                    plot_correlation_paired_invivo(hr_w, sr_w, mask, time_point, p=0.1, fontsize = 16, direction = r'$V_z$', save_as = f'{eval_dir}/{volunteer}_correlation_frame{time_point}_w.png')
                    plt.savefig(f'{eval_dir}/{volunteer}_correlation_frame{time_point}_Vz.png', bbox_inches='tight', transparent=True)

                    # make subplots for all directions
                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    plot_correlation_paired_invivo(hr_u, sr_u, mask, time_point, p=0.1, fontsize = 16, direction = r'$V_x$', ax = axes[0])
                    plot_correlation_paired_invivo(hr_v, sr_v, mask, time_point, p=0.1, fontsize = 16, direction = r'$V_y$', ax = axes[1])
                    plot_correlation_paired_invivo(hr_w, sr_w, mask, time_point, p=0.1, fontsize = 16, direction = r'$V_z$', ax = axes[2])
                    axes[0].set_ylabel(r'V$_{SR}$ [m/s]', fontsize=16)
                    plt.tight_layout()
                    plt.savefig(f'{eval_dir}/{volunteer}_correlation_frame{time_point}_VxVyVz.png', bbox_inches='tight', transparent=True)
                    plt.close('all')
            if show_qual_peak_frames:

                idx_cube = np.index_exp[synthesized_peak_flow_frame, 67, 45:95, 20:70]
                mask_slice = np.index_exp[67, 45:95, 20:70]
                if include_interpolation:
                    comparison_data = [
                        (interpolate_linear_u[idx_cube], interpolate_linear_v[idx_cube], interpolate_linear_w[idx_cube]),
                        (interpolate_sinc_u[idx_cube], interpolate_sinc_v[idx_cube], interpolate_sinc_w[idx_cube])
                    ]
                    comparison_names = ['linear', 'sinc']
                else:
                    comparison_data = []
                    comparison_names = []

                plot_qual_comparison_peak(
                    hr_u[idx_cube], hr_v[idx_cube], hr_w[idx_cube],
                    sr_u[idx_cube], sr_v[idx_cube], sr_w[idx_cube], 
                    mask[mask_slice], None, comparison_data, comparison_names, synthesized_peak_flow_frame, None, None, colormap='plasma',
                    figsize=(7.2, 4.6), save_as=f"{eval_dir}/{volunteer}_qual_peak_timepoint{synthesized_peak_flow_frame}_VxVyVz.png"
                )
                plt.show()
                plt.close('all')

        if show_bland_altman:
            # plot bland altman
            

            print(f'Plot bland altman for volunteer {volunteer}...')
            # plot bland altman
            bland_altman_plot(sr_u, hr_u, mask,timepoint=synthesized_peak_flow_frame, save_as=f'{eval_dir}/{volunteer}_bland_altman_u_peaksyn{synthesized_peak_flow_frame}.png')
            plt.show()
            plt.close('all')
            bland_altman_plot(sr_v, hr_v, mask,timepoint=synthesized_peak_flow_frame, save_as=f'{eval_dir}/{volunteer}_bland_altman_v_peaksyn{synthesized_peak_flow_frame}.png')
            plt.show()
            plt.close('all')
            bland_altman_plot(sr_w, hr_w, mask,timepoint=synthesized_peak_flow_frame, save_as=f'{eval_dir}/{volunteer}_bland_altman_w_peaksyn{synthesized_peak_flow_frame}.png')
            plt.show()

            fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
            max_diff = np.max([np.max(hr_u[synthesized_peak_flow_frame][np.where(mask> 0.5)] - sr_u[synthesized_peak_flow_frame][np.where(mask> 0.5)]),
                                np.max(hr_v[synthesized_peak_flow_frame][np.where(mask> 0.5)] - sr_v[synthesized_peak_flow_frame][np.where(mask> 0.5)]),
                                np.max(hr_w[synthesized_peak_flow_frame][np.where(mask> 0.5)] - sr_w[synthesized_peak_flow_frame][np.where(mask> 0.5)])])
            bland_altman_plot(sr_u, hr_u, mask, timepoint=synthesized_peak_flow_frame,y_lim=(-max_diff, max_diff),  ax=axes[0],  fontsize=16)
            axes[0].tick_params(axis='y', labelleft=True)
            axes[0].set_ylabel(r'V$_{HR}$ - V$_{SR}$ [m/s]', fontsize=16)

            bland_altman_plot(sr_v, hr_v, mask, timepoint=synthesized_peak_flow_frame,y_lim=(-max_diff, max_diff), ax=axes[1],fontsize=16)
            axes[1].tick_params(axis='y', labelleft=True)

            bland_altman_plot(sr_w, hr_w, mask, timepoint=synthesized_peak_flow_frame,y_lim=(-max_diff, max_diff), ax=axes[2],fontsize=16)
            axes[2].tick_params(axis='y', labelleft=True)
            # set background color
            mean_diff_u = np.mean(hr_u[synthesized_peak_flow_frame][np.where(mask> 0.5)] - sr_u[synthesized_peak_flow_frame][np.where(mask> 0.5)])
            mean_diff_v = np.mean(hr_v[synthesized_peak_flow_frame][np.where(mask> 0.5)] - sr_v[synthesized_peak_flow_frame][np.where(mask> 0.5)])
            mean_diff_w = np.mean(hr_w[synthesized_peak_flow_frame][np.where(mask> 0.5)] - sr_w[synthesized_peak_flow_frame][np.where(mask> 0.5)])

            
            print(f'Mean bias u: {mean_diff_u}, v: {mean_diff_v}, w: {mean_diff_w}')    

            plt.tight_layout()
            plt.savefig(f'{eval_dir}/{volunteer}_bland_altman_peaksyn{synthesized_peak_flow_frame}_VxVyVz_centered.png', transparent=True)
            plt.show()
        
            
            fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharey='row') #15 10
            vel_plotname = [r'$V_x$', r'$V_y$', r'$V_z$']
            
            x_rnd, y_rnd, z_rnd = random_indices3D(mask, n = int(0.1*np.sum(mask)))
            idx_core_t = np.where(mask > 0.5)

            abs_max_u = np.maximum(np.max(np.abs(hr_u[synthesized_peak_flow_frame][idx_core_t])), np.max(np.abs(sr_u[synthesized_peak_flow_frame][idx_core_t])))
            abs_max_v = np.maximum(np.max(np.abs(hr_v[synthesized_peak_flow_frame][idx_core_t])), np.max(np.abs(sr_v[synthesized_peak_flow_frame][idx_core_t])))
            abs_max_w = np.maximum(np.max(np.abs(hr_w[synthesized_peak_flow_frame][idx_core_t])), np.max(np.abs(sr_w[synthesized_peak_flow_frame][idx_core_t])))

            abs_max = np.maximum(abs_max_u, np.maximum(abs_max_v, abs_max_w))

            plot_regression_points_new(axes[0, 0],hr_u[synthesized_peak_flow_frame, x_rnd, y_rnd, z_rnd], sr_u[synthesized_peak_flow_frame, x_rnd, y_rnd, z_rnd], 
                                    hr_u[synthesized_peak_flow_frame][idx_core_t], sr_u[synthesized_peak_flow_frame][idx_core_t], abs_max, direction= r'V$_x$', color='black', show_text=True)
            bland_altman_plot(sr_u, hr_u, mask, timepoint=synthesized_peak_flow_frame, ax=axes[1, 0], fontsize=18, centered_ylim=True, y_lim=(-max_diff, max_diff))

            plot_regression_points_new(axes[0, 1], hr_v[synthesized_peak_flow_frame, x_rnd, y_rnd, z_rnd], sr_v[synthesized_peak_flow_frame, x_rnd, y_rnd, z_rnd], 
                                    hr_v[synthesized_peak_flow_frame][idx_core_t], sr_v[synthesized_peak_flow_frame][idx_core_t], abs_max, direction=r'V$_y$', color='black', show_text=True)
            bland_altman_plot(sr_v, hr_v, mask, timepoint=synthesized_peak_flow_frame, ax=axes[1, 1], fontsize=18, centered_ylim=True, y_lim=(-max_diff, max_diff))

            plot_regression_points_new(axes[0, 2], hr_w[synthesized_peak_flow_frame, x_rnd, y_rnd, z_rnd], sr_w[synthesized_peak_flow_frame, x_rnd, y_rnd, z_rnd], 
                                    hr_w[synthesized_peak_flow_frame][idx_core_t], sr_w[synthesized_peak_flow_frame][idx_core_t], abs_max, direction=r'V$_z$', color='black', show_text=True)
            bland_altman_plot(sr_w, hr_w, mask, timepoint=synthesized_peak_flow_frame, ax=axes[1, 2], fontsize=18, centered_ylim=True, y_lim=(-max_diff, max_diff))
            axes[0, 0].tick_params(axis='y', labelleft=True)
            axes[1, 0].tick_params(axis='y', labelleft=True)
            axes[0, 1].tick_params(axis='y', labelleft=True)
            axes[1, 1].tick_params(axis='y', labelleft=True)
            axes[0, 2].tick_params(axis='y', labelleft=True)
            axes[1, 2].tick_params(axis='y', labelleft=True)
            
            axes[1, 0].set_ylabel(r'V$_{HR}$ - V$_{SR}$ [m/s]', fontsize=16)

            plt.tight_layout()
            plt.savefig(f'{eval_dir}/{volunteer}_correlation_and_blandaltman_COMBINED_synpeakframe{synthesized_peak_flow_frame}_core.png')
            plt.show()

        if show_animation:
            include_orig_data = True
            eval_gifs = f'{eval_dir}/gifs'
            os.makedirs(eval_gifs, exist_ok = True)
            spatial_idx = volunteer_plot_settings[volunteer]['idx_anim_slice']
            u_min = np.quantile(hr_u[: spatial_idx[0], spatial_idx[1], spatial_idx[2]], 0.01)
            u_max = np.quantile(hr_u[: spatial_idx[0], spatial_idx[1], spatial_idx[2]], 0.99)
            v_min = np.quantile(hr_v[: spatial_idx[0], spatial_idx[1], spatial_idx[2]], 0.01)
            v_max = np.quantile(hr_v[: spatial_idx[0], spatial_idx[1], spatial_idx[2]], 0.99)
            w_min = np.quantile(hr_w[: spatial_idx[0], spatial_idx[1], spatial_idx[2]], 0.01)
            w_max = np.quantile(hr_w[: spatial_idx[0], spatial_idx[1], spatial_idx[2]], 0.99)

            diff_u_max = np.max((np.abs(hr_u-sr_u)*mask)[: spatial_idx[0], spatial_idx[1], spatial_idx[2]])
            diff_v_max = np.max((np.abs(hr_v-sr_v)*mask)[: spatial_idx[0], spatial_idx[1], spatial_idx[2]])
            diff_w_max = np.max((np.abs(hr_w-sr_w)*mask)[: spatial_idx[0], spatial_idx[1], spatial_idx[2]])

            

            min_V = np.min([u_min, v_min, w_min])
            max_V = np.max([u_max, v_max, w_max])

            min_err = 0
            max_err = np.max([diff_u_max, diff_v_max, diff_w_max])

            fps_anim = 10
            fps_pred = fps_anim
            if include_orig_data: 
                
                print('Create video of original data..')
                # animate_data_over_time_gif(volunteer_plot_settings[volunteer]['idx_anim_slice'], mag, 0, np.quantile(magn, 0.99),      fps = fps_anim , save_as = f'{eval_gifs}/{volunteer}_animate_mag', colormap='Greys_r' )
                # animate_data_over_time_gif(volunteer_plot_settings[volunteer]['idx_anim_slice'], mask, 0, 1,           fps = fps_anim , save_as = f'{eval_gifs}/{volunteer}_animate_mask', colormap='Greys' )
            #     animate_data_over_time_gif(volunteer_plot_settings[volunteer]['idx_anim_slice'], hr_u, min_V, max_V,      fps = fps_anim , save_as = f'{eval_gifs}/{volunteer}_animate_u_gt')
            #     animate_data_over_time_gif(volunteer_plot_settings[volunteer]['idx_anim_slice'], hr_v, min_V, max_V,      fps = fps_anim , save_as = f'{eval_gifs}/{volunteer}_animate_v_gt')
            #     animate_data_over_time_gif(volunteer_plot_settings[volunteer]['idx_anim_slice'], hr_w, min_V, max_V,      fps = fps_anim , save_as = f'{eval_gifs}/{volunteer}_animate_w_gt')
            #     animate_data_over_time_gif(volunteer_plot_settings[volunteer]['idx_anim_slice'], hr_u*mask, min_V, max_V,fps = fps_anim , save_as = f'{eval_gifs}/{volunteer}_animate_u_gt_fluid')
            #     animate_data_over_time_gif(volunteer_plot_settings[volunteer]['idx_anim_slice'], hr_v*mask, min_V, max_V,fps = fps_anim , save_as = f'{eval_gifs}/{volunteer}_animate_v_gt_fluid')
            #     animate_data_over_time_gif(volunteer_plot_settings[volunteer]['idx_anim_slice'], hr_w*mask, min_V, max_V,fps = fps_anim , save_as = f'{eval_gifs}/{volunteer}_animate_w_gt_fluid')

            # animate_data_over_time_gif(volunteer_plot_settings[volunteer]['idx_anim_slice'], lr_u, min_V, max_V, fps = fps_anim//2 , save_as = f'{eval_gifs}/{volunteer}_animate_u_lr')
            # animate_data_over_time_gif(volunteer_plot_settings[volunteer]['idx_anim_slice'], lr_v, min_V, max_V, fps = fps_anim//2 , save_as = f'{eval_gifs}/{volunteer}_animate_v_lr')
            animate_data_over_time_gif(volunteer_plot_settings[volunteer]['idx_anim_slice'], lr_w, min_V, max_V, fps = fps_anim//2 , save_as = f'{eval_gifs}/{volunteer}_animate_w_lr')

            # animate_data_over_time_gif(volunteer_plot_settings[volunteer]['idx_anim_slice'], sr_u, min_V, max_V, fps = fps_anim , save_as = f'{eval_gifs}/{volunteer}_animate_u_sr')
            # animate_data_over_time_gif(volunteer_plot_settings[volunteer]['idx_anim_slice'], sr_v, min_V, max_V, fps = fps_anim , save_as = f'{eval_gifs}/{volunteer}_animate_v_sr')
            # animate_data_over_time_gif(volunteer_plot_settings[volunteer]['idx_anim_slice'], sr_w, min_V, max_V, fps = fps_anim , save_as = f'{eval_gifs}/{volunteer}_animate_w_sr')

            animate_data_over_time_gif(volunteer_plot_settings[volunteer]['idx_anim_slice'], np.abs(hr_u-sr_u)*mask, min_err, max_err, fps = fps_anim, save_as = f'{eval_gifs}/{volunteer}_animate_u_hr_sr_diff')
            animate_data_over_time_gif(volunteer_plot_settings[volunteer]['idx_anim_slice'], np.abs(hr_v-sr_v)*mask, min_err, max_err, fps = fps_anim, save_as = f'{eval_gifs}/{volunteer}_animate_v_hr_sr_diff')
            animate_data_over_time_gif(volunteer_plot_settings[volunteer]['idx_anim_slice'], np.abs(hr_w-sr_w)*mask, min_err, max_err, fps = fps_anim, save_as = f'{eval_gifs}/{volunteer}_animate_w_hr_sr_diff')

    # results in df
    if tabulate_results:
        # plot results
        plt.figure(figsize=(15, 10))

        # RMSE
        plt.subplot(2, 3, 1)
        for volunteer in volunteers:
            plt.plot(results_aorta_lv[volunteer]['RMSE_u'], label=volunteer)
        plt.xlabel('Frame', fontsize=12)
        plt.ylabel('RMSE u', fontsize=12)
        plt.legend(fontsize=10)

        plt.subplot(2, 3, 2)
        for volunteer in volunteers:
            plt.plot(results_aorta_lv[volunteer]['RMSE_v'], label=volunteer)
        plt.xlabel('Frame', fontsize=12)
        plt.ylabel('RMSE v', fontsize=12)
        plt.legend(fontsize=10)

        plt.subplot(2, 3, 3)
        for volunteer in volunteers:
            plt.plot(results_aorta_lv[volunteer]['RMSE_w'], label=volunteer)
        plt.xlabel('Frame', fontsize=12)
        plt.ylabel('RMSE w', fontsize=12)
        plt.legend(fontsize=10)

        # k
        plt.subplot(2, 3, 4)
        for volunteer in volunteers:
            plt.plot(results_aorta_lv[volunteer]['k_u'], label=volunteer)
        plt.xlabel('Frame', fontsize=12)
        plt.ylabel('k u', fontsize=12)
        plt.legend(fontsize=10)

        plt.subplot(2, 3, 5)
        for volunteer in volunteers:
            plt.plot(results_aorta_lv[volunteer]['k_v'], label=volunteer)
        plt.xlabel('Frame', fontsize=12)
        plt.ylabel('k v', fontsize=12)
        plt.legend(fontsize=10)

        plt.subplot(2, 3, 6)
        for volunteer in volunteers:
            plt.plot(results_aorta_lv[volunteer]['k_w'], label=volunteer)
        plt.xlabel('Frame', fontsize=12)
        plt.ylabel('k w', fontsize=12)
        plt.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(f'{eval_dir}/rmse_k_whole_domain.png')
        plt.show()

        # r2
        plt.figure(figsize=(10, 3))
        plt.subplot(1, 3, 1)
        for volunteer in volunteers:
            plt.plot(results_aorta_lv[volunteer]['r2_u'], label=volunteer)
        plt.xlabel('Frame', fontsize=12)
        plt.ylabel('r2 u', fontsize=12)
        plt.legend(fontsize=10)

        plt.subplot(1, 3, 2)
        for volunteer in volunteers:
            plt.plot(results_aorta_lv[volunteer]['r2_v'], label=volunteer)
        plt.xlabel('Frame', fontsize=12)
        plt.ylabel('r2 v', fontsize=12)
        plt.legend(fontsize=10)

        plt.subplot(1, 3, 3)
        for volunteer in volunteers:
            plt.plot(results_aorta_lv[volunteer]['r2_w'], label=volunteer)
        plt.xlabel('Frame', fontsize=12)
        plt.ylabel('r2 w', fontsize=12)
        plt.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(f'{eval_dir}/R2_whole_domain.png')
        plt.show()


        # now create a dataframe with the results, by taking the mean over all timesteps
        results_aorta_lv_mean = defaultdict(list)
        results_aorta_mean = defaultdict(list)
        results_lv_mean = defaultdict(list)

        for volunteer in volunteers:
            for key in results_aorta_lv[volunteer].keys():
                if key != 'Volunteer':
                    results_aorta_lv_mean[key].append(np.mean(results_aorta_lv[volunteer][key]))
                    results_aorta_mean[key].append(np.mean(results_aorta[volunteer][key]))
                    results_lv_mean[key].append(np.mean(results_lv[volunteer][key]))
                else:
                    results_aorta_lv_mean[key].append(volunteer)
                    results_aorta_mean[key].append(volunteer)
                    results_lv_mean[key].append(volunteer)

        # to dataframe
        df_aorta_lv = pd.DataFrame(results_aorta_lv_mean)
        df_aorta = pd.DataFrame(results_aorta_mean)
        df_lv = pd.DataFrame(results_lv_mean)

        # save results
        df_aorta_lv.to_csv(f'{eval_dir}/results_aorta_lv.csv')
        df_aorta.to_csv(f'{eval_dir}/results_aorta.csv')
        df_lv.to_csv(f'{eval_dir}/results_lv.csv')

        # display results
        print('Results aorta and left ventricle:')
        print(df_aorta_lv)
        print('Results aorta:')
        print(df_aorta)
        print('Results left ventricle:')
        print(df_lv)

        # save as csv files
        df_aorta_lv.to_csv(f'{eval_dir}/results_aorta_lv.csv')
        df_aorta.to_csv(f'{eval_dir}/results_aorta.csv')
        df_lv.to_csv(f'{eval_dir}/results_lv.csv')

        # define peak systole and diastole frames for each volunteer 
        t_frames_diastole_systole = {'v3': [9, 21], 'v4':[11,23], 'v5': [9, 21], 'v6': [11, 25], 'v7':[9, 25]}

        df_aorta_lv_frames = pd.DataFrame()
        df_aorta_peak_systole = pd.DataFrame()
        df_aorta_peak_diastole = pd.DataFrame()
        df_lv_peak_systole = pd.DataFrame()
        df_lv_peak_diastole = pd.DataFrame()

        for volunteer in volunteers:
            t_sys = t_frames_diastole_systole[volunteer][0]
            t_dia = t_frames_diastole_systole[volunteer][1]
            df_aorta_lv_frames = pd.concat([df_aorta_lv_frames, pd.DataFrame(results_aorta_lv[volunteer]).iloc[t_sys, :].to_frame().T])
            df_aorta_lv_frames = pd.concat([df_aorta_lv_frames, pd.DataFrame(results_aorta_lv[volunteer]).iloc[t_dia, :].to_frame().T])
            df_aorta_peak_systole = pd.concat([df_aorta_peak_systole, pd.DataFrame(results_aorta[volunteer]).iloc[t_sys, :].to_frame().T])
            df_aorta_peak_diastole = pd.concat([df_aorta_peak_diastole, pd.DataFrame(results_aorta[volunteer]).iloc[t_dia, :].to_frame().T])
            df_lv_peak_systole = pd.concat([df_lv_peak_systole, pd.DataFrame(results_lv[volunteer]).iloc[t_sys, :].to_frame().T])
            df_lv_peak_diastole = pd.concat([df_lv_peak_diastole, pd.DataFrame(results_lv[volunteer]).iloc[t_dia, :].to_frame().T])

        # put in one dataframe
        df_aorta_lv_frames['Cardiac Phase'] = ['Systole', 'Diastole'] * len(volunteers)
        df_aorta_peak_systole['Cardiac Phase'] = ['Systole'] * len(volunteers)
        df_aorta_peak_diastole['Cardiac Phase'] = ['Diastole'] * len(volunteers)
        df_lv_peak_systole['Cardiac Phase'] = ['Systole'] * len(volunteers)
        df_lv_peak_diastole['Cardiac Phase'] = ['Diastole'] * len(volunteers)

        df_aorta_peak_systole['Region'] = ['Aorta'] * len(volunteers)
        df_aorta_peak_diastole['Region'] = ['Aorta'] * len(volunteers)
        df_lv_peak_systole['Region'] = ['LV'] * len(volunteers)
        df_lv_peak_diastole['Region'] = ['LV'] * len(volunteers)


        df_aorta_lv_frames = df_aorta_lv_frames.reset_index(drop=True)
        df_aorta_peak_systole = df_aorta_peak_systole.reset_index(drop=True)
        df_aorta_peak_diastole = df_aorta_peak_diastole.reset_index(drop=True)
        df_lv_peak_systole = df_lv_peak_systole.reset_index(drop=True)
        df_lv_peak_diastole = df_lv_peak_diastole.reset_index(drop=True)

        print("Aorta and LV at systole and diastole frames:")
        print(df_aorta_lv_frames)



        # now make a table with aorta peak systole and lv peak diastole
        # df_peak_systole_diastole = pd.concat([df_aorta_peak_systole.reset_index(drop=True), df_lv_peak_diastole.reset_index(drop=True)], axis=1)
        df_peak_systole_diastole = pd.concat([df_aorta_peak_systole, df_lv_peak_diastole], axis=0)
        # Scale RMSE columns by 1/100
        rmse_columns = [col for col in df_peak_systole_diastole.columns if 'RMSE' in col]
        df_peak_systole_diastole[rmse_columns] = df_peak_systole_diastole[rmse_columns]
        print(df_peak_systole_diastole)

        #print in latex
        print(df_peak_systole_diastole.to_latex(index=False, float_format="%.2f"))
        # save as csv
        df_peak_systole_diastole.to_csv(f'{eval_dir}/results_peak_systole_diastole.csv')

        aorta_df = df_peak_systole_diastole[df_peak_systole_diastole['Region'] == 'Aorta']
        lv_df = df_peak_systole_diastole[df_peak_systole_diastole['Region'] == 'LV']
        print("------Summary of results------")
        k_avg_aorta = (np.mean(aorta_df['k_u']) + np.mean(aorta_df['k_v']) + np.mean(aorta_df['k_w']))/3
        r2_avg_aorta = (np.mean(aorta_df['r2_u']) + np.mean(aorta_df['r2_v']) + np.mean(aorta_df['r2_w']))/3
        rmse_avg_aorta = (np.mean(aorta_df['RMSE_u']) + np.mean(aorta_df['RMSE_v']) + np.mean(aorta_df['RMSE_w']))/3

        k_avg_lv = (np.mean(lv_df['k_u']) + np.mean(lv_df['k_v']) + np.mean(lv_df['k_w']))/3
        r2_avg_lv = (np.mean(lv_df['r2_u']) + np.mean(lv_df['r2_v']) + np.mean(lv_df['r2_w']))/3
        rmse_avg_lv = (np.mean(lv_df['RMSE_u']) + np.mean(lv_df['RMSE_v']) + np.mean(lv_df['RMSE_w']))/3

        print("Aorta")
        print(f"Aorta k, R2 and RMSE across subjects at peak : {k_avg_aorta:.2f}, {r2_avg_aorta:.2f}, {rmse_avg_aorta:.3f} m/s")
        print("Left Ventricle")
        print(f"Left Ventricle k, R2 and RMSE across subjects at peak: {k_avg_lv:.2f}, {r2_avg_lv:.2f}, {rmse_avg_lv:.3f} m/s")
        print("------------------------------")


    if show_plane_velocities:
        fig, axes = plt.subplots(2, len(volunteers), figsize=(12, 5))  # Shared y-axis for each row , sharey='row'
        viridis = cm.get_cmap('viridis', 100)  # Viridis colormap

        N_frames = results_planes_aorta[volunteer]['ascending']['hr'].shape[0]
        N_frames_lr = results_planes_aorta[volunteer]['ascending']['lr'].shape[0]

        axes[0, 0].set_ylabel('velocity [m/s]', fontsize=12)
        axes[1, 0].set_ylabel('velocity [m/s]', fontsize=12)
        for i, volunteer in enumerate(volunteers): 
            t_range_in_s_hr = np.linspace(0, results_planes_aorta[volunteer]['hr_hb_duration'], N_frames)
            t_range_in_s_lr = np.linspace(0, results_planes_aorta[volunteer]['lr_hb_duration'], N_frames_lr)
            t_range_in_s_sr = np.linspace(0, results_planes_aorta[volunteer]['lr_hb_duration'], N_frames)

            # Ascending aorta (row 0)
            axes[0, i].plot(t_range_in_s_hr, results_planes_aorta[volunteer]['ascending']['hr'], '.-',  markersize=3 , label='HR', color='black')
            axes[0, i].plot(t_range_in_s_lr, results_planes_aorta[volunteer]['ascending']['lr'], '.-',  markersize=3 , label='LR', color=viridis(0.25))
            axes[0, i].plot(t_range_in_s_sr, results_planes_aorta[volunteer]['ascending']['sr'], '.--', markersize=3 , label='SR', color=viridis(0.85))

            # Descending aorta (row 1)
            axes[1, i].plot(t_range_in_s_hr, results_planes_aorta[volunteer]['descending']['hr'], '.-', markersize=3,label='HR', color='black')
            axes[1, i].plot(t_range_in_s_lr, results_planes_aorta[volunteer]['descending']['lr'], '.-', markersize=3,label='LR', color=viridis(0.25))
            axes[1, i].plot(t_range_in_s_sr, results_planes_aorta[volunteer]['descending']['sr'], '.--',markersize=3, label='SR', color=viridis(0.85))

            # Set custom ticks
            axes[0, i].set_yticks([0, 1])  # Example: Two ticks on the y-axis
            axes[1, i].set_yticks([0, 1])  # Same for descending aorta
            axes[0, i].yaxis.set_tick_params(labelleft=True, labelright=False)
            axes[1, i].yaxis.set_tick_params(labelleft=True, labelright=False)

            # Set x-ticks for all plots (example: 3 ticks)
            axes[0, i].set_xticks(np.linspace(0, 1, 3))
            axes[1, i].set_xticks(np.linspace(0, 1, 3))

            # Set x-tick labels for all plots (example: 3 ticks)
            axes[1, i].set_xlabel('Time [s]', fontsize=12)

            # Customize ticks: smaller and gray
            for ax in [axes[0, i], axes[1, i]]:
                ax.tick_params(axis='both', which='major', labelsize=8, colors='grey')

            # Grey borders for all subplots
            for ax in [axes[0, i], axes[1, i]]:
                for spine in ax.spines.values():
                    spine.set_edgecolor('grey')

        # Add a shared legend below the second row
        # handles, labels = axes[1, 0].get_legend_handles_labels()  # Retrieve legend info
        
        # fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.2))

        # One y-label for each row, in gray
        # fig.text(0.01, 0.75, 'Ascending Aorta', va='center', ha='center', rotation='vertical', fontsize=12, color='grey')
        # fig.text(0.01, 0.25, 'Descending Aorta', va='center', ha='center', rotation='vertical', fontsize=12, color='grey')
        
        
        axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=9)
        plt.tight_layout()
        # plt.subplots_adjust(bottom=0.1)  # Adjust to make space for the legend
        plt.savefig(f'{eval_dir}/All_volunteers_subplots_planes_desc_asc_aorta2.png')
        plt.savefig(f'{eval_dir}/All_volunteers_subplots_planes_desc_asc_aorta2.svg')
        plt.show()
        plt.close('all')
    #----------------------

#-------------------------------------------------------------------------------------------------------------------------
