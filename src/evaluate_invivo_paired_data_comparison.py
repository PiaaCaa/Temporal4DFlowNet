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
from utils.colors import *
import matplotlib.animation as animation
from data_specifics.INVIVOII_params import *
plt.rcParams['figure.figsize'] = [10, 8]

def plot_plane_flows(u_hr, v_hr, w_hr, u_lr, v_lr, w_lr, u_sr, v_sr, w_sr, mask,
                     cp_plane, plane_normal, idxs_nonflow_area, 
                     order_normal = [0, 1, 2], 
                     show_mask = True, save_as = None, 
                    thickness=2, factor_plane_normal = [1, 1, 1]):

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
        if save_as is not None:
            plt.savefig(f'{eval_dir}/{os.path.basename(save_as).replace("vel", "3D_plot")}')
        plt.show()

    if False:
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
    plt.figure(figsize=(6, 3))
    plt.plot(vel_hr_plane_projection,'-o', label='HR', color='black', markersize = 3)
    plt.plot(range(0, T_hr, 2), vel_lr_plane_projection,'-o', label='LR', color='forestgreen', markersize = 3)
    plt.plot(vel_sr_plane_projection, '-.', label='SR', color=KI_colors['Plum'])
    plt.legend(fontsize=16)
    plt.xlabel('Frame', fontsize=16)
    plt.ylabel('Velocity (cm/s)', fontsize=16)
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

if __name__ == "__main__":
    #'20240709-2057','Baseline', 
    # network_models = [ '20240709-2057','20241014-1443', '20241015-1033', '20241015-1047', '20241015-1050']
    # network_labels = ['Baseline','Baseline-fixed loading', 'New dataloading, no augm.', 'New dataloading, with flip augm.', 'Full augm., more noise ']

    # network_models = [ '20240709-2057', '20241015-1050', '20241018-1552', '20241018-1502']
    # network_labels = ['Baseline', 'Full augm., more noise ', 'Full augm., high and low noise train patc. ', 'low noise train patc.']
    # network_models = ['20241018-1552', '20250502-1741']
    # network_labels = ['baseline', 'targetsnrsdb1445']
    network_models = ['20241018-1552', '20250625-1642']
    network_labels = ['baseline', 'updated data augm(no swapping)', ]


    # network_models = ['20241018-1552', '20241018-1552_rerun']
    # network_labels = ['baseline', 'baseline rerun']


    # for one network evluation on multiple invivo datasets
    
    # set directories 
    # input_dir = 'data/paired_invivo/'
    lr_dir = 'Temporal4DFlowNet/data/paired_invivo'
    hr_dir = 'Temporal4DFlowNet/data/paired_invivo'
    sr_dir = 'Temporal4DFlowNet/results/in_vivo/paired_data'
    eval_base_dir  = 'Temporal4DFlowNet/results/in_vivo/paired_data/plots/baselinererun'
    

    eval_dir = f'Temporal4DFlowNet/results/in_vivo/paired_data/plots/baselinererun'
    os.makedirs(eval_dir, exist_ok=True)

    volunteers = ['v3', 'v4','v5', 'v6',  'v7'] #'v5', 'v6',



    for volunteer in volunteers:
        print(f'Evaluate volunteer {volunteer}...')

        lr_filename = f'{volunteer}_wholeheart_25mm_40ms.h5'
        hr_filename = f'{volunteer}_wholeheart_25mm_20ms.h5'

        os.makedirs(eval_dir, exist_ok=True)

        with h5py.File(f'{lr_dir}/{lr_filename}', 'r') as f:
            lr_u = np.array(f['u'])
            lr_v = np.array(f['v'])
            lr_w = np.array(f['w'])
            mask = np.array(f['mask_smooth'])
            # mask = np.array(f['mask'])
            mask_aorta = np.array(f['mask_aorta'])
            mask_lv = np.array(f['mask_LV'])

        with h5py.File(f'{hr_dir}/{hr_filename}', 'r') as f:
            hr_u = np.array(f['u'])
            hr_v = np.array(f['v'])
            hr_w = np.array(f['w'])
            if 'dx' in f:
                dx = np.array(f['dx'])
                if dx.shape[0] == 1:
                    dx = dx[0]
            else:
                dx = 2.5
        
        T_lr = lr_u.shape[0]
        T_hr = hr_u.shape[0]

        df_descening = {}
        df_ascending = {}

        for n_label, network_model in zip(network_labels, network_models):
            sr_filename = f'{volunteer}_wholeheart_25mm_40ms/{volunteer}_wholeheart_25mm_40ms_{network_model}.h5'

            with h5py.File(f'{sr_dir}/{sr_filename}', 'r') as f:
                sr_u = np.array(f['u_combined'])
                sr_v = np.array(f['v_combined'])
                sr_w = np.array(f['w_combined'])

            
            #------ local heart evaluation (aorta + lv) ------

            plane_normal_ascending = np.array(volunteer_plane_normal_ascending[f'{volunteer}_normal'])
            cp_plane_ascending = np.array(volunteer_plane_normal_ascending[f'{volunteer}_origin'])/dx

            plane_normal_descending = np.array(volunteer_plane_normal_descending[f'{volunteer}_normal'])
            cp_plane_descending = np.array(volunteer_plane_normal_descending[f'{volunteer}_origin'])/dx

            print('Evaluating ascening aorta..')
            # plot ascending aorta
            vel_hr_proj_asc, vel_sr_proj_asc, vel_lr_proj_asc = plot_plane_flows(hr_u, hr_v, hr_w, lr_u, lr_v, lr_w, sr_u, sr_v, sr_w,  mask,
                            cp_plane_ascending, plane_normal_ascending, volunteer_plot_settings[volunteer]['idxs_nonflow_area_ascending'], order_normal = volunteer_plot_settings[volunteer]['order_normal'], 
                            thickness=volunteer_plot_settings[volunteer]['thickness_ascending'], factor_plane_normal = volunteer_plot_settings[volunteer]['factor_plane_normal'], 
                            save_as = None, show_mask = False)


            # plot descending aorta
            print('Evaluating descending aorta..')
            vel_hr_proj_desc, vel_sr_proj_desc, vel_lr_proj_desc = plot_plane_flows(hr_u, hr_v, hr_w, lr_u, lr_v, lr_w, sr_u, sr_v, sr_w, mask,
                            cp_plane_descending, plane_normal_descending, volunteer_plot_settings[volunteer]['idxs_nonflow_area_descending'], order_normal = volunteer_plot_settings[volunteer]['order_normal'],  
                            thickness=volunteer_plot_settings[volunteer]['thickness_descending'], factor_plane_normal = volunteer_plot_settings[volunteer]['factor_plane_normal'], 
                            save_as = None, show_mask = False)
            
            df_descening['HR'] =  vel_hr_proj_desc
            df_descening[f'SR_{network_model}'] =  vel_sr_proj_desc
            df_descening['LR'] =  vel_lr_proj_desc

            df_ascending['HR'] =  vel_hr_proj_asc
            df_ascending[f'SR_{network_model}'] =  vel_sr_proj_asc
            df_ascending['LR'] =  vel_lr_proj_asc
        
        #now plot results for ascending and descending aorta
        plt.figure(figsize=(10, 5))
        plt.plot(df_ascending['HR'],'-o', label='HR', color='black', markersize = 3)
        plt.plot(range(0, T_hr, 2), df_ascending['LR'],'-o', label='LR', color='grey', markersize = 3)
        for n_label, network_model in zip(network_labels, network_models):
            plt.plot(df_ascending[f'SR_{network_model}'], '-.', label=f'SR_{n_label}')
        plt.legend(fontsize=16)
        plt.xlabel('Frame', fontsize=16)
        plt.ylabel('Velocity (cm/s)', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize = 16)
        plt.locator_params(axis='y', nbins=3) 
        plt.savefig(f'{eval_dir}/{volunteer}_COMPARISON_ascending_aorta_vel.png', bbox_inches='tight')
        plt.show()
            
        # descedning
        plt.figure(figsize=(10, 5))
        plt.plot(df_descening['HR'],'-o', label='HR', color='black', markersize = 3)
        plt.plot(range(0, T_hr, 2), df_descening['LR'],'-o', label='LR', color='grey', markersize = 3)
        for n_label, network_model in zip(network_labels, network_models):
            plt.plot(df_descening[f'SR_{network_model}'], '-.', label=f'SR_{n_label}')
        plt.legend(fontsize=16)
        plt.xlabel('Frame', fontsize=16)
        plt.ylabel('Velocity (cm/s)', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize = 16)
        plt.locator_params(axis='y', nbins=3) 
        plt.savefig(f'{eval_dir}/{volunteer}_COMPARISON_descending_aorta_vel.png', bbox_inches='tight')
        plt.show()
            


    # results in df

    # load data from csv files!
    df_model_comparison_aorta_lv = pd.DataFrame()
    df_model_comparison_aorta = pd.DataFrame()
    df_model_comparison_lv = pd.DataFrame()
    df_model_comparison_peak_systole_diastole = pd.DataFrame()

    for network_model in network_models:
        # Load CSV files for each network model
        if not os.path.exists(f'{eval_base_dir}/{network_model}/results_aorta_lv.csv'):
            continue
        #TODO run the evaluation instead
        df_aorta_lv = pd.read_csv(f'{eval_base_dir}/{network_model}/results_aorta_lv.csv')
        df_aorta = pd.read_csv(f'{eval_base_dir}/{network_model}/results_aorta.csv')
        df_lv = pd.read_csv(f'{eval_base_dir}/{network_model}/results_lv.csv')
        df_peak_systole_diastole = pd.read_csv(f'{eval_base_dir}/{network_model}/results_peak_systole_diastole.csv')

        # Add a column with the network model name
        df_aorta_lv['network_model'] = network_model
        df_aorta['network_model'] = network_model
        df_lv['network_model'] = network_model
        df_peak_systole_diastole['network_model'] = network_model

        # Concatenate the new data to the model comparison DataFrames
        df_model_comparison_aorta_lv = pd.concat([df_model_comparison_aorta_lv, df_aorta_lv], ignore_index=True)
        df_model_comparison_aorta = pd.concat([df_model_comparison_aorta, df_aorta], ignore_index=True)
        df_model_comparison_lv = pd.concat([df_model_comparison_lv, df_lv], ignore_index=True)
        df_model_comparison_peak_systole_diastole = pd.concat([df_model_comparison_peak_systole_diastole, df_peak_systole_diastole], ignore_index=True)
    
    # Save the model comparison DataFrames to CSV files
    # sort by volunteer and network model
    df_model_comparison_aorta_lv = df_model_comparison_aorta_lv.sort_values(by=['Volunteer', 'network_model'])
    df_model_comparison_aorta = df_model_comparison_aorta.sort_values(by=['Volunteer', 'network_model'])
    df_model_comparison_lv = df_model_comparison_lv.sort_values(by=['Volunteer', 'network_model'])
    df_model_comparison_peak_systole_diastole = df_model_comparison_peak_systole_diastole.sort_values(by=['Volunteer', 'network_model'])
    df_model_comparison_aorta_lv.to_csv(f'{eval_dir}/COMPARSION_results_aorta_lv.csv', index=False, float_format='%.3f')
    df_model_comparison_aorta.to_csv(f'{eval_dir}/COMPARSION_results_aorta.csv', index=False, float_format='%.3f')
    df_model_comparison_lv.to_csv(f'{eval_dir}/COMPARSION_results_lv.csv', index=False, float_format='%.3f')
    df_model_comparison_peak_systole_diastole.to_csv(f'{eval_dir}/COMPARSION_results_peak_systole_diastole.csv', index=False, float_format='%.3f')
    # plot results
    # plot a plot for each metric
    # plt.figure(figsize=(15, 10))
    print(df_model_comparison_aorta_lv)
    print('model comparison peak systole/disatole-----------')
    print(df_model_comparison_peak_systole_diastole.sort_values(by=['Region', 'Volunteer',]))
    metrics = ['RMSE', 'k', 'r2']

    for metric in metrics:
        plt.figure(figsize=(15, 10))
        print('output y', df_model_comparison_aorta_lv[df_model_comparison_aorta_lv['Volunteer'] == volunteer][f'{metric}_v'])
        plt.subplot(1, 3, 1)
        for i, volunteer in enumerate(volunteers):
            plt.scatter(i, df_model_comparison_aorta_lv[df_model_comparison_aorta_lv['Volunteer'] == volunteer][f'{metric}_u'], label=volunteer)
        plt.subplot(1, 3, 2)
        for i, volunteer in enumerate(volunteers):
            plt.scatter(i, df_model_comparison_aorta_lv[df_model_comparison_aorta_lv['Volunteer'] == volunteer][f'{metric}_v'], label=volunteer)
        plt.subplot(1, 3, 3)
        for i, volunteer in enumerate(volunteers):
            plt.scatter(i, df_model_comparison_aorta_lv[df_model_comparison_aorta_lv['Volunteer'] == volunteer][f'{metric}_w'], label=volunteer)
        plt.legend()
        plt.savefig(f'{eval_base_dir}/COMPARSION_{metric}_aorta_lv.png')

        






#-------------------------------------------------------------------------------------------------------------------------
