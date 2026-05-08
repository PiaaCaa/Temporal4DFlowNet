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
# from utils.vtkwriter_per_dir import uvw_mask_to_vtk

plt.rcParams['figure.figsize'] = [10, 8]


def animate_invivo_HR_pred(idx, v_orig, v_gt_fluid, v_pred, min_v, max_v, save_as, fps = 10):
    N_frames = v_orig.shape[0]
    N = 3

    fig = plt.figure(frameon=False)

    plt.subplot(1, N, 1)
    im1 = plt.imshow(v_orig[0, idx, :, :],vmin=min_v, vmax=max_v)
    plt.axis('off')
    plt.title('Native resolution')
    plt.subplot(1, N, 2)
    im2 = plt.imshow(v_gt_fluid[0, idx, :, :],vmin=min_v, vmax=max_v)
    plt.title('Native resolution (masked)')
    plt.axis('off')
    plt.subplot(1, N, 3)
    ax = plt.gca()
    im3 = plt.imshow(v_pred[0, idx, :, :],vmin=min_v, vmax=max_v)
    plt.axis('off')
    plt.title('Network prediction')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05, ) 
    plt.colorbar(im3, cax=cax, label = 'velocity (m/s)')
    plt.tight_layout()

    #initialization function: plot the background of each frame
    def init():
        im1.set_data(np.random.random((5,5)))
        im2.set_data(np.random.random((5,5)))
        im3.set_data(np.random.random((5,5)))
        return [im1, im2, im3]

    # animation function.  This is called sequentially
    def animate(i):
        
        im1.set_array(v_orig[i, idx, :, :])
        im2.set_array(v_gt_fluid[i, idx, :, :])
        im3.set_array(v_pred[i, idx, :, :])
        
        return [im1, im2, im3]

    anim = animation.FuncAnimation(fig,animate, init_func=init, frames = N_frames - 1, interval = 100) # in ms
    print('Saved animation as', save_as)
    anim.save(f'{save_as}_{fps}fps.gif', fps=fps)


def plot_slices_over_time3(gt_cube,lr_cube,  mask_cube, rel_error_cube, comparison_lst, comparison_name,timepoints, idxs,min_v, max_v,exclude_rel_error = True, save_as = "Frame_comparison.png", figsize = (30,20)):
    # assert len(timepoints) == gt_cube.shape[0] # timepoints must match size of first dimension of HR

    def row_based_idx(num_rows, num_cols, idx):
        return np.arange(1, num_rows*num_cols + 1).reshape((num_rows, num_cols)).transpose().flatten()[idx-1]

    
    T = 3 + len(comparison_lst)
    N = len(timepoints)
    if exclude_rel_error: T -= 1

    # fig = plt.figure(figsize=(10,10))
    fig, axes = plt.subplots(nrows=T, ncols=N, constrained_layout=True, figsize=figsize)

    i = 1
    #idxs = get_indices(timepoints, axis, idx)
    gt_cube = gt_cube[idxs]
    mask_cube = mask_cube[idxs]
    
    # pred_cube = pred_cube[idxs]
    #lr = lr[idxs]

    # min_v = np.quantile(gt_cube[np.where(mask_cube !=0)].flatten(), 0.01)
    # max_v = np.quantile(gt_cube[np.where(mask_cube !=0)].flatten(), 0.99)
    if not exclude_rel_error:
        rel_error_slices =rel_error_cube[idxs]#[get_slice(rel_error_cube, t, axis, idx) for t in timepoints]
        min_rel_error = np.min(np.array(rel_error_slices))
        max_rel_error = np.max(np.array(rel_error_slices))
    for j,t in enumerate(timepoints):
        
        gt_slice = gt_cube[j]
        # pred_slice = pred_cube[j]

        lr_slice = np.zeros_like(gt_slice)
        if t%2 == 0: lr_slice = lr_cube[idxs][j]#get_slice(lr_cube, t//2, axis=axis, slice_idx=idx )
        plt.subplot(T, N, row_based_idx(T, N, i))

        if t%2 == 0:
            plt.imshow(lr_slice, vmin = min_v, vmax = max_v, cmap='viridis', aspect='auto')
            plt.xticks([])
            plt.yticks([])
        if i == 1: plt.ylabel("LR", fontsize = 'small')
            
        plt.title('frame '+ str(t))
        plt.xticks([])
        plt.yticks([])
        # plt.axis('off')
        

        i +=1
        plt.subplot(T, N, row_based_idx(T, N, i))
        plt.imshow(gt_slice, vmin = min_v, vmax = max_v, cmap='viridis', aspect='auto')
        if i == 2: plt.ylabel("HR", fontsize = 'small')
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
            im = plt.imshow(comp[idxs][j], vmin = min_v, vmax = max_v, cmap='viridis', aspect='auto')
            if i-1 == (i-1)%T: plt.ylabel(name, fontsize = 'small')
            plt.xticks([])
            plt.yticks([])

        if not exclude_rel_error:
            i +=1
            plt.subplot(T, N, row_based_idx(T, N, i))
            re_img = plt.imshow(rel_error_cube[idxs][j],vmin=min_rel_error, vmax=max_rel_error, cmap='viridis',aspect='auto')
            if i-1 == (i-1)%T: plt.ylabel("abs. error", fontsize = 'small')
            plt.xticks([])
            plt.yticks([])
            if t == timepoints[-1]:
                plt.colorbar(re_img, ax = axes[-1], aspect = 10, label = 'abs. error ')

        
        i +=1
    # plt.tight_layout()
    # plt.subplots_adjust(wspace=0, hspace=0)
    fig.colorbar(im, ax=axes.ravel().tolist(), aspect = 50, label = 'velocity (m/s)')
    plt.savefig(save_as,bbox_inches='tight' )



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="My script description")
    parser.add_argument("--model", type=str, help="Optional argument to pass the name of the model")
    args = parser.parse_args()

    # Define directories and filenames
    if args.model is not None:
        network_model = args.model
    else:
        network_model = '20241018-1552' #20250620-1244

    # set directories 
    input_dir = 'Temporal4DFlowNet/data/PIA/THORAX'
    res_dir   = 'Temporal4DFlowNet/results/in_vivo/THORAX'
    eval_dir  = f'Temporal4DFlowNet/results/in_vivo/THORAX/plots/{network_model}/'
    eval_dir_overview = f'{eval_dir}/overview'
    eval_dir_detailed = f'{eval_dir}/detailed_view'

    # options for plotting
    plot_animation = True
    plot_corrplot = False
    plot_qualtimeseries = False
    plot_meanspeed = False
    tabulate_results = False
    save_as_vti = False

    # create directories if needed 
    os.makedirs(eval_dir, exist_ok = True)
    os.makedirs(eval_dir_overview, exist_ok=True)
    os.makedirs(eval_dir_detailed, exist_ok=True)
    

    dict_results = defaultdict(list)
    cases = ['P01', 'P02',  'P03', 'P04', 'P05'] 
    for case in cases:
        print('-------------------', case, '-------------------')
        in_vivo = f'{input_dir}/{case}/h5/{case}.h5' #_transformed
        in_vivo_upsampled = f'{res_dir}/{case}/{os.path.basename(in_vivo)[:-3]}_{network_model}_25Frames.h5' #results\in_vivo\P05_20230602-1701_8_4_arch_25Frames.h5
        
        name_evaluation = f'THORAX_{case}'

        #set slice index for animation
        idx_slice = np.index_exp[20, :, :]

        data_original = {}
        data_predicted = {}
        vencs = {}
        vel_colnames = ['u', 'v','w']
        mag_colnames = ['mag_u', 'mag_v', 'mag_w']
        venc_colnames = [  'u_max', 'v_max', 'w_max'] #['venc_u', 'venc_v', 'venc_w']#
        vel_plotnames = [r'$V_x$', r'$V_y$', r'$V_z$']
        mag_plotnames = [r'$M_x$', r'$M_y$', r'$M_z$']


        # load in-vivo data
        with h5py.File(in_vivo, mode = 'r' ) as p1:
            data_original['mask'] =  np.asarray(p1['mask']).squeeze()

            for vel, venc in zip(vel_colnames, venc_colnames):
                vencs[venc] = np.asarray(p1[venc])
                data_original[vel] = np.asarray(p1[vel], dtype = float).squeeze()#/np.max(vencs[venc]) #TODO change this
                print('original', vel, data_original[vel].shape)
                data_original[f'{vel}_fluid'] = np.multiply(data_original[vel], data_original['mask'])
            for mag in mag_colnames:
                data_original[mag] =  np.asarray(p1[mag]).squeeze()   

        # load prediction
        with h5py.File(in_vivo_upsampled, mode = 'r' ) as h_pred:
            for vel, venc in zip(vel_colnames, venc_colnames):
                data_predicted[vel] = np.asarray(h_pred[f'{vel}_combined']) #/np.max(vencs[venc]) 

                # add information considering only the fluid regions  
                if data_predicted[vel].shape[0] == data_original[vel].shape[0]:
                    data_predicted[f'{vel}_fluid'] = np.multiply(data_predicted[vel], data_original['mask'])
                    data_predicted['mask'] = data_original['mask']
                    
                elif  data_predicted[vel].shape[0] == 2*data_original[vel].shape[0]:
                    t, x, y, z = data_original['mask'].shape
                    data_predicted['mask']= np.zeros((2*t,x, y, z ))
                    data_predicted['mask'][::2, :, :, :] = data_original['mask']
                    data_predicted['mask'][1::2, :, :, :] = data_original['mask']
                    data_predicted[f'{vel}_fluid'] = np.multiply(data_predicted[vel], data_predicted['mask'])

        print('Shape of predicted data and original data:', data_predicted['u'].shape, data_original['u'].shape)
        N_frames = data_original['u'].shape[0]
        print("Max val:", np.max(data_original['u']), np.max(data_original['v']), np.max(data_original['w']))
        print("Min val:", np.min(data_original['u']), np.min(data_original['v']), np.min(data_original['w']))

        N_frames_input_data = data_original['u'].shape[0]
        N_frames_pred_data = data_predicted['u'].shape[0]

        super_resolved_prediction = False if N_frames_input_data == N_frames_pred_data else True
        if super_resolved_prediction: print('Evaluation of higher resolved velocity field')
        if super_resolved_prediction: print('Prediction increases temporal resolution of original data by 2x. (super resolved) ..')

        if super_resolved_prediction:
            name_evaluation = f'{name_evaluation}_SR'
        else:
            name_evaluation = f'{name_evaluation}_reconstructed_resolution'

        #find lower and higher values to display velocity fields
        min_v = {}
        max_v = {}
        for vel in vel_colnames:
            min_v[vel] = np.quantile(data_original[vel][np.where(data_original['mask'] !=0)].flatten(), 0.01)
            max_v[vel] = np.quantile(data_original[vel][np.where(data_original['mask'] !=0)].flatten(), 0.99)

        max_V = np.max([max_v['u'], max_v['v'], max_v['w']])
        min_V = np.min([min_v['u'], min_v['v'], min_v['w']])
        #-----------------save img slices over time---------------------
        if plot_qualtimeseries:
            time_point = 10
            slice_idx = np.index_exp[time_point, 18, :, :]
            fig, axes = plt.subplots(nrows=2, ncols=3, ) #constrained_layout=True
            for i, (vel, mag, nam_vel, name_mag) in enumerate(zip(vel_colnames, mag_colnames, vel_plotnames, mag_plotnames)):
                plt.subplot(2, 3, i+1)
                ax = plt.gca()
                im1 = plt.imshow(data_original[vel][slice_idx], vmin = min_V, vmax = max_V)
                plt.axis('off')
                plt.title(nam_vel)
                # plt.savefig(f'{eval_dir}/{c}_{vel}_Invivo_Original_Frame{time_point}.png', bbox_inches='tight')
                
                if i == 2:
                    # plt.colorbar(im1, ax = axes[0], aspect = 25, label = 'velocity')
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im1, cax=cax, label = 'velocity [m/s]')

                plt.subplot(2, 3, 4+i)
                ax = plt.gca()
                im2 = plt.imshow(data_original[mag][slice_idx], cmap = 'Greys_r')
                plt.axis('off')
                plt.title(name_mag)
                # plt.savefig(f'{eval_dir}/{c}_{mag}_Invivo_Original_Frame{time_point}.png', bbox_inches='tight')
                if i == 2:
                    # plt.colorbar(im2, ax = axes[-1], aspect = 25, label = 'magnitude')
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im2, cax=cax, label = 'magnitude')

        
            plt.tight_layout()
            plt.savefig(f'{eval_dir_detailed}/{case}_SUBPLOT_Invivo_Original_Frame{time_point}.png', bbox_inches='tight')
            plt.show()
        
        #--------------------calculate mean speed --------------------------
        magn = np.sqrt(data_original['mag_u']**2 + data_original['mag_v']**2 + data_original['mag_w']**2)
        speed = np.sqrt(data_original['u']**2 + data_original['v']**2 + data_original['w']**2)
        pc_mri = np.multiply(magn, speed)
        data_original['mean_speed']  = calculate_mean_speed(data_original["u_fluid"], data_original["v_fluid"] , data_original["w_fluid"], data_original["mask"])
        data_predicted['mean_speed'] = calculate_mean_speed(data_predicted["u_fluid"], data_predicted["v_fluid"] , data_predicted["w_fluid"], data_predicted["mask"])
        
        T_peak_flow_frame = np.argmax(data_original['mean_speed'])
        synthesized_peak_flow_frame = T_peak_flow_frame.copy() 
        # take next frame if peak flow frame included in lr data
        if synthesized_peak_flow_frame % 2 == 0: 
            if data_original['mean_speed'][synthesized_peak_flow_frame-1] > data_original['mean_speed'][synthesized_peak_flow_frame+1]:
                synthesized_peak_flow_frame -=1
            else:
                synthesized_peak_flow_frame +=1
        print('Synthesized peak flow frame:', synthesized_peak_flow_frame, 'Peak flow frame:', T_peak_flow_frame)


        if plot_meanspeed:
            #-------------mean speed plot---------------------
            plt.figure(figsize=(7, 4))
            step_pred = 0.5 if super_resolved_prediction else 1
            

            frame_range_input = np.arange(0, N_frames_input_data)#np.linspace(0, data_1['u'].shape[0]-1,  data_1['u'].shape[0])
            frame_range_predicted = np.arange(0, N_frames_input_data, step_pred)#np.linspace(0, data_1['u'].shape[0], data_predicted['u'].shape[0])

            plt.title('Mean speed')
            plt.plot(frame_range_predicted, data_predicted['mean_speed'], '.-', label = 'prediction', color= KI_colors['Blue'])
            plt.plot(frame_range_input, data_original['mean_speed'],'--', label = 'noisy input data', color= 'black')
            if not super_resolved_prediction:
                plt.plot(frame_range_input[::2], data_original['mean_speed'][::2],'.',  label ='sample points',  color= 'black')
            else:
                plt.plot(frame_range_input, data_original['mean_speed'],'.',  label ='sample points',  color= 'black')
            plt.xlabel("Frame")
            plt.ylabel("Mean speed (cm/s)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{eval_dir_overview}/{name_evaluation}_meanspeed.png')
            plt.show()

            # plot mean velocity over time
            plt.figure(figsize=(15, 5))
            colors = ['r', 'g', 'b']
            for i, vel in enumerate(vel_colnames):
                plt.subplot(1,3, i+1)
                plt.plot(frame_range_predicted, np.mean(data_predicted[vel], axis = (1, 2, 3), where=data_original['mask'].astype(bool)), '-', label = f'sr {vel}', color = 'black')
                plt.plot(frame_range_input, np.mean(data_original[vel], axis = (1, 2, 3), where=data_original['mask'].astype(bool)), '--', label = f'native res {vel}', color = colors[i])
                if not super_resolved_prediction:
                    plt.plot(frame_range_input[::2], np.mean(data_original[vel][::2], axis=(1, 2, 3), where=data_original['mask'][::2].astype(bool)),'.',  label ='sample points',  color= colors[i])
                else:
                    plt.plot(frame_range_input, np.mean(data_original[vel], axis=(1, 2, 3), where=data_original['mask'].astype(bool)),'.',  label ='sample points',  color= colors[i])
                plt.xlabel("Frame")
                plt.ylabel(f"Mean velocity (cm/s)")
                plt.legend()
                plt.title(f'Mean {vel} over time')
            plt.tight_layout()
            plt.savefig(f'{eval_dir_overview}/{name_evaluation}_mean_velocity_VxVyVz.png')


            plt.show()
        
        if save_as_vti:
            print("Save as vti files..")
            spacing = [3, 3, 3]

            os.makedirs(f'{eval_dir}/vti', exist_ok = True)

            for t in range(N_frames):
                
                output_filepath = f'{eval_dir}/vti/{case}_SR_{network_model}_frame{t}_uvw.vti'
                if os.path.isfile(output_filepath):
                    print(f'File {output_filepath} already exists')
                else:
                    uvw_mask_to_vtk((data_predicted['u'][t],data_predicted['v'][t],data_predicted['w'][t]),data_predicted['mask'][t], spacing, output_filepath, include_mask = True)
                    
                if True: 
                    output_filepath = f'{eval_dir}/vti/{case}_original_data_frame{t}_uvw.vti'
                    if not os.path.isfile(output_filepath):
                        uvw_mask_to_vtk((data_original['u'][t],data_original['v'][t],data_original['w'][t]),data_original['mask'][t], spacing, output_filepath, include_mask = True)

                    output_filepath = f'{eval_dir}/vti/{case}_SR_{network_model}_error_frame{t}_uvw.vti'
                    if not os.path.isfile(output_filepath):
                        uvw_mask_to_vtk((data_original['u'][t] - data_predicted['u'][t],data_original['v'][t] -data_predicted['v'][t],data_original['w'][t] -data_predicted['v'][t]),data_original['mask'][t], spacing, output_filepath, include_mask = True)

        
        if not super_resolved_prediction:
            #---------------correlation + k + r^2 plots -------------------
            if plot_corrplot: 
                font = { # 'weight' : 'bold',
                    'size'   : 12}

                matplotlib.rc('font', **font)
                #---------------linear regression plot and its parameters-------------------------------
                
                bounds, core_mask = get_boundaries(data_original['mask'])
                plt.figure(figsize=(15, 5))
                plot_correlation_nobounds(data_original, data_predicted, frame_idx=synthesized_peak_flow_frame,show_text=True,  save_as=f'{eval_dir_detailed}/{name_evaluation}_correlation_synthetic_peakframe{synthesized_peak_flow_frame}')
                plot_correlation_nobounds(data_original, data_predicted, frame_idx=T_peak_flow_frame,show_text=True,  save_as=f'{eval_dir_detailed}/{name_evaluation}_correlation_peakframe{T_peak_flow_frame}')

                # plot_correlation_nobounds(data_original, data_predicted, frame_idx=3,show_text=True,  save_as=f'{eval_dir_detailed}/{name_evaluation}_correlation_frame3_')
                # plot_correlation_nobounds(data_original, data_predicted, frame_idx=5,show_text=True,  save_as=f'{eval_dir_detailed}/{name_evaluation}_correlation_frame5')
                # plot_correlation_nobounds(data_original, data_predicted, frame_idx=7,show_text=True,  save_as=f'{eval_dir_detailed}/{name_evaluation}_correlation_frame7')
                # plot_correlation_nobounds(data_original, data_predicted, frame_idx=9,show_text=True,  save_as=f'{eval_dir_detailed}/{name_evaluation}_correlation_frame9')
                # plot_correlation_nobounds(data_original, data_predicted, frame_idx=11,show_text=True, save_as=f'{eval_dir_detailed}/{name_evaluation}_correlation_frame11')

                bounds_mask = bounds.copy()
                core_mask = data_original['mask'] - bounds_mask


                #plot k and r^2 values
                # plot_k_r2_vals(frames, k,k_bounds, r2,  r2_bounds, peak_flow_frame, name_evaluation, eval_dir)
                plot_k_r2_vals(data_original, data_predicted, bounds, synthesized_peak_flow_frame,color_b = KI_colors['Plum'] , save_as= f'{eval_dir_detailed}/{name_evaluation}_k_r2_synthetic_peakframe{synthesized_peak_flow_frame}_{name_evaluation}')

                k, r2 = calculate_and_plot_k_r2_vals_nobounds(data_original, data_predicted,data_original['mask'], synthesized_peak_flow_frame,figsize=(15, 5), save_as = None)

                fig2, axs1 = plot_k_r2_vals_nobounds(k, r2, synthesized_peak_flow_frame, figsize = (15, 5),exclude_tbounds = True)
                fig1 = plot_correlation_nobounds(data_original, data_predicted, synthesized_peak_flow_frame, show_text = True)
                
                # Merge the two figures into a single figure
                fig = plt.figure(figsize=(15, 10))
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)
                ax1.axis('off')  # Turn off the axes for the first subplot
                ax2.axis('off')  # Turn off the axes for the second subplot
                fig.subplots_adjust(wspace=0)  # Adjust the spacing between subplots
                fig1.subplots_adjust(wspace=0)  # Adjust the spacing between subplots in the first figure
                fig2.subplots_adjust(wspace=0)  # Adjust the spacing between subplots in the second figure
                ax1.imshow(fig1.canvas.renderer._renderer)
                ax2.imshow(fig2.canvas.renderer._renderer)
                plt.tight_layout()
                plt.savefig(f'{eval_dir_overview}/{name_evaluation}_{network_model}_correlation_k_r2_synpeakframe{synthesized_peak_flow_frame}.png')
                


            if tabulate_results:
                #print mean k and r^2 values
                dict_intermediate_results = defaultdict(list)

                for i, vel in enumerate(vel_colnames):
                    for t in range(N_frames):
                        k, r2 = calculate_k_R2( data_predicted[vel][t], data_original[vel][t], data_original['mask'][t])
                        
                        dict_intermediate_results[f'k_{vel}'].append(k)
                        dict_intermediate_results[f'R2_{vel}'].append(r2)
                        dict_intermediate_results[f'|1-k|_{vel}'].append(np.abs(1-k))
                    rmse = calculate_rmse(data_predicted[vel], data_original[vel], data_original['mask'], return_std=False)
                    mae = np.mean(np.abs(data_predicted[vel] - data_original[vel]), axis=(1, 2, 3), where=data_original['mask'].astype(bool))
                    dict_intermediate_results[f'rmse_{vel}'] = rmse
                    dict_intermediate_results[f'mae_{vel}'] = mae
                
                dict_results['Patient'].append(case)
                for key in dict_intermediate_results.keys():
                    dict_results[key].append(np.mean(dict_intermediate_results[key]))
                    dict_results[f'{key}_std'].append(np.std(dict_intermediate_results[key]))
                    print('key', key,dict_results[key])
                    print('key', len(dict_results[key]))
                    dict_results[f'{key}_peak'].append(dict_intermediate_results[key][synthesized_peak_flow_frame])

                

                print('Mean k values over all patients:', np.average(dict_results[f'k_u']), np.average(dict_results[f'k_v']), np.average(dict_results[f'k_w']))
                print('Mean R2 values over all patients:', np.average(dict_results[f'R2_u']), np.average(dict_results[f'R2_v']), np.average(dict_results[f'R2_w']))
                print('Eval peak flow frame:', synthesized_peak_flow_frame, T_peak_flow_frame)

            #-----------------plot slices over time---------------------
            if plot_qualtimeseries:
                

                time_points = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                idx_cube = np.index_exp[time_points[0]:time_points[-1]+1, 30, 15:90, 25:100]
                idx_cube_lr = np.index_exp[time_points[0]//2:time_points[-1]//2+1, 30, 15:90, 25:100]
                # idx_cube = np.index_exp[time_points[0]:time_points[-1]+1, 30, 30:60, 60:90]
                # idx_cube_lr = np.index_exp[time_points[0]//2:time_points[-1]//2+1, 30, 30:60, 60:90]

                plot_correlation_nobounds(data_original, data_predicted, frame_idx=3,show_text=True,  save_as=f'{eval_dir_detailed}/{name_evaluation}_Correlation_frame3', figsize = (4, 4), show_title=False)
                plot_correlation_nobounds(data_original, data_predicted, frame_idx=5,show_text=True,  save_as=f'{eval_dir_detailed}/{name_evaluation}_Correlation_frame5', figsize = (4, 4), show_title=False)
                plot_correlation_nobounds(data_original, data_predicted, frame_idx=7,show_text=True,  save_as=f'{eval_dir_detailed}/{name_evaluation}_Correlation_frame7', figsize = (4, 4), show_title=False)
                plot_correlation_nobounds(data_original, data_predicted, frame_idx=9,show_text=True,  save_as=f'{eval_dir_detailed}/{name_evaluation}_Correlation_frame9', figsize = (4, 4), show_title=False)
                plot_correlation_nobounds(data_original, data_predicted, frame_idx=11,show_text=True, save_as=f'{eval_dir_detailed}/{name_evaluation}_Correlation_frame11',figsize = (4, 4), show_title=False)
                
                # idx_cube = np.index_exp[:, :, 15:70, 20:100]
                # plot_slices_over_time1(gt_cube,lr_cube,  mask_cube, rel_error_cube, comparison_lst, comparison_name, timepoints, axis, idx,min_v, max_v,exclude_rel_error = True, save_as = "Frame_comparison.png", figsize = (30,20)):
                # plot_slices_over_time1(data_original['u'][idx_cube], data_original['u'][::2][idx_cube], data_original['mask'][idx_cube] , None, [data_predicted['u'][idx_cube], ], ['SR',], time_points, 0, 30,  min_v['u'], max_v['u'], exclude_rel_error = True, save_as = f"{eval_dir}/{name_evaluation}_u_comparison.png", figsize = (12, 4))
                # plot_comparison_temporal_slices(data_original, data_predicted, idx_slice, eval_dir, name_evaluation, super_resolved_prediction, min_v, max_v, exclude_rel_error = True)
                # plt.show()
                # def plot_qual_comparsion(gt_cube,lr_cube,  pred_cube,mask_cube, abserror_cube, comparison_lst, comparison_name, timepoints, min_v, max_v, include_error = False,  figsize = (10, 10), save_as = "Qualitative_frame_seq.png"):

                # plot_qual_comparsion(data_original['u'][idx_cube], data_original['u'][::2][idx_cube_lr],data_predicted['u'][idx_cube], data_original['mask'][idx_cube], None, [], [], time_points, None, None, include_error = False,  figsize = (12, 3), save_as = f"{eval_dir}/{name_evaluation}_u_qual_comparison_smaller_patch.png")
                plot_qual_comparsion(data_original['u'][idx_cube], data_original['u'][::2][idx_cube_lr],data_predicted['u'][idx_cube], data_original['mask'][idx_cube], None, [], [], time_points, None, None, include_error = False,  figsize = (13, 3.5),fontsize_lr=8, save_as = f"{eval_dir_detailed}/{name_evaluation}_u_qual_comparison_new.png")

                # plot_qual_comparsion(data_original['u'][idx_cube], data_original['u'][::2][idx_cube_lr],data_predicted['u'][idx_cube], data_original['mask'][idx_cube], data_original['mask'][idx_cube]*np.abs(data_original['u'][idx_cube] -data_predicted['u'][idx_cube]), [], [], time_points, None, None, include_error = True,  figsize = (12, 4), save_as = f"{eval_dir}/{name_evaluation}_u_qual_comparison_smaller_patch_abs_error.png")
                plot_qual_comparsion(data_original['u'][idx_cube], data_original['u'][::2][idx_cube_lr],data_predicted['u'][idx_cube], data_original['mask'][idx_cube], data_original['mask'][idx_cube]*np.abs(data_original['u'][idx_cube] -data_predicted['u'][idx_cube]), [], [], time_points, None, None, include_error = True,  figsize = (12, 4), save_as = f"{eval_dir_detailed}/{name_evaluation}_u_qual_comparison_new_abs_error.png")


                plt.show()
                
        #---------create animation------------------------
        if plot_animation:
            
            inlcude_orig_data = False
            eval_gifs = f'{eval_dir_detailed}/gifs'
            os.makedirs(eval_gifs, exist_ok = True)

            fps_anim = 10
            fps_pred = fps_anim*2 if super_resolved_prediction else fps_anim
            if inlcude_orig_data: 
                if not os.path.exists(f'{eval_dir}/Animate_invivo_case00{case}_mag_{fps_anim}fps.gif'):
                    print('Create video of original data..')
                    animate_data_over_time_gif(idx_slice, magn, 0, np.quantile(magn, 0.99),      fps = fps_anim , save_as = f'{eval_gifs}/{name_evaluation}_animate_{case}_mag', colormap='Greys_r' )
                    animate_data_over_time_gif(idx_slice, data_original['mask'], 0, 1,           fps = fps_anim , save_as = f'{eval_gifs}/{name_evaluation}_animate_{case}_mask', colormap='Greys' )
                    animate_data_over_time_gif(idx_slice, data_original['u'], min_V, max_V,      fps = fps_anim , save_as = f'{eval_gifs}/{name_evaluation}_animate_{case}_u_gt')
                    animate_data_over_time_gif(idx_slice, data_original['v'], min_V, max_V,      fps = fps_anim , save_as = f'{eval_gifs}/{name_evaluation}_animate_{case}_v_gt')
                    animate_data_over_time_gif(idx_slice, data_original['w'], min_V, max_V,      fps = fps_anim , save_as = f'{eval_gifs}/{name_evaluation}_animate_{case}_w_gt')
                    animate_data_over_time_gif(idx_slice, data_original['u_fluid'], min_V, max_V,fps = fps_anim , save_as = f'{eval_gifs}/{name_evaluation}_animate_{case}_u_gt_fluid')
                    animate_data_over_time_gif(idx_slice, data_original['v_fluid'], min_V, max_V,fps = fps_anim , save_as = f'{eval_gifs}/{name_evaluation}_animate_{case}_v_gt_fluid')
                    animate_data_over_time_gif(idx_slice, data_original['w_fluid'], min_V, max_V,fps = fps_anim , save_as = f'{eval_gifs}/{name_evaluation}_animate_{case}_w_gt_fluid')

            animate_data_over_time_gif(idx_slice, data_original['u'][::2], min_V, max_V, fps = fps_anim//2 , save_as = f'{eval_gifs}/{name_evaluation}_animate_{case}_u_gt_lr_downsampled')
            animate_data_over_time_gif(idx_slice, data_original['v'][::2], min_V, max_V, fps = fps_anim//2 , save_as = f'{eval_gifs}/{name_evaluation}_animate_{case}_v_gt_lr_downsampled')
            animate_data_over_time_gif(idx_slice, data_original['w'][::2], min_V, max_V, fps = fps_anim//2 , save_as = f'{eval_gifs}/{name_evaluation}_animate_{case}_w_gt_lr_downsampled')

            if True: 
                print('Create video of predicted data..')
                animate_data_over_time_gif(idx_slice, data_original['mask'], 0, 1,         fps = fps_anim , save_as = f'{eval_gifs}/{name_evaluation}_animate_{case}_mask', colormap='Greys')
                animate_data_over_time_gif(idx_slice, data_predicted['u'], min_V, max_V, fps = fps_pred , save_as = f'{eval_gifs}/{name_evaluation}_animate_u_pred')
                animate_data_over_time_gif(idx_slice, data_predicted['v'], min_V, max_V, fps = fps_pred , save_as = f'{eval_gifs}/{name_evaluation}_animate_v_pred')
                animate_data_over_time_gif(idx_slice, data_predicted['w'], min_V, max_V, fps = fps_pred , save_as = f'{eval_gifs}/{name_evaluation}_animate_w_pred')

                animate_data_over_time_gif(idx_slice, data_predicted['u'], min_V, max_V, fps = fps_pred , save_as = f'{eval_gifs}/{name_evaluation}_animate_u_pred_inclColorbar', show_colorbar=True)
                for vel in vel_colnames:
                    plt.imshow(data_predicted[vel][0, idx_slice[0], idx_slice[1], idx_slice[2]],vmin=min_v[vel], vmax=max_v[vel])
                    plt.axis('off')
                    plt.savefig(f'{eval_dir}/{name_evaluation}_{vel}_pred.png', bbox_inches='tight')

                    # also gt
                    plt.imshow(data_original[vel][0, idx_slice[0], idx_slice[1], idx_slice[2]],vmin=min_v[vel], vmax=max_v[vel])
                    plt.axis('off')
                    plt.savefig(f'{eval_dir}/{name_evaluation}_{vel}_gt.png', bbox_inches='tight')
                    plt.close()
                
                # animate absolute difference
            max_abs_err = np.max([np.abs(data_original['u'] - data_predicted['u'])[idx_slice]*data_original['mask'][idx_slice], 
                                  np.abs(data_original['v'] - data_predicted['v'])[idx_slice]*data_original['mask'][idx_slice], 
                                  np.abs(data_original['w'] - data_predicted['w'])[idx_slice]]*data_original['mask'][idx_slice])

            animate_data_over_time_gif(idx_slice, np.abs(data_original['u']-data_predicted['u'])*data_original['mask'], 0, max_abs_err, fps = fps_pred , save_as = f'{eval_gifs}/{name_evaluation}_animate_u_abs_error')
            animate_data_over_time_gif(idx_slice, np.abs(data_original['v']-data_predicted['v'])*data_original['mask'], 0, max_abs_err, fps = fps_pred , save_as = f'{eval_gifs}/{name_evaluation}_animate_v_abs_error')
            animate_data_over_time_gif(idx_slice, np.abs(data_original['w']-data_predicted['w'])*data_original['mask'], 0, max_abs_err, fps = fps_pred , save_as = f'{eval_gifs}/{name_evaluation}_animate_w_abs_error')

            animate_data_over_time_gif(idx_slice, np.abs(data_original['u']-data_predicted['u'])*data_original['mask'], 0, max_abs_err, fps = fps_pred , save_as = f'{eval_gifs}/{name_evaluation}_animate_u_abs_error_inclColorbar', show_colorbar=True)



            if not super_resolved_prediction:
                animate_invivo_HR_pred(18, data_original['u'], data_original['u_fluid'], data_predicted['u'], min_V, max_V, f'{eval_dir}/{name_evaluation}_animation_{name_evaluation}_3view_gt_fluid_pred_u', fps = fps_pred)
                animate_invivo_HR_pred(18, data_original['v'], data_original['v_fluid'], data_predicted['v'], min_V, max_V, f'{eval_dir}/{name_evaluation}_animation_{name_evaluation}_3view_gt_fluid_pred_v', fps = fps_pred)
                animate_invivo_HR_pred(18, data_original['w'], data_original['w_fluid'], data_predicted['w'], min_V, max_V, f'{eval_dir}/{name_evaluation}_animation_{name_evaluation}_3view_gt_fluid_pred_w', fps = fps_pred)

            #     animate_invivo_HR_pred(idx, v_orig, v_gt_fluid, v_pred, min_v, max_v, save_as, fps = 10)
            # animate_invivo_HR_pred(idx_slice, data_original['u'][::2], hr, pred, vel,min_v, max_v, save_as, fps = 10)

    if tabulate_results:
        # Display tabular for all subject evaluations
        result_df = pd.DataFrame(dict_results).round(2)
        print(result_df.columns)
        rearaanged_columns = ['Patient','rmse_u','rmse_u_std', 'rmse_v','rmse_v_std', 'rmse_w' ,'rmse_w_std', '|1-k|_u', '|1-k|_u_std', '|1-k|_v', '|1-k|_v_std', '|1-k|_w', '|1-k|_w_std','R2_u', 'R2_u_std', 'R2_v', 'R2_v_std', 'R2_w', 'R2_w_std','rmse_u_peak', 'rmse_v_peak', 'rmse_w_peak', 'k_u_peak',  'k_v_peak',  'k_w_peak', 'R2_u_peak',  'R2_v_peak', 'R2_w_peak']
        result_df = result_df[rearaanged_columns]

        # print(result_df.to_latex(index=False, float_format="%.2f"))
        result_df.to_csv(f'{eval_dir_overview}/Results_{network_model}_k_r2_avg_peak.csv', index = False)


        k_avg_peak = (np.mean(dict_results['k_u_peak']) + np.mean(dict_results['k_v_peak']) + np.mean(dict_results['k_w_peak']))/3
        r2_avg_peak = (np.mean(dict_results['R2_u_peak']) + np.mean(dict_results['R2_v_peak']) + np.mean(dict_results['R2_w_peak']))/3
        rmse_avg_peak = (np.mean(dict_results['rmse_u_peak']) + np.mean(dict_results['rmse_v_peak']) + np.mean(dict_results['rmse_w_peak']))/3
        rmse_avg_all = (np.mean(dict_results['rmse_u']) + np.mean(dict_results['rmse_v']) + np.mean(dict_results['rmse_w']))/3
        print("-----------------Overall results for all patients:-------------------")
        print(f"k Average of peak flow frames: {k_avg_peak:.2f}")
        print(f"R2 Average of peak flow frames: {r2_avg_peak:.2f}")
        print(f"RMSE across subjects {rmse_avg_all:.2f}")
        print(f"RMSE avg at peak systole: {rmse_avg_peak:.2f}")
