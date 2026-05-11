"""
4DFlowNet: Super Resolution ResNet
Author: Edward Ferdian
Date:   14/06/2019
"""
import pickle
import tensorflow as tf
import numpy as np
import datetime
import time

import shutil
import os
from .Temporal4DFlowNetModel import T4DFlowNet
from . import h5util

def calculate_time_elapsed(start):
    '''
        This function calculates the time elapsed
        Input:  
            start = start time
        Output: 
            hrs, mins, secs = time elapsed in hours, minutes, seconds format
    '''
    end = time.time()
    hrs = (end-start)//60//60
    mins = ((end-start) - hrs*60*60)//60
    secs = int((end-start) - mins*60 - hrs*60*60)

    return hrs, mins, secs

def log_to_file(filepath, msg):
    with open(filepath, 'a') as f:
        f.write(msg)


class  TrainerController:
    # constructor
    def __init__(self, patch_size, res_increase, 
                # Training params
                initial_learning_rate=1e-4, 
                lr_decay_epochs=0,
                L2_regularization=0.001,
                # Network params
                quicksave_enable=True, 
                network_name='4DFlowNet', 
                n_low_resblock=8, 
                n_hi_resblock=4, 
                low_res_block='resnet_block', 
                high_res_block='resnet_block', 
                upsampling_block='default', 
                post_processing_block=None, 
                include_mag_input=True,
                # Loss params
                alpha=0.8,
                epsilon=1,
                weighting_fluid=1.0,
                weighting_non_fluid=1.0,
                separate_mse=True,
                loss_type='mse',
                use_directional_loss=True):
        """
            TrainerController constructor
            Setup all the placeholders, network graph, loss functions and optimizer here.

            Loss params:
                alpha:                Weighting between MSE and directional loss (default 0.8)
                epsilon:              Minimum pixel count to avoid division by zero (default 1)
                weighting_fluid:      Weighting for fluid region loss (default 1.0)
                weighting_non_fluid:  Weighting for non-fluid region loss (default 1.0)
                separate_mse:         If True, compute fluid and non-fluid loss separately (default True)
                loss_type:            Loss function type: ' 'mse', 'mae', 'huber' (default 'mse')
                use_directional_loss:  If True, use directional loss (default True)

            Training params:
                initial_learning_rate: Initial learning rate (default 1e-4)
                lr_decay_epochs:       Decay learning rate every N epochs, 0 to disable (default 0)
                L2_regularization:     L2 regularization weight, 0 to disable (default 0.001)
        """
        print("Initializing TrainerController...")
        
        # Loss weights
        self.lr_decay_epoch = lr_decay_epochs
        self.L2_regularization = L2_regularization

        # Loss params
        self.alpha = alpha
        self.epsilon = epsilon
        self.weighting_fluid = weighting_fluid
        self.weighting_non_fluid = weighting_non_fluid
        self.separate_mse = separate_mse
        self.loss_type = loss_type
        self.use_directional_loss = use_directional_loss

        # General param
        self.res_increase = res_increase
        
        # Training params
        self.QUICKSAVE_ENABLED = quicksave_enable
        
        # Network
        self.network_name = network_name
        self.low_res_block = low_res_block
        self.high_res_block = high_res_block
        self.post_processing_block = post_processing_block
        self.including_mag_input = include_mag_input

        t_patchsize, s1_ps, s2_ps = patch_size
        input_shape = (t_patchsize, s1_ps, s2_ps, 1)

        # Prepare Input 
        u     = tf.keras.layers.Input(shape=input_shape, name='u')
        v     = tf.keras.layers.Input(shape=input_shape, name='v')
        w     = tf.keras.layers.Input(shape=input_shape, name='w')
        u_mag = tf.keras.layers.Input(shape=input_shape, name='u_mag')
        v_mag = tf.keras.layers.Input(shape=input_shape, name='v_mag')
        w_mag = tf.keras.layers.Input(shape=input_shape, name='w_mag')

        input_layer = [u, v, w, u_mag, v_mag, w_mag] if include_mag_input else [u, v, w]

        net = T4DFlowNet(res_increase, low_res_block=low_res_block, high_res_block=high_res_block, 
                        upsampling_block=upsampling_block, post_processing_block=self.post_processing_block)
        self.predictions = net.build_network(u, v, w, u_mag, v_mag, w_mag, n_low_resblock, n_hi_resblock, include_mag=include_mag_input)
        self.model = tf.keras.Model(input_layer, self.predictions)
        self.model.summary()

        # ===== Metrics =====
        self.loss_metrics = dict([
            ('train_loss',     tf.keras.metrics.Mean(name='train_loss')),
            ('val_loss',       tf.keras.metrics.Mean(name='val_loss')),
            ('train_accuracy', tf.keras.metrics.Mean(name='train_accuracy')),
            ('val_accuracy',   tf.keras.metrics.Mean(name='val_accuracy')),
            ('train_mse',      tf.keras.metrics.Mean(name='train_mse')),
            ('train_cos_sim',  tf.keras.metrics.Mean(name='train_cos_sim')), 
            ('val_mse',        tf.keras.metrics.Mean(name='val_mse')),
            ('train_div',      tf.keras.metrics.Mean(name='train_div')),
            ('val_div',        tf.keras.metrics.Mean(name='val_div')),
            ('val_cos_sim',    tf.keras.metrics.Mean(name='val_cos_sim')), 
            ('l2_reg_loss',    tf.keras.metrics.Mean(name='l2_reg_loss')),
        ])
        
        self.accuracy_metric = 'val_loss'

        print(f"Loss type: {self.loss_type}")
        print(f"Alpha: {self.alpha}, Epsilon: {self.epsilon}")
        print(f"Weighting fluid: {self.weighting_fluid}, non-fluid: {self.weighting_non_fluid}")
        print(f"Separate MSE: {self.separate_mse}")
        print(f"L2 regularization: {self.L2_regularization}")
        print(f"Accuracy metric: {self.accuracy_metric}")

        # Optimizer
        self.learning_rate = initial_learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Gradient info
        self.gradient_norm = []
        self.gradient_threshold = 1
        self.gradient_over_threshold = False
        

    # def save_latest_model(self, epoch):
    #     if epoch > 0 and epoch % 10 == 0:
    #         self.model.save(f'{self.model_path}-latest.h5')
    #         message = f'Saving current model - {time.ctime()}\n'
    #         print(message)

    #--------LOSS FUNCTIONS ------------
    def loss_function(self, y_true, y_pred, mask):
        """
            Calculate Total Loss function.

            Loss type is controlled by self.loss_type:
                'mse'          - L2 error only
                'mae'          - L1 error only
                'huber'        - Huber loss only
                'l1_projected' - L2 error + directional L1 mutually projected loss, 
                                weighted by self.alpha
            
            Fluid/non-fluid separation is controlled by self.separate_mse.
        """
        u, v, w          = y_true[..., 0], y_true[..., 1], y_true[..., 2]
        u_pred, v_pred, w_pred = y_pred[..., 0], y_pred[..., 1], y_pred[..., 2]

        # === Select error metric ===
        if self.loss_type == 'mae':
            s_err = self.calculate_l1_error(u, v, w, u_pred, v_pred, w_pred)
        elif self.loss_type == 'huber':
            s_err = self.calculate_huber_loss(u, v, w, u_pred, v_pred, w_pred)
        else:  # 'mse' or 'l1_projected' both use L2 as the base error
            s_err = self.calculate_l2_error(u, v, w, u_pred, v_pred, w_pred)

        # === Compute base loss (fluid/non-fluid separated or combined) ===
        if self.separate_mse:
            non_fluid_mask = tf.cast(tf.less(mask, tf.constant(0.5)), dtype=tf.float32)

            fluid_loss = s_err * mask
            fluid_loss = tf.reduce_sum(fluid_loss, axis=[1, 2, 3]) / (tf.reduce_sum(mask, axis=[1, 2, 3]) + self.epsilon)

            non_fluid_loss = s_err * non_fluid_mask
            non_fluid_loss = tf.reduce_sum(non_fluid_loss, axis=[1, 2, 3]) / (tf.reduce_sum(non_fluid_mask, axis=[1, 2, 3]) + self.epsilon)

            mse_total = self.weighting_fluid * fluid_loss + self.weighting_non_fluid * non_fluid_loss
        else:
            mse_total = tf.reduce_sum(s_err, axis=[1, 2, 3]) / (mask.shape[1] * mask.shape[2] * mask.shape[3])
            mse_total *= 2

        # === Add directional loss if l1_projected ===
        if self.use_directional_loss:
            directional_loss = self.calculate_l1_mutually_projected_loss(u, v, w, u_pred, v_pred, w_pred, alpha=0.5)
            directional_loss_fluid = tf.reduce_sum(directional_loss * mask, axis=[1, 2, 3]) / (tf.reduce_sum(mask, axis=[1, 2, 3]) + self.epsilon)
            data_loss = self.alpha * mse_total + (1 - self.alpha) * directional_loss_fluid
        else:
            data_loss = mse_total

        return tf.reduce_mean(data_loss), data_loss, 0

    def mse_loss(self, y_true, y_pred, mask):
        u,v,w = y_true[...,0],y_true[...,1], y_true[...,2]
        u_pred,v_pred,w_pred = y_pred[...,0],y_pred[...,1], y_pred[...,2]

        mse = self.calculate_l2_error(u,v,w, u_pred,v_pred,w_pred) 

        # === Separate mse ===
        non_fluid_mask = tf.less(mask, tf.constant(0.5))
        non_fluid_mask = tf.cast(non_fluid_mask, dtype=tf.float32)


        fluid_loss = mse * mask
        fluid_loss = tf.reduce_sum(fluid_loss, axis=[1,2,3]) / (tf.reduce_sum(mask, axis=[1,2,3]) + self.epsilon)

        non_fluid_loss = mse * non_fluid_mask
        non_fluid_loss = tf.reduce_sum(non_fluid_loss, axis=[1,2,3]) / (tf.reduce_sum(non_fluid_mask, axis=[1,2,3]) + self.epsilon)

        mse = fluid_loss + non_fluid_loss
        return mse

    def cosine_similarity_loss(self, y_true, y_pred, mask):
        u,v,w = y_true[...,0],y_true[...,1], y_true[...,2]
        u_pred,v_pred,w_pred = y_pred[...,0],y_pred[...,1], y_pred[...,2]

        cs = self.calculate_cosine_similarity(u,v,w, u_pred,v_pred,w_pred) 

        # === Separate mse ===
        non_fluid_mask = tf.less(mask, tf.constant(0.5))
        non_fluid_mask = tf.cast(non_fluid_mask, dtype=tf.float32)

        fluid_loss = cs * mask
        fluid_loss = tf.reduce_sum(fluid_loss, axis=[1,2,3]) / (tf.reduce_sum(mask, axis=[1,2,3]) + self.epsilon)

        return fluid_loss


    def calculate_regularizer_loss(self):
        """
            https://stackoverflow.com/questions/62440162/how-do-i-take-l1-and-l2-regularizers-into-account-in-tensorflow-custom-training
        """
        loss = 0
        for l in self.model.layers:
            # if hasattr(l,'layers') and l.layers: # the layer itself is a model
            #     loss+=add_model_loss(l)
            if hasattr(l,'kernel_regularizer') and l.kernel_regularizer:
                loss+=l.kernel_regularizer(l.kernel)
            if hasattr(l,'bias_regularizer') and l.bias_regularizer:
                loss+=l.bias_regularizer(l.bias)
        return loss

    def accuracy_function(self, y_true, y_pred, mask):
        """
            Calculate relative speed error
        """
        u,v,w = y_true[...,0],y_true[...,1], y_true[...,2]
        u_pred,v_pred,w_pred = y_pred[...,0],y_pred[...,1], y_pred[...,2]

        return self.calculate_relative_error(u_pred, v_pred, w_pred, u, v, w, mask)

    def calculate_l2_error(self, u, v, w, u_pred, v_pred, w_pred):
        """
            Calculate Speed magnitude error
        """
        return (u_pred - u) ** 2 +  (v_pred - v) ** 2 + (w_pred - w) ** 2

    def calculate_l1_error(self, u, v, w, u_pred, v_pred, w_pred):
        """
            Calculate L1 norm
        """
        return tf.abs(u_pred - u) +  tf.abs(v_pred - v) + tf.abs(w_pred - w)

    def calculate_l2norm(self, u, v, w):
        """
            Calculate L2 norm
        """
        return tf.sqrt(u ** 2 + v ** 2 + w ** 2)

    def calculate_relative_error(self, u_pred, v_pred, w_pred, u_hi, v_hi, w_hi, binary_mask, threshold = 0.5):
        # if epsilon is set to 0, we will get nan and inf
        epsilon = 1e-5

        u_diff = tf.square(u_pred - u_hi)
        v_diff = tf.square(v_pred - v_hi)
        w_diff = tf.square(w_pred - w_hi)

        diff_speed = tf.sqrt(u_diff + v_diff + w_diff)
        actual_speed = tf.sqrt(tf.square(u_hi) + tf.square(v_hi) + tf.square(w_hi)) 
        relative_speed_loss = diff_speed / (actual_speed + epsilon)
        
        # Make sure the range is between 0 and 1 usign tanh
        #relative_speed_loss = tf.clip_by_value(relative_speed_loss, 0., 1.)
        relative_speed_loss = tf.tanh(relative_speed_loss)

        condition = tf.not_equal(actual_speed, tf.constant(0.))
        corrected_speed_loss = tf.where(condition, relative_speed_loss, diff_speed)

        multiplier = 1e4 # round it so we don't get any infinitesimal number
        corrected_speed_loss = tf.round(corrected_speed_loss * multiplier) / multiplier
        
        # Apply mask
        binary_mask_condition = (binary_mask > threshold)
        # binary_mask_condition = tf.equal(binary_mask, 1.0)          
        corrected_speed_loss = tf.where(binary_mask_condition, corrected_speed_loss, tf.zeros_like(corrected_speed_loss))

        mean_err = tf.reduce_sum(corrected_speed_loss, axis=[1,2,3]) / (tf.reduce_sum(binary_mask, axis=[1,2,3]) + 1) * 100

        return mean_err
    
    def calculate_cosine_similarity(self, u, v, w, u_pred, v_pred, w_pred):
        """
        cosine similarity calculation. 1 if simlar direction, 0 if orthogonal, -1 if opposite direction
        """
        eps = 0.00005
        return (u*u_pred + v*v_pred + w*w_pred)/(self.calculate_l2norm(u, v, w)* self.calculate_l2norm(u_pred, v_pred, w_pred)+ eps)

    def calculate_huber_loss(self, u, v, w, u_pred, v_pred, w_pred, delta = 0.05):
        """
            Calculate huberloss depending on delta
        """
        huber_mse = 0.5*((u_pred - u) ** 2 +  (v_pred - v) ** 2 + (w_pred - w) ** 2)
        huber_mae = delta * (tf.abs(u_pred - u) + tf.abs(v_pred - v) + tf.abs(w_pred - w) - 0.5 * delta)

        return tf.where(tf.abs(u_pred - u) + tf.abs(v_pred - v) + tf.abs(w_pred - w) <= delta, huber_mse, huber_mae)
    
    def calculate_pseudo_huber_loss(self, u, v, w, u_pred, v_pred, w_pred, delta= 0.05):
        a = (u_pred - u) + (v_pred-v) + (w-w_pred)
        return delta**2*(tf.sqrt(1+(a**2/delta**2))-1)

    def calculate_l1_mutually_projected_loss(self,  u, v, w, u_pred, v_pred, w_pred, alpha= 0.5):
        """
            Calculate L1 mutually projected loss
        """
        eps = 0.00005
        proj_l1_u_v = tf.abs(self.calculate_l2norm(u,v,w) - (u*u_pred + v*v_pred + w*w_pred)/ (self.calculate_l2norm(u,v,w) + eps))
        proj_l1_v_u = tf.abs(self.calculate_l2norm(u_pred,v_pred,w_pred) - (u*u_pred + v*v_pred + w*w_pred)/ (self.calculate_l2norm(u_pred,v_pred,w_pred) + eps))
        return alpha * proj_l1_u_v + (1-alpha)* proj_l1_v_u


    def combined_l1_mutually_projected_loss(self, u, v, w, u_pred, v_pred, w_pred, weight, alpha= 0.5):
        """
            Calculate L1 mutually projected loss
        """
        l1_mutuall_proj = self.calculate_l1_mutually_projected_loss(u, v, w, u_pred, v_pred, w_pred, alpha)
        l1 = self.calculate_l1_error(u, v, w, u_pred, v_pred, w_pred)
        return weight * l1_mutuall_proj + (1-weight) * l1

    def calculate_cosine_similarity_loss(self, u, v, w, u_pred, v_pred, w_pred):
        eps = 0.00005
        return 1 - (u*u_pred + v*v_pred + w*w_pred)/(self.calculate_l2norm(u, v, w)* (self.calculate_l2norm(u_pred, v_pred, w_pred) )+ eps)

    #--------INIT AND LOGGING------------

    def init_model_dir(self, config_path=None):
        """
            Create model directory to save the weights with a [network_name]_[datetime] format
            Also prepare logfile and tensorboard summary within the directory.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        self.unique_model_name = f'{self.network_name}_{timestamp}'

        self.model_dir = f"Temporal4DFlowNet/models/{self.unique_model_name}"
        # Do not use .ckpt on the model_path
        self.model_path = f"{self.model_dir}/{self.network_name}"

        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)

        if config_path is not None and os.path.exists(config_path):
            shutil.copy2(config_path, os.path.join(self.model_dir, os.path.basename(config_path)))

        # summary - Tensorboard stuff
        self._prepare_logfile_and_summary()
        print("Model compiled, starting training...")


    
    def _prepare_logfile_and_summary(self):
        """
            Prepare csv logfile to keep track of the loss and Tensorboard summaries
        """
        self.train_writer = tf.summary.create_file_writer(self.model_dir + '/tensorboard/train')
        self.val_writer   = tf.summary.create_file_writer(self.model_dir + '/tensorboard/validate')

        self.logfile = self.model_dir + '/loss.csv'

        # Log network and training configuration
        log_to_file(self.logfile, f'Network: {self.network_name}\n')
        log_to_file(self.logfile, f'Initial learning rate: {self.learning_rate}\n')
        log_to_file(self.logfile, f'LR decay epochs: {self.lr_decay_epoch}\n')
        log_to_file(self.logfile, f'L2 regularization: {self.L2_regularization}\n')
        log_to_file(self.logfile, f'Accuracy metric: {self.accuracy_metric}\n')
        # Log loss configuration
        log_to_file(self.logfile, f'Loss type: {self.loss_type}\n')
        log_to_file(self.logfile, f'Use directional loss: {self.use_directional_loss}\n')
        log_to_file(self.logfile, f'Alpha: {self.alpha}\n')
        log_to_file(self.logfile, f'Epsilon: {self.epsilon}\n')
        log_to_file(self.logfile, f'Weighting fluid: {self.weighting_fluid}\n')
        log_to_file(self.logfile, f'Weighting non-fluid: {self.weighting_non_fluid}\n')
        log_to_file(self.logfile, f'Separate MSE: {self.separate_mse}\n')

        # Header
        stat_names = ','.join(self.loss_metrics.keys())
        log_to_file(self.logfile, f'epoch, {stat_names}, learning rate, elapsed (sec), best_model, benchmark_err, benchmark_rel_err, benchmark_mse, benchmark_divloss\n')

        print("Copying source code to model directory...")
        base_path = "Temporal4DFlowNet/src/"
        
        directory_to_backup = [base_path + ".", base_path + "Network"]
        for directory in directory_to_backup:
            files = os.listdir(directory)
            for fname in files:
                if fname.endswith(".py") or fname.endswith(".ipynb"):
                    dest_fpath = os.path.join(self.model_dir, "backup_source", directory, fname)
                    os.makedirs(os.path.dirname(dest_fpath), exist_ok=True)
                    shutil.copy2(f"{directory}/{fname}", dest_fpath)

    #--------TRAINING------------
      
    @tf.function
    def train_step(self, data_pairs):
        u,v,w, u_mag, v_mag, w_mag, u_hr,v_hr, w_hr, venc, mask = data_pairs
        hires = tf.concat([u_hr, v_hr, w_hr], axis=-1)
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            if self.including_mag_input:
                input_data = [u,v,w, u_mag, v_mag, w_mag]
            else:
                input_data = [u,v,w]
            predictions = self.model(input_data, training=True)

            loss = self.calculate_and_update_metrics(hires, predictions, mask, 'train')
            

        # Get the gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)

        gradients_not_none = [g for g in gradients if g is not None]
        grad_norm_tensor = tf.linalg.global_norm(gradients_not_none)
        if tf.math.is_nan(grad_norm_tensor):
            print("\nGradient norm is NaN")
        self.gradient_over_threshold = tf.cond(
            tf.math.greater(grad_norm_tensor, self.gradient_threshold),
            lambda: tf.constant(True),
            lambda: tf.constant(False)
        )

        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return grad_norm_tensor

    @tf.function
    def test_step(self, data_pairs):
        u,v,w, u_mag, v_mag, w_mag, u_hr,v_hr, w_hr, venc, mask = data_pairs
        hires = tf.concat((u_hr, v_hr, w_hr), axis=-1)
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        if self.including_mag_input:
            input_data = [u,v,w, u_mag, v_mag, w_mag]
        else:
            input_data = [u,v,w]
        predictions = self.model(input_data, training=False)
        
        self.calculate_and_update_metrics(hires, predictions, mask, 'val')
       
        return predictions

    def calculate_and_update_metrics(self, hires, predictions, mask, metric_set):
        loss, data_loss, divloss = self.loss_function(hires, predictions, mask)
        rel_error = self.accuracy_function(hires, predictions, mask)
        mse = self.mse_loss(hires, predictions, mask)
        cs = self.cosine_similarity_loss(hires, predictions, mask)
        
        if metric_set == 'train':
            if self.L2_regularization > 0:
                l2_reg_loss = self.calculate_regularizer_loss()
            else:
                l2_reg_loss = 0

            self.loss_metrics[f'l2_reg_loss'].update_state(l2_reg_loss)
            loss += l2_reg_loss

        # Update the loss and accuracy
        self.loss_metrics[f'{metric_set}_loss'].update_state(loss)
        self.loss_metrics[f'{metric_set}_mse'].update_state(mse)
        self.loss_metrics[f'{metric_set}_cos_sim'].update_state(cs)
        self.loss_metrics[f'{metric_set}_div'].update_state(divloss)
        self.loss_metrics[f'{metric_set}_accuracy'].update_state(rel_error)
        return loss

    def reset_metrics(self):
        for key in self.loss_metrics.keys():
            self.loss_metrics[key].reset_states()

    def learning_rate_decay(self, epoch):
        '''
        taken from derek long :https://github.com/dlon450/4DFlowNetv2/blob/master/src/Network/TrainerSetup.py#L244
        '''
        # For 14k rows of data and batch 20, this is ~10k iterations
        if epoch > 0 and epoch % self.lr_decay_epoch == 0:
            self.optimizer.lr = self.optimizer.lr / np.sqrt(2)
            message = f'Learning rate adjusted to {self.optimizer.lr.numpy():.6f} - {time.ctime()}\n'
            print(message)

    def train_network(self, trainset, valset, n_epoch, testset=None):
        """
            Main training function. Receives trainining and validation TF dataset.
        """
        # ----- Run the training -----
        print("==================== TRAINING =================")
        print(f'Learning rate {self.optimizer.lr.numpy():.7f}')
        print(f"Start training at {time.ctime()} - {self.unique_model_name}\n", flush=True)
        start_time = time.time()
        
        # Setup acc and data count
        previous_loss = np.inf
        total_batch_train = tf.data.experimental.cardinality(trainset).numpy()
        total_batch_val = tf.data.experimental.cardinality(valset).numpy()

        for epoch in range(n_epoch):
            # ------------------------------- Training -------------------------------
            
            # Reset the metrics at the start of the next epoch
            self.reset_metrics()
            self.gradient_norm = []
            start_loop = time.time()
            if self.lr_decay_epoch > 0: self.learning_rate_decay(epoch)

            # --- Training ---
            try:
                for i, (data_pairs) in enumerate(trainset):
                    # Train the network
                    try:
                        grad_norm = self.train_step(data_pairs)
                        self.gradient_norm.append(grad_norm.numpy())
                    except Exception as e_train:
                        print(f"\nError during training step: {e_train}", flush=True)
                        print("Skipping the rest of the training for this epoch.")
                        continue

                    message = f"Epoch {epoch+1} Train batch {i+1}/{total_batch_train} | loss: {self.loss_metrics['train_loss'].result():.5f} ({self.loss_metrics['train_accuracy'].result():.1f} %) - {time.time()-start_loop:.1f} secs "
                    print(f"\r{message}", end='', flush=True)

                    
            except Exception as e:
                print(f"\nError during training: {e}", flush=True)
                print("Skipping the rest of the training for this epoch.")
                continue

            # --- Validation ---
            for i, (data_pairs) in enumerate(valset):
                try:
                    self.test_step(data_pairs)
                except Exception as e_val:
                    print(f"\nError during validation step: {e_val}", flush=True)
                    print("Skipping the rest of the validation for this epoch.")
                    continue
                message = f"Epoch {epoch+1} Validation batch {i+1}/{total_batch_val} | loss: {self.loss_metrics['val_loss'].result():.5f} ({self.loss_metrics['val_accuracy'].result():.1f} %) - {time.time()-start_loop:.1f} secs"
                print(f"\r{message}", end='')

            # --- Epoch logging ---
            message = f"\rEpoch {epoch+1} Train loss: {self.loss_metrics['train_loss'].result():.5f} ({self.loss_metrics['train_accuracy'].result():.1f} %), Val loss: {self.loss_metrics['val_loss'].result():.5f} ({self.loss_metrics['val_accuracy'].result():.1f} %) - {time.time()-start_loop:.1f} secs"
            
            loss_values = []
            # Get the loss values from the loss_metrics dict
            for key, value in self.loss_metrics.items():
                
                loss_values.append(f'{value.result():.5f}')
            loss_str = ','.join(loss_values)
            log_line = f"{epoch+1},{loss_str},{self.optimizer.lr.numpy():.6f},{time.time()-start_loop:.1f}"
            

            self._update_summary_logging(epoch)

            # --- Save criteria ---
            # also save the latest model every epoch
            self.save_latest_model()

            # only save best model according to the accuracy metric
            if self.loss_metrics[self.accuracy_metric].result() < previous_loss:
                self.save_best_model()
                
                # Update best acc
                previous_loss = self.loss_metrics[self.accuracy_metric].result()
                
                # logging
                message  += ' **' # Mark as saved
                log_line += ',**'

                # Benchmarking
                if self.QUICKSAVE_ENABLED and testset is not None:
                    quick_loss, quick_accuracy, quick_mse, quick_div = self.quicksave(testset, epoch+1)
                    quick_loss, quick_accuracy, quick_mse, quick_div = np.mean(quick_loss), np.mean(quick_accuracy), np.mean(quick_mse), np.mean(quick_div)

                    message  += f' Benchmark loss: {quick_loss:.5f} ({quick_accuracy:.1f} %)'
                    log_line += f', {quick_loss:.7f}, {quick_accuracy:.2f}%, {quick_mse:.7f}, {quick_div:.7f}'
            # Logging
            print(message)
            log_to_file(self.logfile, log_line+"\n")
            # /END of epoch loop

        # End
        hrs, mins, secs = calculate_time_elapsed(start_time)
        message =  f"\nTraining {self.network_name} completed! - name: {self.unique_model_name}"
        message += f"\nTotal training time: {hrs} hrs {mins} mins {secs} secs."
        message += f"\nFinished at {time.ctime()}"
        message += f"\n==================== END TRAINING ================="
        log_to_file(self.logfile, message)
        print(message)
        
        # Finish!
    
    def _save_model(self, suffix, optimizer_filename):
        self.model.save(f'{self.model_path}-{suffix}.h5')
        symbolic_weights = getattr(self.optimizer, 'weights')
        if symbolic_weights:
            weight_values = tf.keras.backend.batch_get_value(symbolic_weights)
            with open(f'{self.model_dir}/{optimizer_filename}', 'wb') as f:
                pickle.dump(weight_values, f)
        
    def save_best_model(self):
        self._save_model('best', 'optimizer.pkl')

    def save_latest_model(self):
        self._save_model('latest', 'optimizer_latest.pkl')

    def save_datapairs(self, data_pairs, epoch, index_batch):
        """
            Save the current datapairs that caused high gradient norm
        """
        dir_datapairs = f"{self.model_dir}/{self.network_name}/datapairs"
        os.makedirs(dir_datapairs, exist_ok=True)
        u,v,w, u_mag, v_mag, w_mag, u_hr,v_hr, w_hr, venc, mask = data_pairs
        datapair_filename = f"datapairs_epoch{epoch}_{index_batch}.h5"
        h5util.save_predictions(dir_datapairs, datapair_filename, "lr_u", u, compression='gzip')
        h5util.save_predictions(dir_datapairs, datapair_filename, "lr_v", v, compression='gzip')
        h5util.save_predictions(dir_datapairs, datapair_filename, "lr_w", w, compression='gzip')

        if self.including_mag_input:
            h5util.save_predictions(dir_datapairs, datapair_filename, "lr_u_mag", u_mag, compression='gzip')
            h5util.save_predictions(dir_datapairs, datapair_filename, "lr_v_mag", v_mag, compression='gzip')
            h5util.save_predictions(dir_datapairs, datapair_filename, "lr_w_mag", w_mag, compression='gzip')

        h5util.save_predictions(dir_datapairs, datapair_filename, "hr_u", np.squeeze(u_hr, -1), compression='gzip')
        h5util.save_predictions(dir_datapairs, datapair_filename, "hr_v", np.squeeze(v_hr, -1), compression='gzip')
        h5util.save_predictions(dir_datapairs, datapair_filename, "hr_w", np.squeeze(w_hr, -1), compression='gzip')

        h5util.save_predictions(dir_datapairs, datapair_filename, "venc", venc, compression='gzip')
        h5util.save_predictions(dir_datapairs, datapair_filename, "mask", mask, compression='gzip')

    def restore_model(self, old_model_dir, old_model_file):
        """
            Restore model weights and optimizer weights for uncompiled model
            Based on: https://stackoverflow.com/questions/49503748/save-and-load-model-optimizer-state

            For an uncompiled model, we cannot just set the optmizer weights directly because they are zero.
            We need to at least do an apply_gradients once and then set the optimizer weights.
        """
        # Set the path for the weights and optimizer
        model_weights_path = f"{old_model_dir}/{old_model_file}"
        opt_path   = f"{old_model_dir}/optimizer.pkl"

        # Load the optimizer weights
        with open(opt_path, 'rb') as f:
            opt_weights = pickle.load(f)
        
        # Get the model's trainable weights
        grad_vars = self.model.trainable_weights
        # This need not be model.trainable_weights; it must be a correctly-ordered list of 
        # grad_vars corresponding to how you usually call the optimizer.
        zero_grads = [tf.zeros_like(w) for w in grad_vars]

        # Apply gradients which don't do nothing with Adam
        self.optimizer.apply_gradients(zip(zero_grads, grad_vars))

        # Set the weights of the optimizer
        self.optimizer.set_weights(opt_weights)

        # NOW set the trainable weights of the model
        self.model.load_weights(model_weights_path)

    def _update_summary_logging(self, epoch):
        """
            Tf.summary for epoch level loss
        """
        # Filter out the train and val metrics
        train_metrics = {k.replace('train_',''): v for k, v in self.loss_metrics.items() if k.startswith('train_')}
        val_metrics = {k.replace('val_',''): v for k, v in self.loss_metrics.items() if k.startswith('val_')}
        
        # Summary writer
        with self.train_writer.as_default():
            tf.summary.scalar(f"{self.network_name}/learning_rate", self.optimizer.lr, step=epoch)
            for key in train_metrics.keys():
                tf.summary.scalar(f"{self.network_name}/{key}",  train_metrics[key].result(), step=epoch)  
            # also save gradient norm   
            tf.summary.scalar(f"{self.network_name}/gradient_norm",  np.mean(self.gradient_norm) if self.gradient_norm else 0, step=epoch)
        
        with self.val_writer.as_default():
            for key in val_metrics.keys():
                tf.summary.scalar(f"{self.network_name}/{key}",  val_metrics[key].result(), step=epoch)
   
        
    def quicksave(self, testset, epoch_nr):
        """
            Predict a batch of data from the benchmark testset.
            This is saved under the model directory with the name quicksave_[network_name].h5
            Quicksave is done everytime the best model is saved.
        """
        for i, (data_pairs) in enumerate(testset):
            u,v,w, u_mag, v_mag, w_mag, u_hr,v_hr, w_hr, venc, mask = data_pairs
            hires = tf.concat((u_hr, v_hr, w_hr), axis=-1)
            if self.including_mag_input:
                input_data = [u,v,w, u_mag, v_mag, w_mag]
            else:
                input_data = [u,v,w]

            preds = self.model.predict(input_data)

            loss_val, mse, divloss = self.loss_function(hires, preds, mask)
            rel_loss = self.accuracy_function(hires, preds, mask)
            # Do only 1 batch
            break

        quicksave_filename = f"quicksave_{self.network_name}.h5"
        h5util.save_predictions(self.model_dir, quicksave_filename, "epoch", np.asarray([epoch_nr]), compression='gzip')

        preds = np.expand_dims(preds, 0) # Expand dim to [epoch_nr, batch, ....]
        h5util.save_predictions(self.model_dir, quicksave_filename, "sr_u", preds[...,0], compression='gzip')
        h5util.save_predictions(self.model_dir, quicksave_filename, "sr_v", preds[...,1], compression='gzip')
        h5util.save_predictions(self.model_dir, quicksave_filename, "sr_w", preds[...,2], compression='gzip')

        if epoch_nr == 1:
            # Save the actual data only for the first epoch
            h5util.save_predictions(self.model_dir, quicksave_filename, "lr_u", u, compression='gzip')
            h5util.save_predictions(self.model_dir, quicksave_filename, "lr_v", v, compression='gzip')
            h5util.save_predictions(self.model_dir, quicksave_filename, "lr_w", w, compression='gzip')

            h5util.save_predictions(self.model_dir, quicksave_filename, "hr_u", np.squeeze(u_hr, -1), compression='gzip')
            h5util.save_predictions(self.model_dir, quicksave_filename, "hr_v", np.squeeze(v_hr, -1), compression='gzip')
            h5util.save_predictions(self.model_dir, quicksave_filename, "hr_w", np.squeeze(w_hr, -1), compression='gzip')
            
            h5util.save_predictions(self.model_dir, quicksave_filename, "venc", venc, compression='gzip')
            h5util.save_predictions(self.model_dir, quicksave_filename, "mask", mask, compression='gzip')
        
        return loss_val, rel_loss, mse, divloss