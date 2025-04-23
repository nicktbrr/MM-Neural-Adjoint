"""
The class wrapper for the networks
"""
# Built-in
import os
import time
import sys
import mlflow
from .base_model import BaseModel

# Torch
import torch
from torch import nn
from torch.optim import lr_scheduler

# Libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

class Network(object):
    def __init__(self,
                 geometry,
                 spectrum,
                 device,
                 num_linear_layers = 4,
                 num_conv_layers = 3,
                 num_linear_neurons = 1000,
                 num_conv_out_channel = 4,
                 ):
        self.model = BaseModel(geometry, spectrum, num_linear_layers, num_conv_layers, num_linear_neurons, num_conv_out_channel)
        self.device = device
        self.loss = self.make_loss()
        self.optm = None
        self.optm_eval = None
        self.best_validation_loss = float('inf')  
        self.geometry_mean = None
        self.geometry_lower_bound = None
        self.geometry_upper_bound = None
        mlflow.set_tracking_uri('sqlite:///mlflow.db')              # Set the BVL to large number


    def make_loss(self,logit=None, labels=None, G=None):
        """
        Create a tensor that represents the loss. This is consistant both at training time \
        and inference time for Backward model
        :param logit: The output of the network
        :param labels: The ground truth labels
        :return: the total loss
        """
        if logit is None:
            return None
        MSE_loss = nn.functional.mse_loss(logit, labels)          # The MSE Loss
        BDY_loss = 0
        if G is not None:         # This is using the boundary loss
            X_range, X_lower_bound, X_upper_bound = self.geometry_mean, self.geometry_lower_bound, self.geometry_upper_bound
            X_mean = (X_lower_bound + X_upper_bound) / 2        # Get the mean
            relu = torch.nn.ReLU()
            BDY_loss_all = 1 * relu(torch.abs(G - self.build_tensor(X_mean)) - 0.5 * self.build_tensor(X_range))
            BDY_loss = 10*torch.mean(BDY_loss_all)
        self.MSE_loss = MSE_loss
        self.Boundary_loss = BDY_loss
        return torch.add(MSE_loss, BDY_loss)


    def build_tensor(self, nparray, requires_grad=False):
        if isinstance(nparray, torch.Tensor):
            return nparray.clone().detach().to(device=self.device, dtype=torch.float).requires_grad_(requires_grad)
        return torch.tensor(nparray, requires_grad=requires_grad, device=self.device, dtype=torch.float)
    
    def make_optimizer_eval(self, geometry_eval, optimizer_type=None, lr=0.0005):
        """
        The function to make the optimizer during evaluation time.
        The difference between optm is that it does not have regularization and it only optmize the self.geometr_eval tensor
        :return: the optimizer_eval
        """
        if optimizer_type == 'Adam' or optimizer_type == None:
            op = torch.optim.Adam([geometry_eval], lr=lr)
        elif optimizer_type == 'RMSprop':
            op = torch.optim.RMSprop([geometry_eval], lr=lr)
        elif optimizer_type == 'SGD':
            op = torch.optim.SGD([geometry_eval], lr=lr)
        else:
            raise Exception("Your Optimizer is neither Adam, RMSprop or SGD, please change in param or contact Ben")
        return op


    def make_optimizer(self, optimizer_type=None, lr=0.0005, reg_scale=2e-05):
        """
        Make the corresponding optimizer from the flags. Only below optimizers are allowed. Welcome to add more
        :return:
        """
        if optimizer_type == 'Adam' or optimizer_type == None:
            op = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=reg_scale)
        elif optimizer_type == 'RMSprop':
            op = torch.optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=reg_scale)
        elif optimizer_type == 'SGD':
            op = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=reg_scale)
        else:
            raise Exception("Your Optimizer is neither Adam, RMSprop or SGD, please change in param or contact Ben")
        return op

    def make_lr_scheduler(self, optm):
        """
        Make the learning rate scheduler as instructed. More modes can be added to this, current supported ones:
        1. ReduceLROnPlateau (decrease lr when validation error stops improving
        :return:
        """
        return lr_scheduler.ReduceLROnPlateau(optimizer=optm, mode='min',
                                              factor=0.5,
                                              patience=10, threshold=1e-4)

    def load(self):
        """
        Loading the model from the check point folder with name best_model_forward.pt
        :return:
        """
        if torch.cuda.is_available():
            self.model = torch.load(os.path.join(self.ckpt_dir, 'best_model_forward.pt'), weights_only=False)
        else:
            self.model = torch.load(os.path.join(self.ckpt_dir, 'best_model_forward.pt'), map_location=torch.device('cpu'), weights_only=False)

    def train(self,
              epochs,
              train_loader,
              val_loader,
              save=False,
              progress_bar=None):
        """
        The major training function. This would start the training using information given in the flags
        :return: None
        """
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.to(self.device)

        self.optm = self.make_optimizer()
        my_lr_scheduler = self.make_lr_scheduler(self.optm)

        # Time keeping
        start_time = time.time()
        with mlflow.start_run(run_name=time.strftime('%Y%m%d_%H%M%S', time.localtime())):
            mlflow.log_params({
                    "learning_rate": 0.0005,
                    "batch_size": train_loader.batch_size,
                    "reg_scale": 2e-05,
                    "train_step": epochs,
                    "optimizer": "Adam",
            })

            for epoch in range(epochs):
                self.model.train()
                train_loss = 0
                # Add progress bar for batches
                batch_iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False) if progress_bar is None else train_loader
                for j, (geometry, spectra) in enumerate(batch_iterator):
                    if cuda:
                        geometry = geometry.to(self.device)
                        spectra = spectra.to(self.device)
                    self.optm.zero_grad()
                    logit = self.model(geometry)
                    loss = self.make_loss(logit, spectra)
                    loss.backward()
                    self.optm.step()
                    train_loss += loss

                # Calculate the avg loss of training
                train_avg_loss = train_loss.cpu().data.numpy() / (j + 1)
                mlflow.log_metric("train_loss", train_avg_loss, step=epoch)

                if epoch % 2 == 0:
                    # Set to Evaluation Mode
                    self.model.eval()
                    test_loss = 0
                    for j, (geometry, spectra) in enumerate(val_loader):
                        geometry = geometry.to(self.device)
                        spectra = spectra.to(self.device)
                        logit = self.model(geometry)
                        loss = self.make_loss(logit, spectra)                   # compute the loss
                        test_loss += loss 
                        

                    test_avg_loss = test_loss.cpu().data.numpy() / (j+1)
                    mlflow.log_metric("validation_loss", test_avg_loss, step=epoch)

                    if test_avg_loss < self.best_validation_loss:
                        self.best_validation_loss = test_avg_loss
                        if save:
                            sample_input = next(iter(val_loader))[0][:1]
                            sample_input = sample_input.cpu().data.numpy()

                            mlflow.pytorch.log_model(self.model, "best_model", input_example=sample_input)
                        mlflow.log_metric("best_validation_loss", self.best_validation_loss)

                        if self.best_validation_loss < 1E-05:
                            print("Training finished EARLIER at epoch %d, reaching loss of %.5f" %\
                                (epoch, self.best_validation_loss))
                            break

                # Learning rate decay upon plateau
                my_lr_scheduler.step(train_avg_loss)
                
                # Update the main progress bar if provided
                if progress_bar is not None:
                    progress_bar.update(1)
                    progress_bar.set_postfix({'train_loss': f'{train_avg_loss:.6f}', 
                                           'val_loss': f'{test_avg_loss:.6f}'})
                    
            mlflow.log_metric("total_training_time", time.time() - start_time)

    def evaluate_geometry(self, val_loader, save_dir='results/', back_prop_steps=300, save_num=1):
        """
        The function to evaluate how good the Neural Adjoint is and output results
        :param save_dir: The directory to save the results
        :param device: The device to run the evaluation on
        :param back_prop_steps: The number of back propagation steps to take
        :param save_num: The number of results to save
        :param save_all: Save all the results instead of the best one (T_200 is the top 200 ones)
        :param MSE_Simulator: Use simulator loss to sort (DO NOT ENABLE THIS, THIS IS OK ONLY IF YOUR APPLICATION IS FAST VERIFYING)
        :param save_misc: save all the details that are probably useless
        :param save_Simulator_Ypred: Save the Ypred that the Simulator gives
        (This is useful as it gives us the true Ypred instead of the Ypred that the network "thinks" it gets, which is
        usually inaccurate due to forward model error)
        :return:
        """
        if self.model is None:
            raise ValueError("Model is not trained yet. Please train the model first or call load() to load a pre-trained model.")
        if next(iter(val_loader))[0].shape[0] != 1:
            raise ValueError("The batch size of the test loader must be 1. Please change the batch size of the test loader.")
        
        self.model.to(self.device)
        self.model.eval()
        # Get the file names
        Ypred_file = os.path.join(save_dir, 'test_Ypred.csv')
        Xtruth_file = os.path.join(save_dir, 'test_Xtruth.csv')
        Ytruth_file = os.path.join(save_dir, 'test_Ytruth.csv')
        Xpred_file = os.path.join(save_dir, 'test_Xpred.csv')

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Open those files to append
        with open(Xtruth_file, 'a') as fxt,open(Ytruth_file, 'a') as fyt,\
              open(Xpred_file, 'a') as fxp:
            # Loop through the eval data and evaluate
            for _, (geometry, spectra) in enumerate(val_loader):
                geometry = geometry.to(self.device)
                spectra = spectra.to(self.device)
                # Initialize the geometry first
                Xpred, _, _ = self.evaluate_one(spectra, val_loader, back_prop_steps)

                np.savetxt(fxt, geometry.cpu().data.numpy())
                np.savetxt(fyt, spectra.cpu().data.numpy())
                np.savetxt(fxp, Xpred)
        return Ypred_file, Ytruth_file

    def evaluate_one(self, target_spectra, val_loader, back_prop_steps=300, num_geometry_eval=2048):
        """
        The function which being called during evaluation and evaluates one target y using # different trails
        :param target_spectra: The target spectra/y to backprop to 
        :param save_dir: The directory to save to when save_all flag is true
        :param MSE_Simulator: Use Simulator Loss to get the best instead of the default NN output logit
        :param save_all: The multi_evaluation where each trail is monitored (instad of the best) during backpropagation
        :param ind: The index of this target_spectra in the batch
        :param save_misc: The flag to print misc information for degbugging purposes, usually printed to best_mse
        :return: Xpred_best: The 1 single best Xpred corresponds to the best Ypred that is being backproped 
        :return: Ypred_best: The 1 singe best Ypred that is reached by backprop
        :return: MSE_list: The list of MSE at the last stage
        """

        # Initialize the geometry_eval or the initial guess xs
        geometry_eval = torch.rand([num_geometry_eval, self.model.linears[0].in_features], requires_grad=True)
        
        self.optm_eval = self.make_optimizer_eval(geometry_eval)
        self.lr_scheduler = self.make_lr_scheduler(self.optm_eval)
        
        target_spectra_expand = target_spectra.expand([num_geometry_eval, -1])

        # Begin NA
        pbar = tqdm(range(back_prop_steps), desc='Backpropagation Progress', leave=False)
        for i in pbar:
            # Make the initialization from [-1, 1], can only be in loop due to gradient calculator constraint
            geometry_eval_input = self.initialize_from_uniform_to_dataset_distrib(geometry_eval, val_loader)

            self.optm_eval.zero_grad()                                  # Zero the gradient first
            logit = self.model(geometry_eval_input)
            
            ###################################################
            # Boundar loss controled here: with Boundary Loss #
            ###################################################
            loss = self.make_loss(logit, target_spectra_expand, G=geometry_eval_input)         # Get the loss
            loss.backward()                                             # Calculate the Gradient
            # update weights and learning rate scheduler
            if i != back_prop_steps - 1:
                self.optm_eval.step()  # Move one step the optimizer
                self.lr_scheduler.step(loss.data)
            
            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        ###################################
        # From candidates choose the best #
        ###################################
        Ypred = logit.cpu().data.numpy()

        if len(np.shape(Ypred)) == 1:           # If this is the ballistics dataset where it only has 1d y'
            Ypred = np.reshape(Ypred, [-1, 1])
        
        # calculate the MSE list and get the best one
        MSE_list = np.mean(np.square(Ypred - target_spectra_expand.cpu().data.numpy()), axis=1)
        best_estimate_index = np.argmin(MSE_list)
        Xpred_best = np.reshape(np.copy(geometry_eval_input.cpu().data.numpy()[best_estimate_index, :]), [1, -1])
        Ypred_best = np.reshape(np.copy(Ypred[best_estimate_index, :]), [1, -1])

        return Xpred_best, Ypred_best, MSE_list

    def initialize_from_uniform_to_dataset_distrib(self, geometry_eval, val_loader):
        """
        since the initialization of the backprop is uniform from [0,1], this function transforms that distribution
        to suitable prior distribution for each dataset. The numbers are accquired from statistics of min and max
        of the X prior given in the training set and data generation process
        :param geometry_eval: The input uniform distribution from [0,1]
        :return: The transformed initial guess from prior distribution
        """
        X_range, X_lower_bound, _ = self.get_boundary_lower_bound_uper_bound(val_loader)
        geometry_eval_input = geometry_eval * self.build_tensor(X_range) + self.build_tensor(X_lower_bound)
        return geometry_eval_input
        #return geometry_eval

    
    def get_boundary_lower_bound_uper_bound(self, val_loader):
        """
        Due to the fact that the batched dataset is a random subset of the training set, mean and range would fluctuate.
        Therefore we pre-calculate the mean, lower boundary and upper boundary to avoid that fluctuation. Replace the
        mean and bound of your dataset here
        :return:
        """
        if self.geometry_mean is None:
            for x, _ in val_loader:
                num_features = x.shape[1]
                break
            
            # Initialize accumulators
            sum_x = torch.zeros(num_features)
            count = 0
            
            # Initialize mins with infinity and maxs with negative infinity
            mins = torch.full((num_features,), float('inf'))
            maxs = torch.full((num_features,), float('-inf'))
            
            # Iterate through the dataloader
            for x, _ in val_loader:
                batch_size = x.shape[0]
                count += batch_size
                sum_x += torch.sum(x, dim=0)
                
                # Update minimum values
                batch_mins = torch.min(x, dim=0).values
                mins = torch.minimum(mins, batch_mins)
                
                # Update maximum values
                batch_maxs = torch.max(x, dim=0).values
                maxs = torch.maximum(maxs, batch_maxs)
            
            # Calculate mean
            means = sum_x / count
            self.geometry_mean = means
            self.geometry_lower_bound = mins
            self.geometry_upper_bound = maxs

        return self.geometry_mean, self.geometry_lower_bound, self.geometry_upper_bound


    def predict(self, Xpred_file, no_save=False, load_state_dict=None):
        """
        The prediction function, takes Xpred file and write Ypred file using trained model
        :param Xpred_file: Xpred file by (usually VAE) for meta-material
        :param no_save: do not save the txt file but return the np array
        :param load_state_dict: If None, load model using self.load() (default way), If a dir, load state_dict from that dir
        :return: pred_file, truth_file to compare
        """
        if load_state_dict is None:
            self.load()         # load the model in the usual way
        else:
            self.model.load_state_dict(torch.load(load_state_dict))
       
        Ypred_file = Xpred_file.replace('Xpred', 'Ypred')
        Ytruth_file = Ypred_file.replace('Ypred', 'Ytruth')
        Xpred = pd.read_csv(Xpred_file, header=None, delimiter=',')     # Read the input
        if len(Xpred.columns) == 1: # The file is not delimitered by ',' but ' '
            Xpred = pd.read_csv(Xpred_file, header=None, delimiter=' ')
        Xpred.info()
        Xpred_tensor = torch.from_numpy(Xpred.values).to(torch.float)
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()
            Xpred_tensor = Xpred_tensor.cuda()
        # Put into evaluation mode
        self.model.eval()
        Ypred = self.model(Xpred_tensor)
        if load_state_dict is not None:
            Ypred_file = Ypred_file.replace('Ypred', 'Ypred' + load_state_dict[-7:-4])
        elif self.flags.model_name is not None:
                Ypred_file = Ypred_file.replace('Ypred', 'Ypred' + self.flags.model_name)
        if no_save:                             # If instructed dont save the file and return the array
             return Ypred.cpu().data.numpy(), Ytruth_file
        np.savetxt(Ypred_file, Ypred.cpu().data.numpy())

        return Ypred_file, Ytruth_file

    def plot_histogram(self, loss, ind):
        """
        Plot the loss histogram to see the loss distribution
        """
        f = plt.figure()
        plt.hist(loss, bins=100)
        plt.xlabel('MSE loss')
        plt.ylabel('cnt')
        plt.suptitle('(Avg MSE={:4e})'.format(np.mean(loss)))
        plt.savefig(os.path.join('data','loss{}.png'.format(ind)))
        return None
