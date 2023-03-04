#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 10:37:01 2023

@author: daniel
"""
import os
import tensorflow as tf
os.environ['PYTHONHASHSEED'], os.environ["TF_DETERMINISTIC_OPS"] = '0', '1'
os.environ['CUDA_VISIBLE_DEVICES'], os.environ['TF_CPP_MIN_LOG_LEVEL'] = '-1', '3'

import joblib
import numpy as np
import pkg_resources
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 

import random as python_random
##https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development##
np.random.seed(1909), python_random.seed(1909), tf.random.set_seed(1909)

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.backend import clear_session 
from tensorflow.keras.models import Sequential, save_model, load_model, Model
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.optimizers import SGD, Adagrad, Adam, RMSprop
from tensorflow.keras.losses import categorical_crossentropy, Hinge, SquaredHinge, KLDivergence, LogCosh
from tensorflow.keras.layers import Input, Activation, Dense, Dropout, Conv2D, MaxPool2D, \
    AveragePooling2D, GlobalAveragePooling2D, Flatten, BatchNormalization, Lambda, concatenate

from optuna.importance import get_param_importances, FanovaImportanceEvaluator
from MicroLIA.data_processing import process_class, create_training_set, concat_channels
from MicroLIA.data_augmentation import augmentation, resize
from MicroLIA import optimization


class Classifier:
    """
    Creates and trains the convolutional neural network. The built-in methods can be used predict new samples, and also optimize the engine and output visualizations.


    Args:
        positive_class (ndarray): The samples for the first class should be passed, which will automatically 
            be assigned the positive label '1'.
        negative_class (ndarray, str): The samples for the second class should be passed, which will automatically 
            be assigned the negative label '0'.
        val_positive (ndarray, optional): Positive class data to be used for validation. Defaults to None.
        val_negative (ndarray, optional): Negative class data to be used for validation. Defaults to None.
        test_positive (ndarray, optional): Positive class data to be used for post-trial testing. Defaults to None.
        test_negative (ndarray, optional): Negative class data to be used for post-trial testing. Defaults to None.
        optimize (bool): If True the Boruta algorithm will be run to identify the features
            that contain useful information, after which the optimal Random Forest hyperparameters will be calculated 
            using Bayesian optimization. 
        n_iter (int): The maximum number of iterations to perform during the hyperparameter search. Defaults to 25. 
        train_epochs (int): Number of epochs to the train the CNN to during the optimization trials. Defaults to 25.
        img_num_channels (int): The number of filters. Defaults to 1.
        normalize (bool, optional): If True the data will be min-max normalized using the 
            input min and max pixels. Defaults to True.
        min_pixel (int, optional): The minimum pixel count, pixels with counts 
            below this threshold will be set to this limit. Defaults to 0.
        max_pixel (int, list, optional): The maximum pixel count, pixels with counts 
            above this threshold will be set to this limit. Defaults to 100. If img_num_channels
            is not 1, the max_pixel should be a list containing two values, one for each band.
        metric (str): Assesment metric to use when both pruning and scoring the hyperparameter optimization trial.
            Defaults to 'loss'. Options include: 'loss' 'binary_accuracy', 'f1_score' 'all' or the validation equivalents (e.g. 'val_loss').
        patience (int): Number of epochs without improvement before the optimization trial is terminated. Defaults to 0, which
            disables this feature.
        average (bool): If False, the designated metric will be calculated according to its value at the end of the train_epochs. 
            If True, the metric will be averaged out across all train_epochs. Defaults to True.
        opt_model (bool): If True, the architecture parameters will be optimized. Defaults to True.
        opt_aug (bool): If True, the augmentation procedure will be optimized. Defaults to False.
        batch_min (int): The minimum number of augmentations to perform per image on the positive class, only applicable 
        if opt_aug=True. Defaults to 2.
        batch_max (int): The maximum number of augmentations to perform per image on the positive class, only applicable 
        if opt_aug=True. Defaults to 25.
        image_size_min (int): The minimum image size to assess, only applicable if opt_aug=True. Defaults to 50.
        image_size_max (int): The maximum image size to assess, only applicable if opt_aug=True. Defaults to 100.
        opt_max_min_pix (int, optional): The minimum max pixel value to use when tuning the normalization procedure, 
            only applicable if opt_aug=True. Defaults to None.
        opt_max_max_pix (int, optional): The maximum max pixel value to use when tuning the normalization procedure, 
            only applicable if opt_aug=True. Defaults to None.
        shift (int): The max allowed vertical/horizontal shifts to use during the data augmentation routine, only applicable
            if opt_aug=True. Defaults to 10 pixels.
        mask_size (int, optional): If enabled, this will set the pixel length of a square cutout, to be randomly placed
            somewhere in the augmented image. This cutout will replace the image values with 0, therefore serving as a 
            regularizear. Only applicable if opt_aug=True. Defaults to None.
        num_masks (int, optional): The number of masks to create, to be used alongside the mask_size parameter. If 
            this is set to a value greater than one, overlap may occur. 
        verbose (int): Controls the amount of output printed during the training process. A value of 0 is for silent mode, 
            a value of 1 is used for progress bar mode, and 2 for one line per epoch mode. Defaults to 1.
        opt_cv (int): Cross-validations to perform when assesing the performance at each
            hyperparameter optimization trial. For example, if cv=3, then each optimization trial
            will be assessed according to the 3-fold cross validation accuracy. Defaults to 10.
            NOTE: The higher this number, the longer the optimization will take.
        balance (bool, optional): This will determine whether the two classes
            are kept the same size during optimization, applicable if tuning the augmentation
            parameters. Defaults to True.
        limit_search (bool): If False, the AlexNet layer paremters will be tuned, including the number of filters to learn
            as well as the convolution sizes and strides. Defaults to False due to memory allocation issues when handling
            lots of tunable parameters.
        train_acc_threshold (float, optional): A value between 0 and 1 that designates the training binary_accuracy threshold that should
            not be exceeded. As soon as the training exceeds this limit, the trial is stopped early. Defaults to None.
        monitor1 (str, optional): The first metric to monitor, can take the same values as the metric argument. Defaults to None.
        monitor2 (str, optional): The second metric to monitor, can take the same values as the metric argument. Defaults to None.
        monitor1_thresh (float, optional): The threshold value of the first monitor metric. If the metric is loss-related
            the training will stop early if the value falls below this threshold. Similarly, if the metric is accuracy-related,
            then the training will stop early if the value falls above this threshold. Defaults to None.
        monitor2_thresh (float, optional): The threshold value of the second monitor metric. If the metric is loss-related
            the training will stop early if the value falls below this threshold. Similarly, if the metric is accuracy-related,
            then the training will stop early if the value falls above this threshold. Defaults to None.

        Attributes:
            model (object): The machine learning model.
            history (object): The machine learning model history, generated during training.
            optimization_results (object): The Optuna hyperparameter optimization history result.
            best_params (dict): The best parameters output from the optimization routine.

    """

    def __init__(self, positive_class=None, negative_class=None, val_positive=None, val_negative=None, img_num_channels=1, 
        optimize=True, n_iter=25, normalize=True, min_pixel=0, max_pixel=1000, epochs=25, train_epochs=25, 
        patience=5, opt_model=True, opt_aug=False, batch_min=2, batch_max=25, image_size_min=50, image_size_max=100, 
        balance=True, opt_max_min_pix=None, opt_max_max_pix=None, metric='loss', average=True, test_positive=None, 
        test_negative=None, shift=10, opt_cv=None, mask_size=None, num_masks=None, verbose=0, train_acc_threshold=None,
        limit_search=True, monitor1=None, monitor2=None, monitor1_thresh=None, monitor2_thresh=None):

        self.positive_class = positive_class
        self.negative_class = negative_class
        self.optimize = optimize 
        self.metric = metric 
        self.average = average
        self.n_iter = n_iter
        self.img_num_channels = img_num_channels
        self.normalize = normalize 
        self.min_pixel = min_pixel
        self.max_pixel = max_pixel
        self.val_positive = val_positive
        self.val_negative = val_negative
        self.epochs = epochs
        self.train_epochs = train_epochs
        self.patience = patience

        self.opt_model = opt_model 
        self.opt_aug = opt_aug
        self.batch_min = batch_min 
        self.batch_max = batch_max 
        self.image_size_min = image_size_min
        self.image_size_max = image_size_max
        self.balance = balance
        self.opt_max_min_pix = opt_max_min_pix
        self.opt_max_max_pix = opt_max_max_pix
        self.test_positive = test_positive
        self.test_negative = test_negative
        self.shift = shift  
        self.opt_cv = opt_cv 
        self.mask_size = mask_size
        self.num_masks = num_masks
        self.verbose = verbose
        self.train_acc_threshold = train_acc_threshold
        self.limit_search = limit_search

        self.monitor1 = monitor1
        self.monitor2 = monitor2
        self.monitor1_thresh = monitor1_thresh
        self.monitor2_thresh = monitor2_thresh

        self.model = None
        self.history = None 
        self.best_params = None 
        self.optimization_results = None 

    def create(self):
        """
        Creates the CNN machine learning engine.
        
        Returns:
            Trained classifier.
        """

        if self.optimize is False and self.best_params is None:
            print("Returning base AlexNet model...")
            self.model, self.history = AlexNet(self.positive_class, self.negative_class, img_num_channels=self.img_num_channels, normalize=self.normalize,
                min_pixel=self.min_pixel, max_pixel=self.max_pixel, val_blob=self.val_blob, val_other=self.val_other, epochs=self.epochs)
            
            return      

        if self.best_params is None:
            self.best_params, self.optimization_results = optimization.hyper_opt(self.positive_class, self.negative_class, clf='cnn', metric=self.metric, average=self.average, 
                n_iter=self.n_iter, balance=False, return_study=True, img_num_channels=self.img_num_channels, normalize=self.normalize, min_pixel=self.min_pixel, max_pixel=self.max_pixel, 
                val_X=self.val_blob, val_Y=self.val_other, train_epochs=self.train_epochs, patience=self.patience, opt_model=self.opt_model, opt_aug=self.opt_aug, 
                batch_min=self.batch_min, batch_max=self.batch_max, image_size_min=self.image_size_min, image_size_max=self.image_size_max, balance=self.balance,
                opt_max_min_pix=self.opt_max_min_pix, opt_max_max_pix=self.opt_max_max_pix, test_blob=self.test_blob, test_other=self.test_other, shift=self.shift, opt_cv=self.opt_cv, 
                mask_size=self.mask_size, num_masks=self.num_masks, verbose=self.verbose, train_acc_threshold=self.train_acc_threshold, limit_search=self.limit_search,
                monitor1=self.monitor1, monitor2=self.monitor2, monitor1_thresh=self.monitor1_thresh, monitor2_thresh=self.monitor2_thresh)
            print("Fitting and returning final model...")
        else:
            print('Fitting model using the loaded best_params...')
            
        if self.epochs != 0:

            clear_session()

            if self.opt_max_min_pix is not None:
                self.normalize = True #In case it's mistakenly set to False by user
                min_pix, max_pix = 0.0, []
                if self.img_num_channels >= 1:
                    max_pix.append(self.best_params['max_pixel_1'])
                if self.img_num_channels >= 2:
                    max_pix.append(self.best_params['max_pixel_2'])
                if self.img_num_channels == 3:
                    max_pix.append(self.best_params['max_pixel_3'])
            else:
                min_pix, max_pix = self.min_pixel, self.max_pixel

            if self.opt_aug:
                if self.img_num_channels == 1:
                    channel1, channel2, channel3 = copy.deepcopy(self.positive_class), None, None 
                elif self.img_num_channels == 2:
                    channel1, channel2, channel3 = copy.deepcopy(self.positive_class[:,:,:,0]), copy.deepcopy(self.positive_class[:,:,:,1]), None 
                elif self.img_num_channels == 3:
                    channel1, channel2, channel3 = copy.deepcopy(self.positive_class[:,:,:,0]), copy.deepcopy(self.positive_class[:,:,:,1]), copy.deepcopy(self.positive_class[:,:,:,2])
                else:
                    raise ValueError('Only three filters are supported!')

                augmented_images = augmentation(channel1=channel1, channel2=channel2, channel3=channel3, batch=self.best_params['num_aug'], 
                    width_shift=self.shift, height_shift=self.shift, horizontal=True, vertical=True, rotation=False,#self.best_params['rotation'], 
                    image_size=self.best_params['image_size'], mask_size=self.mask_size, num_masks=self.num_masks)

                if self.img_num_channels > 1:
                    class_1=[]
                    if self.img_num_channels == 2:
                        for i in range(len(augmented_images[0])):
                            class_1.append(concat_channels(augmented_images[0][i], augmented_images[1][i]))
                    else:
                        for i in range(len(augmented_images[0])):
                            class_1.append(concat_channels(augmented_images[0][i], augmented_images[1][i], augmented_images[2][i]))
                    class_1 = np.array(class_1)
                else:
                    class_1 = augmented_images

                #Perform same augmentation techniques on other data, batch=1
                if self.img_num_channels == 1:
                    channel1, channel2, channel3 = copy.deepcopy(self.negative_class), None, None 
                elif self.img_num_channels == 2:
                    channel1, channel2, channel3 = copy.deepcopy(self.negative_class[:,:,:,0]), copy.deepcopy(self.negative_class[:,:,:,1]), None 
                elif self.img_num_channels == 3:
                    channel1, channel2, channel3 = copy.deepcopy(self.negative_class[:,:,:,0]), copy.deepcopy(self.negative_class[:,:,:,1]), copy.deepcopy(self.negative_class[:,:,:,2])
                
                augmented_images_other = augmentation(channel1=channel1, channel2=channel2, channel3=channel3, batch=1, 
                    width_shift=self.shift, height_shift=self.shift, horizontal=horizontal, vertical=vertical, rotation=False, 
                    image_size=image_size, mask_size=self.mask_size, num_masks=self.num_masks)

                #The augmentation routine returns an output for each filter, e.g. 3 outputs for RGB
                if self.img_num_channels > 1:
                    class_2=[]
                    if self.img_num_channels == 2:
                        for i in range(len(augmented_images_other[0])):
                            class_2.append(data_processing.concat_channels(augmented_images_other[0][i], augmented_images_other[1][i]))
                    else:
                        for i in range(len(augmented_images_other[0])):
                            class_2.append(data_processing.concat_channels(augmented_images_other[0][i], augmented_images_other[1][i], augmented_images_other[2][i]))
                    class_2 = np.array(class_2)
                else:
                    class_2 = augmented_images_other

                #Balance the class sizes
                if self.balance:
                    class_2 = class_2[:len(class_1)]     

                if self.img_num_channels == 1:
                    class_2 = resize(class_2, size=self.best_params['image_size'])
                else:
                    channel1 = resize(class_2[:,:,:,0], size=self.best_params['image_size'])
                    channel2 = resize(class_2[:,:,:,1], size=self.best_params['image_size'])
                    if self.img_num_channels == 2:
                        class_2 = data_processing.concat_channels(channel1, channel2)
                    else:
                        channel3 = resize(class_2[:,:,:,2], size=self.best_params['image_size'])
                        class_2 = data_processing.concat_channels(channel1, channel2, channel3)

                if self.val_blob is not None:
                    if self.img_num_channels == 1:
                        val_class_1 = resize(self.val_blob, size=self.best_params['image_size'])
                    else:
                        val_channel1 = resize(self.val_blob[:,:,:,0], size=self.best_params['image_size'])
                        val_channel2 = resize(self.val_blob[:,:,:,1], size=self.best_params['image_size'])
                        if self.img_num_channels == 2:
                            val_class_1 = data_processing.concat_channels(val_channel1, val_channel2)
                        else:
                            val_channel3 = resize(self.val_blob[:,:,:,2], size=self.best_params['image_size'])
                            val_class_1 = data_processing.concat_channels(val_channel1, val_channel2, val_channel3)
                else:
                    val_class_1 = None

                if self.val_other is not None:
                    if self.img_num_channels == 1:
                        val_class_2 = resize(self.val_other, size=self.best_params['image_size'])
                    elif self.img_num_channels > 1:
                        val_channel1 = resize(self.val_other[:,:,:,0], size=self.best_params['image_size'])
                        val_channel2 = resize(self.val_other[:,:,:,1], size=self.best_params['image_size'])
                        if self.img_num_channels == 2:
                            val_class_2 = data_processing.concat_channels(val_channel1, val_channel2)
                        else:
                            val_channel3 = resize(self.val_other[:,:,:,2], size=self.best_params['image_size'])
                            val_class_2 = data_processing.concat_channels(val_channel1, val_channel2, val_channel3)
                else:
                    val_class_2 = None
            else:
                class_1, class_2 = self.positive_class, self.negative_class
                val_class_1, val_class_2 = self.val_blob, self.val_other

            if self.opt_model:
                if self.best_params['optimizer'] != 'adam' and self.best_params['optimizer'] != 'adagrad':
                    decay = self.best_params['decay']
                    if self.best_params['optimizer'] == 'sgd':
                        momentum = self.best_params['momentum']
                        nesterov = self.best_params['nesterov']
                    else:
                        momentum, nesterov = 0.0, False
                else:
                    decay, momentum, nesterov = 0.0, 0.0, False

                if self.limit_search:
                    self.model, self.history = AlexNet(class_1, class_2, img_num_channels=self.img_num_channels, 
                        normalize=self.normalize, min_pixel=min_pix, max_pixel=max_pix, val_blob=val_class_1, val_other=val_class_2, 
                        epochs=self.train_epochs, batch_size=self.best_params['batch_size'], optimizer=self.best_params['optimizer'], 
                        lr=self.best_params['lr'], momentum=momentum, decay=decay, nesterov=nesterov, loss=self.best_params['loss'], 
                        model_reg=self.best_params['model_reg'], activation_conv=self.best_params['activation_conv'], activation_dense=self.best_params['activation_dense'], 
                        pooling_1=self.best_params['pooling_1'], pooling_2=self.best_params['pooling_2'], pooling_3=self.best_params['pooling_3'],  
                        conv_init=self.best_params['conv_init'], dense_init=self.best_params['dense_init'], conv_reg=self.best_params['conv_reg'], dense_reg=self.best_params['dense_reg'], verbose=self.verbose)
                else:
                    self.model, self.history = AlexNet(class_1, class_2, img_num_channels=self.img_num_channels, 
                        normalize=self.normalize, min_pixel=min_pix, max_pixel=max_pix, val_blob=val_class_1, val_other=val_class_2, 
                        epochs=self.train_epochs, batch_size=self.best_params['batch_size'], optimizer=self.best_params['optimizer'], lr=self.best_params['lr'], momentum=momentum, decay=decay, nesterov=nesterov, 
                        loss=self.best_params['loss'], model_reg=self.best_params['model_reg'], activation_conv=self.best_params['activation_conv'], 
                        activation_dense=self.best_params['activation_dense'], pooling_1=self.best_params['pooling_1'], pooling_2=self.best_params['pooling_2'], 
                        pooling_3=self.best_params['pooling_3'], pool_size_1=self.best_params['pool_size_1'], pool_stride_1=self.best_params['pool_stride_1'], 
                        pool_size_2=self.best_params['pool_size_2'], pool_stride_2=self.best_params['pool_stride_2'], pool_size_3=self.best_params['pool_size_3'], 
                        pool_stride_3=self.best_params['pool_stride_3'], filter_1=self.best_params['filter_1'], filter_size_1=self.best_params['filter_size_1'], 
                        strides_1=self.best_params['strides_1'], filter_2=self.best_params['filter_2'], filter_size_2=self.best_params['filter_size_2'], 
                        strides_2=self.best_params['strides_2'], filter_3=self.best_params['filter_3'], filter_size_3=self.best_params['filter_size_3'], 
                        strides_3=self.best_params['strides_3'], filter_4=self.best_params['filter_4'], filter_size_4=self.best_params['filter_size_4'], 
                        strides_4=self.best_params['strides_4'], filter_5=self.best_params['filter_5'], filter_size_5=self.best_params['filter_size_5'], 
                        strides_5=self.best_params['strides_5'], dense_neurons_1=self.best_params['dense_neurons_1'], dense_neurons_2=self.best_params['dense_neurons_2'], 
                        dropout_1=self.best_params['dropout_1'], dropout_2=self.best_params['dropout_2'], conv_init=self.best_params['conv_init'], dense_init=self.best_params['dense_init'],
                        conv_reg=self.best_params['conv_reg'], dense_reg=self.best_params['dense_reg'], verbose=self.verbose)
            else: 
                self.model, self.history = AlexNet(class_1, class_2, img_num_channels=self.img_num_channels, normalize=self.normalize,
                    min_pixel=min_pix, max_pixel=max_pix, val_blob=val_class_1, val_other=val_class_2, epochs=self.epochs, optimizer=self.best_params['optimizer'],
                    verbose=self.verbose)
       
        return 

    def save(self, path=None, overwrite=False):
        """
        Saves the trained classifier in a new directory named 'MicroLIA_models', 
        as well as the imputer and the features to use, if applicable.
        
        Args:
            path (str): Absolute path where the data folder will be saved
                Defaults to None, in which case the directory is saved to the
                local home directory.
            overwrite (bool, optional): If True the 'MicroLIA_models' folder this
                function creates in the specified path will be deleted if it exists
                and created anew to avoid duplicate files. 
        """

        if self.model is None:
            print('The model has not been created!')

        if path is None:
            path = str(Path.home())
        if path[-1] != '/':
            path+='/'

        try:
            os.mkdir(path+'MicroLIA_cnn_model')
        except FileExistsError:
            if overwrite:
                try:
                    os.rmdir(path+'MicroLIA_cnn_model')
                except OSError:
                    for file in os.listdir(path+'MicroLIA_cnn_model'):
                        os.remove(path+'MicroLIA_cnn_model/'+file)
                    os.rmdir(path+'MicroLIA_cnn_model')
                os.mkdir(path+'MicroLIA_cnn_model')
            else:
                raise ValueError('Tried to create "MicroLIA_cnn_model" directory in specified path but folder already exists! If you wish to overwrite set overwrite=True.')
        
        path += 'MicroLIA_cnn_model/'
        if self.model is not None:      
            np.savetxt(path+'model_acc', self.history.history['binary_accuracy'])
            np.savetxt(path+'model_loss', self.history.history['loss'])
            np.savetxt(path+'model_f1', self.history.history['f1_score'])
            if self.val_positive is not None:
                np.savetxt(path+'model_val_acc', self.history.history['val_binary_accuracy'])
                np.savetxt(path+'model_val_loss', self.history.history['val_loss'])
                np.savetxt(path+'model_val_f1', self.history.history['val_f1_score'])

            save_model(self.model, path+'Keras_Model.h5')#,  custom_objects={'f1_score': f1_score})

        if self.best_params is not None:
            joblib.dump(self.best_params, path+'Best_Params')
        if self.optimization_results is not None:
            joblib.dump(self.optimization_results, path+'HyperOpt_Results')
        print('Files saved in: {}'.format(path))

        self.path = path

        return 

    def load(self, path=None):
        """ 
        Loads the model, imputer, and feats to use, if created and saved.
        This function will look for a folder named 'MicroLIA_models' in the
        local home directory, unless a path argument is set. 

        Args:
            path (str): Path where the directory 'MicroLIA_models' is saved. 
            Defaults to None, in which case the folder is assumed to be in the 
            local home directory.
        """

        if path is None:
            path = str(Path.home())
        if path[-1] != '/':
            path+='/'

        path += 'MicroLIA_cnn_model/'

        try:
            self.model = load_model(path+'Keras_Model.h5', custom_objects={'f1_score': f1_score})
            model = 'model,'
        except:
            model = ''
            pass

        try:
            self.optimization_results = joblib.load(path+'HyperOpt_Results')
            optimization_results = 'optimization_results,'
        except:
            optimization_results = '' 
            pass

        try:
            self.best_params = joblib.load(path+'Best_Params')
            best_params = 'best_params'
        except:
            best_params = '' 
            pass        

        print('Successfully loaded the following class attributes: {} {} {}'.format(model, optimization_results, best_params))
        
        self.path = path
        
        return

    def predict(self, data, target='ML', return_proba=False):
        """
        Returns the class prediction. The input can either be a single 2D array 
        or a 3D array if there are multiple samples.

        Args:
            data: 2D array for single image, 3D array for multiple images.
            target (str): The name of the target class, assuming binary classification in 
                which there is an 'OTHER' class. Defaults to 'ML'. 
            return_proba (bool): If True the output will return the probability prediction.
                Defaults to False. 

        Returns:
            The class prediction(s).
        """
      
        data = process_class(data, normalize=self.normalize, min_pixel=self.min_pixel, max_pixel=self.max_pixel, img_num_channels=self.img_num_channels)
        predictions = self.model.predict(data)

        output, probas = [], [] 
        for i in range(len(predictions)):
            if np.argmax(predictions[i]) == 1:
                prediction = target
                probas.append(predictions[i][1])
            else:
                prediction = 'OTHER'
                probas.append(predictions[i][0])

            output.append(prediction)

        if return_proba:
            output = np.c_[output, probas]
            
        return np.array(output)

    def plot_hyper_opt(self, baseline=None, xlim=None, ylim=None, xlog=True, ylog=False, 
        savefig=False):
        """
        Plots the hyperparameter optimization history.
    
        Args:
            baseline (float): Baseline accuracy achieved when using only
                the default engine hyperparameters. If input a vertical
                line will be plot to indicate this baseline accuracy.
                Defaults to None.
            xlim: Limits for the x-axis. Ex) xlim = (0, 1000)
            ylim: Limits for the y-axis. Ex) ylim = (0.9, 0.94)
            xlog (boolean): If True the x-axis will be log-scaled.
                Defaults to True.
            ylog (boolean): If True the y-axis will be log-scaled.
                Defaults to False.
            savefig (bool): If True the figure will not disply but will be saved instead.
                Defaults to False. 

        Returns:
            AxesImage
        """

        trials = self.optimization_results.get_trials()
        trial_values, best_value = [], []
        for trial in range(len(trials)):
            try:
                value = trials[trial].values[0]
            except TypeError:
                value = np.min(trial_values) 
            trial_values.append(value)
            if trial == 0:
                best_value.append(value)
            else:
                if any(y > value for y in best_value): #If there are any numbers in best values that are higher than current one
                    best_value.append(np.array(best_value)[trial-1])
                else:
                    best_value.append(value)

        best_value, trial_values = np.array(best_value), np.array(trial_values)
        best_value[1] = trial_values[1] #Make the first trial the best model, since technically it is.
        for i in range(2, len(trial_values)):
            if trial_values[i] < best_value[1]:
                best_value[i] = best_value[1]
            else:
                break

        if baseline is not None:
            plt.axhline(y=baseline, color='k', linestyle='--', label='Baseline Model')
            ncol=3
        else:
            ncol=2

        if self.metric == 'val_accuracy':
            ylabel = 'Validation Accuracy'
        elif self.metric == 'accuracy':
            ylabel = 'Training Accuracy'
        elif self.metric == 'val_loss':
            ylabel = '1 - Validation Loss'
        elif self.metric == 'loss':
            ylabel = '1 - Training Loss'
        else:
            ylabel = 'Optimization Metric'

        plt.plot(range(len(trials)), best_value, color='r', alpha=0.83, linestyle='-', label='Best Model')
        plt.scatter(range(len(trials)), trial_values, c='b', marker='+', s=35, alpha=0.45, label='Trial')
        plt.xlabel('Trial #', alpha=1, color='k')
        plt.ylabel(ylabel, alpha=1, color='k')
        plt.title('CNN Hyperparameter Optimization')#, size=18) Make this a f" string option!!
        plt.grid(False)
        if xlim is not None:
            plt.xlim(xlim)
        else:
            plt.xlim((1, len(trials)))
        if ylim is not None:
            plt.ylim(ylim)
        if xlog:
            plt.xscale('log')
        if ylog:
            plt.yscale('log')
        #plt.tight_layout()
        #plt.legend(prop={'size': 12}, loc='upper left')
        plt.legend(loc='upper center', ncol=ncol, frameon=False)#, handlelength=4)#prop={'size': 14}
        plt.rcParams['axes.facecolor']='white'
        
        if savefig:
            plt.savefig('CNN_Hyperparameter_Optimization.png', bbox_inches='tight', dpi=300)
            plt.clf()
        else:
            plt.show()

    def plot_hyper_param_importance(self, plot_time=True, savefig=False):
        """
        Plots the hyperparameter optimization history.
    
        Args:
            plot_tile (bool): If True, the importance on the duration will also be included. Defaults to True.
            savefig (bool): If True the figure will not disply but will be saved instead. Defaults to False. 

        Returns:
            AxesImage
        """

        try:
            if isinstance(self.path, str):
                try:
                    hyper_importances = joblib.load(self.path+'Hyperparameter_Importance')
                except FileNotFoundError:
                    raise ValueError('Could not find the importance file in the '+self.path+' folder')

                try:
                    duration_importances = joblib.load(self.path+'Duration_Importance')
                except FileNotFoundError:
                    raise ValueError('Could not find the importance file in the '+self.path+' folder')
            else:
                raise ValueError('Call the save_hyper_importance() attribute first.')
        except:
            raise ValueError('Call the save_hyper_importance() attribute first.')

        params, importance, duration_importance = [], [], []
        for key in hyper_importances:       
            params.append(key)

        for name in params:
            importance.append(hyper_importances[name])
            duration_importance.append(duration_importances[name])

        xtick_labels = format_labels(params)

        fig, ax = plt.subplots()
        #fig.subplots_adjust(top=0.8)
        ax.barh(xtick_labels, importance, label='Importance for Classification', color=mcolors.TABLEAU_COLORS["tab:blue"], alpha=0.87)
        if plot_time:
            ax.barh(xtick_labels, duration_importance, label='Impact on Engine Speed', color=mcolors.TABLEAU_COLORS["tab:orange"], alpha=0.7, hatch='/')

        ax.set_ylabel("Hyperparameter")
        ax.set_xlabel("Importance Evaluation")
        ax.legend(ncol=2, frameon=False, handlelength=2, bbox_to_anchor=(0.5, 1.1), loc='upper center')
        ax.set_xscale('log')
        plt.gca().invert_yaxis()
        plt.xlim((0, 1.))#np.max(importance+duration_importance)))#np.max(importance+duration_importance)))
        #fig = plot_param_importances(self.optimization_results)
        #fig = plot_param_importances(self.optimization_results, target=lambda t: t.duration.total_seconds(), target_name="duration")
        #plt.tight_layout()
        if savefig:
            if plot_time:
                plt.savefig('CNN_Hyperparameter_Importance.png', bbox_inches='tight', dpi=300)
            else:
                plt.savefig('CNN_Hyperparameter_Duration_Importance.png', bbox_inches='tight', dpi=300)
            plt.clf()
        else:
            plt.show()

    def save_hyper_importance(self):
        """
        Calculates and saves binary files containing
        dictionaries with importance information, one
        for the importance and one for the duration importance

        Note:
            This procedure is time-consuming but must be run once before
            plotting the importances. This function will save
            two files in the model folder for future use. 

        Returns:
            Saves two binary files, importance and duration importance.
        """

        print('Calculating and saving importances, this could take up to an hour...')

        try:
            if isinstance(self.path, str):
                path = self.path  
            else:
                path = str(Path.home())
        except:
            path = str(Path.home())

        hyper_importance = get_param_importances(self.optimization_results)
        joblib.dump(hyper_importance, path+'Hyperparameter_Importance')

        importance = FanovaImportanceEvaluator()
        duration_importance = importance.evaluate(self.optimization_results, target=lambda t: t.duration.total_seconds())
        joblib.dump(duration_importance, path+'Duration_Importance')
        
        print(f"Files saved in: {path}")

        self.path = path

        return  

    def plot_performance(self, metric='binary_accuracy', combine=False, ylabel=None, title=None,
        xlim=None, ylim=None, xlog=False, ylog=True, savefig=False):
        """
        Plots the training/performance histories.
    
        Args:
            metric (str): Metric to plot, options are: 'binary_accuracy', 'f1_score', 'loss'. Defaults to 'binary_accuracy'
            combine (bool): If True the validation history will also be included, if applicable.
            ylabel (str, optional): The y-label of the plot.
            title (str, optional): The title of the plot.
            xlim (tuple, optional): The xlim range, matplotlib style.
            ylim (tuple, optional): The ylim range, matplotlib style.
            xlog (bool): Whether to log-scale the x-axis. Defaults to False.
            ylog (bool): Whether to log-scale the y-axis. Defaults to False.
            savefig (bool): If True the figure will not disply but will be saved instead. Defaults to False. 

        Returns:
            AxesImage
        """

        metric1 = np.loadtxt(self.path+'model_'+metric)

        if combine:
            if 'val' not in metric:
                metric2 = np.loadtxt(self.path+'model_val_'+metric)
                label1, label2 = 'Training', 'Validation'
            else:
                metric2 = np.loadtxt(self.path+'model_'+metric)
                label1, label2 = 'Validation', 'Training'
        else:
            if 'val' not in metric:
                label1 = 'Training'
            else:
                label1 = 'Validation'

        plt.plot(range(1, len(metric1)+1), metric1, color='r', alpha=0.83, linestyle='-', label=label1)
        if combine:
            plt.plot(range(1, len(metric2)+1), metric2, color='b', alpha=0.83, linestyle='--', label=label2)

        if ylabel is None:
            ylabel = metric
        if title is None:
            title = metric

        plt.ylabel(ylabel, alpha=1, color='k')
        plt.title(title)
        plt.xlabel('Epoch', alpha=1, color='k'), plt.grid(False)
        if xlim is not None:
            plt.xlim(xlim)
        else:
            plt.xlim((1, len(metric1)))
        if ylim is not None:
            plt.ylim(ylim)
        if xlog:
            plt.xscale('log')
        if ylog:
            plt.yscale('log')
        plt.legend(loc='upper center', frameon=False) #ncol
        plt.rcParams['axes.facecolor']='white'
        
        if savefig:
            plt.savefig('CNN_Training_History_'+metric+'.png', bbox_inches='tight', dpi=300)
            plt.clf()
        else:
            plt.show()


def AlexNet(positive_class, negative_class, img_num_channels=1, normalize=True, 
        min_pixel=0, max_pixel=100, val_positive=None, val_negative=None, epochs=100, 
        batch_size=32, lr=0.0001, momentum=0.9, decay=0.0, nesterov=False, 
        loss='binary_crossentropy', conv_init='uniform_scaling', dense_init='TruncatedNormal',
        activation_conv='relu', activation_dense='relu', model_reg='local_response', padding='same', 
        pooling_1='max', pooling_2='max', pooling_3='max', pool_size_1=3, pool_stride_1=2, pool_size_2=3, 
        pool_stride_2=2, pool_size_3=3, pool_stride_3=2, filter_1=96, filter_size_1=11, strides_1=4, 
        filter_2=256, filter_size_2=5, strides_2=1, filter_3=384, filter_size_3=3, strides_3=1, filter_4=384, 
        filter_size_4=3, strides_4=1, filter_5=256, filter_size_5=3, strides_5=1, dense_neurons_1=4096, 
        dense_neurons_2=4096, dropout_1=0.5, dropout_2=0.5, conv_reg=0, dense_reg=0, optimizer='sgd', 
        early_stop_callback=None, checkpoint=False, verbose=1):
        """
        The CNN model infrastructure presented by the 2012 ImageNet Large Scale 
        Visual Recognition Challenge, AlexNet. Parameters were adapted for
        our astronomy case of detecting diffuse emission.

        To avoid exploding gradients we need to normalize our pixels to be 
        between 0 and 1. By default normalize=True, which will perform
        min-max normalization using the min_pixel and max_pixel arguments, 
        which should be set carefully.

        Args:
            positive_class (ndarray): 3D array containing more than one image of diffuse objects.
            negative_class (ndarray): 3D array containing more than one image of non-diffuse objects.
            img_num_channels (int): The number of filters used. Defaults to 1, as pyBIA version 1
                has been trained with only blue broadband data.
            normalize (bool, optional): If True the data will be min-max normalized using the 
                input min and max pixels. Defaults to True.
            min_pixel (int, optional): The minimum pixel count, defaults to 638. 
                Pixels with counts below this threshold will be set to this limit.
            max_pixel (int, optional): The maximum pixel count, defaults to 3000. 
                Pixels with counts above this threshold will be set to this limit.
            val_positive (array, optional): 3D matrix containing the 2D arrays (images)
                to be used for validationm, for the positive class. Defaults to None.
            val_negative(array, optional): 3D matrix containing the 2D arrays (images)
                to be used for validationm, for the negative class. Defaults to None.
            epochs (int): Number of epochs used for training. 
            batch_size (int): The size of each sub-sample used during the training
                epoch. Large batches are likely to get stuck in local minima. Defaults to 32.
            lr (float): Learning rate, the rate at which the model updates the gradient. Defaults to 0.0001
            momentum (float): Momentum is a float greater than 0 that accelerates gradient descent. Defaults to 0.9.
            decay (float): The rate of learning rate decay applied after each epoch. Defaults to 0.0005. It is recommended
                to set decay to the learning rate divded by the total number of epochs.
            nesterov (bool): Whether to apply Nesterov momentum or not. Defaults to False.
            loss (str): The loss function used to calculate the gradients. Defaults to 'categorical_crossentropy'.
                Loss functions can be set by calling the Keras API losses module.
            conv_init (str): Weight initializer for the convolutional layers.
            dense_init (str): Weight initializer for the dense layers.
            activation_conv (str): Activation function to use for the convolutional layer. Default is 'relu'.'
            activation_dense (str): Activation function to use for the dense layers. Default is 'tanh'.
            model_reg (str): The model regularization technique to use, can be None.
            padding (str): Either 'same' or 'valid'. When set to 'valid', the dimensions reduce as the boundary 
                that doesn't make it within even convolutions get cuts off. Defaults to 'same', which applies
                zero-value padding around the boundary, ensuring even convolutional steps across each dimension.
            dropout (float): Droupout rate after the dense layers. This is the percentage of dense neurons
                that are turned off at each epoch. This prevents inter-neuron depedency, and thus overfitting. 
            pooling (bool): True to enable max pooling, false to disable. 
                Note: Max pooling can result in loss of positional information, it computation allows
                setting pooling=False may yield more robust accuracy.
            pool_size (int, optional): The pool size of the max pooling layers. Defaults to 3.
            pool_stride (int, optional): The stride to use in the max pooling layers. Defaults to 2.
            early_stop_callback (list, optional): Callbacks for early stopping and pruning with Optuna, defaults
                to None. Should only be used during optimization, refer to pyBIA.optimization.objective_cnn().
            checkpoint (bool, optional): If False no checkpoint will be saved. Defaults to True.
            verbose (int): Controls the amount of output printed during the training process. A value of 0 is for silent mode, 
                a value of 1 is used for progress bar mode, and 2 for one line per epoch mode. Defaults to 1.

        Returns:
            The trained CNN model and accompanying history.
        """
        
        if len(positive_class.shape) != len(negative_class.shape):
            raise ValueError("Shape of blob and other data must be the same.")
        if batch_size < 16 and model_reg == 'batch_norm':
            print("Batch Normalization can be unstable with low batch sizes, if loss returns nan try a larger batch size and/or smaller learning rate.")
        
        if val_blob is not None:
            val_X1, val_Y1 = process_class(val_blob, label=1, img_num_channels=img_num_channels, min_pixel=min_pixel, max_pixel=max_pixel, normalize=normalize)
            if val_other is None:
                val_X, val_Y = val_X1, val_Y1
            else:
                val_X2, val_Y2 = process_class(val_other, label=0, img_num_channels=img_num_channels, min_pixel=min_pixel, max_pixel=max_pixel, normalize=normalize)
                val_X, val_Y = np.r_[val_X1, val_X2], np.r_[val_Y1, val_Y2]
        else:
            if val_other is None:
                val_X, val_Y = None, None
            else:
                val_X2, val_Y2 = process_class(val_other, label=0, img_num_channels=img_num_channels, min_pixel=min_pixel, max_pixel=max_pixel, normalize=normalize)
                val_X, val_Y = val_X2, val_Y2

        img_width = positive_class[0].shape[0]
        img_height = positive_class[0].shape[1]
        
        ix = np.random.permutation(len(positive_class))
        positive_class = positive_class[ix]

        ix = np.random.permutation(len(negative_class))
        negative_class = negative_class[ix]

        X_train, Y_train = create_training_set(positive_class, negative_class, normalize=normalize, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)

        if normalize:
            X_train[X_train > 1] = 1
            X_train[X_train < 0] = 0
            
        num_classes, input_shape = 2, (img_width, img_height, img_num_channels)
       
        if verbose == 1:
            print_params(batch_size, lr, decay, momentum, nesterov, loss, optimizer, model_reg, conv_init, activation_conv, dense_init, activation_dense, 
                filter_size_1, filter_size_2, filter_size_3, filter_size_4, filter_size_5, pooling_1, pooling_2, pooling_3, pool_size_1, pool_size_2, pool_size_3, conv_reg, dense_reg)

        #Uniform scaling initializer
        if conv_init == 'uniform_scaling':
            conv_init = VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform', seed=None)
        if dense_init == 'uniform_scaling':
            dense_init = VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform', seed=None)

        #Call the appropriate tf.keras.losses.Loss function
        loss = get_loss_function(loss)

        #Model configuration
        model = Sequential()
        
        #Convolutional layers
        model.add(Conv2D(filter_1, filter_size_1, strides=strides_1, activation=activation_conv, input_shape=input_shape,
                         padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))
        if pooling_1 == 'min':
            model.add(tf.keras.layers.Lambda(lambda x: -tf.nn.max_pool2d(-x, ksize=(pool_size_1, pool_size_1), strides=(pool_stride_1, pool_stride_1), padding='SAME')))
        elif pooling_1 == 'max':
            model.add(MaxPool2D(pool_size=pool_size_1, strides=pool_stride_1, padding=padding))
        elif pooling_1 == 'average':
            model.add(AveragePooling2D(pool_size=pool_size_1, strides=pool_stride_1, padding=padding)) 
        if model_reg == 'local_response':
            model.add(Lambda(lambda x: tf.nn.local_response_normalization(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75))) #Lambda allows for the application of the lambda function to the input tensor
        elif model_reg == 'batch_norm':
            model.add(BatchNormalization())

        model.add(Conv2D(filter_2, filter_size_2, strides=strides_2, activation=activation_conv, padding=padding,
                         kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))
        if pooling_2 == 'min':
            model.add(tf.keras.layers.Lambda(lambda x: -tf.nn.max_pool2d(-x, ksize=(pool_size_2, pool_size_2), strides=(pool_stride_2, pool_stride_2), padding='SAME')))
        elif pooling_2 == 'max':
            model.add(MaxPool2D(pool_size=pool_size_2, strides=pool_stride_2, padding=padding))
        elif pooling_2 == 'average':
            model.add(AveragePooling2D(pool_size=pool_size_2, strides=pool_stride_2, padding=padding))            
        if model_reg == 'local_response':
            model.add(Lambda(lambda x: tf.nn.local_response_normalization(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75)))
        elif model_reg == 'batch_norm':
            model.add(BatchNormalization())

        model.add(Conv2D(filter_3, filter_size_3, strides=strides_3, activation=activation_conv, padding=padding,
                         kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))
        if model_reg == 'batch_norm':
            model.add(BatchNormalization())

        model.add(Conv2D(filter_4, filter_size_4, strides=strides_4, activation=activation_conv, padding=padding,
                         kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))
        if model_reg == 'batch_norm':
            model.add(BatchNormalization())

        model.add(Conv2D(filter_5, filter_size_5, strides=strides_5, activation=activation_conv, padding=padding,
                         kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))
        if pooling_3 == 'min':
            model.add(tf.keras.layers.Lambda(lambda x: -tf.nn.max_pool2d(-x, ksize=(pool_size_3, pool_size_3), strides=(pool_stride_3, pool_stride_3), padding='SAME')))
        if pooling_3 == 'max':
            model.add(MaxPool2D(pool_size=pool_size_3, strides=pool_stride_3, padding=padding))
        elif pooling_3 == 'average':
            model.add(AveragePooling2D(pool_size=pool_size_3, strides=pool_stride_3, padding=padding))            
        if model_reg == 'batch_norm':
            model.add(BatchNormalization())

        #Dense layers
        model.add(Flatten())

        #FCC 1
        model.add(Dense(dense_neurons_1, activation=activation_dense, kernel_initializer=dense_init, kernel_regularizer=l2(dense_reg)))
        model.add(Dropout(dropout_1))
        if model_reg == 'batch_norm':
            model.add(BatchNormalization())
        
        #FCC 2
        model.add(Dense(dense_neurons_2, activation=activation_dense, kernel_initializer=dense_init, kernel_regularizer=l2(dense_reg)))
        model.add(Dropout(dropout_2))
        if model_reg == 'batch_norm':
            model.add(BatchNormalization())

        #Output layer
        model.add(Dense(num_classes, activation='sigmoid', kernel_initializer=dense_init)) #adding a kernel model_reg has no effect if sigmoid is being used, but works for layers with trainable weights like softmax
        if model_reg == 'batch_norm':
            model.add(BatchNormalization())

        #Call the appropriate tf.keras.optimizers function
        optimizer = get_optimizer(optimizer, lr, momentum=momentum, decay=decay, nesterov=nesterov)

        #Compile the Model
        model.compile(loss=loss, optimizer=optimizer, metrics=[tf.keras.metrics.BinaryAccuracy(), f1_score])
        
        #Optional checkpoint callback, with the monitor hardcoded to loss of the validation data.
        callbacks_list = []
        if checkpoint:
            callbacks_list.append(ModelCheckpoint(str(Path.home())+'/'+'checkpoint.hdf5', monitor='val_binary_loss', verbose=2, save_best_only=True, mode='min'))
        
        #Early stop callback for use during the optimization routine
        if early_stop_callback is not None:
            callbacks_list.append(early_stop_callback)

        #Fit the Model
        if val_X is None:
            history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, verbose=verbose)
        else:
            history = model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(val_X, val_Y), epochs=epochs, callbacks=callbacks_list, verbose=verbose)

        return model, history


### Score and Loss Functions ###

def f1_score(y_true, y_pred):
    """
    Computes the F1 score between true and predicted labels.

    Args:
        y_true (tensor): The true labels.
        y_pred (tensor): The predicted labels.

    Returns:
        The F1 score between true and predicted labels.
    """

    tp = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    fp = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred - y_true, 0, 1)))
    fn = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())

    return 2.0 * precision * recall / (precision + recall + tf.keras.backend.epsilon())

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    This function computes focal loss, which is used to address class imbalance in classification tasks.

    Args:
        y_true (Tensor): The ground truth labels.
        y_pred (Tensor): The predicted labels.
        gamma (float, optional): The focusing parameter. Defaults to 2.0.
        alpha (float, optional): The weighting parameter. Defaults to 0.25.

    Returns:
        Tensor: The computed focal loss.
    """

    ce = tf.keras.backend.binary_crossentropy(y_true, y_pred, from_logits=True)
    pt = tf.math.exp(-ce)

    return alpha * tf.math.pow(1.0 - pt, gamma) * ce

def dice_loss(y_true, y_pred, smooth=1e-7):
    """
    This function computes the Dice loss, which is a similarity metric commonly used in segmentation tasks.

    Args:
        y_true (Tensor): The ground truth labels.
        y_pred (Tensor): The predicted labels.
        smooth (float, optional): A smoothing parameter to prevent division by zero. Defaults to 1e-7.

    Returns:
        Tensor: The computed Dice loss.
    """

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2.0 * intersection + smooth) / (union + smooth)

    return 1.0 - dice

def jaccard_loss(y_true, y_pred, smooth=1e-7):
    """
    This function computes the Jaccard loss, which is another similarity metric used in segmentation tasks.

    Args:
        y_true (Tensor): The ground truth labels.
        y_pred (Tensor): The predicted labels.
        smooth (float, optional): A smoothing parameter to prevent division by zero. Defaults to 1e-7.

    Returns:
        Tensor: The computed Jaccard loss.
    """

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    jaccard = (intersection + smooth) / (union + smooth)

    return 1.0 - jaccard


### AlexNet Helper Functions ###

def get_optimizer(optimizer, lr, momentum=None, decay=None, nesterov=False):
    """
    Returns an optimizer object based on the specified parameters.

    Args:
        optimizer (str): Name of the optimizer.
        lr (float): Learning rate.
        momentum (float): Momentum for SGD optimizer (default=None).
        decay (float): Learning rate decay for RMSprop optimizer (default=None).
        nesterov (bool): Whether to use Nesterov momentum for SGD optimizer (default=False).

    Returns:
        optimizer: Optimizer object.

    Raises:
        ValueError: If an invalid optimizer name is provided.
    """

    if optimizer == 'sgd':
        optimizer = SGD(learning_rate=lr, momentum=momentum, decay=decay, nesterov=nesterov)
    elif optimizer == 'adam':
        optimizer = Adam(learning_rate=lr)
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate=lr, decay=decay)
    elif optimizer == 'adagrad':
        optimizer = Adagrad(learning_rate=lr)
    else:
        raise ValueError("If an invalid optimizer name is provided, options are: 'sgd', 'adam', 'rmsprop', 'adagrad'")

    return optimizer

def get_loss_function(loss):
    """
    Returns the specified loss function as a Keras loss object.

    Args:
        loss (str): The name of the loss function to use. Possible values are: 'hinge', 'squared_hinge', 'kld', 
            'logcosh', 'focal_loss', 'dice_loss', and 'jaccard_loss'.

    Returns:
        tf.keras.losses.Loss: The Keras loss object for the specified loss function.

    Raises:
        ValueError: If an invalid loss function name is provided, options are: 'hinge', 'square_hinge', 'kld', 'logcosh', 'focal_loss', 'dice_loss', 'jaccard_loss'
    """

    if loss == 'binary_crossentropy':
        return loss 
    elif loss == 'hinge':
        return Hinge()
    elif loss == 'squared_hinge':
        return SquaredHinge()
    elif loss == 'kld':
        return KLDivergence()
    elif loss == 'logcosh':
        return LogCosh()
    elif loss == 'focal_loss':
        return focal_loss
    elif loss == 'dice_loss':
        return dice_loss
    elif loss == 'jaccard_loss':
        return jaccard_loss
    else:
        raise ValueError("Invalid loss function name")
   
def print_params(batch_size, lr, decay, momentum, nesterov, loss, optimizer, model_reg, conv_init, activation_conv, dense_init, activation_dense,
    filter_size_1, filter_size_2, filter_size_3, filter_size_4, filter_size_5, pooling_1, pooling_2, pooling_3, pool_size_1, pool_size_2, pool_size_3,
    conv_reg, dense_reg):
    """
    Prints the model training parameters and architecture parameters.

    Args:
        batch_size (int): The number of samples per gradient update.
        lr (float): The learning rate for the model.
        decay (float): The learning rate decay over each update.
        momentum (float): The SGD optimizer momentum.
        nesterov (bool): Whether to apply Nesterov momentum.
        loss (str): The loss function used for the model.
        optimizer (str): The optimization algorithm used to train the model.
        model_reg (str): The regularization method used to prevent overfitting.
        conv_init (str): The initialization method used for the convolutional layers.
        activation_conv (str): The activation function used for the convolutional layers.
        dense_init (str): The initialization method used for the dense layers.
        activation_dense (str): The activation function used for the dense layers.
    """

    print()
    print('===== Training Parameters =====')
    print()
    print('|| Batch Size : '+str(batch_size), '|| Loss Function : '+loss, '||')
    if optimizer == 'sgd':
        print('|| Optimizer : '+optimizer, '|| lr : '+str(lr), '|| Decay : '+str(decay), '|| Momentum : '+str(momentum), '|| Nesterov : '+str(nesterov)+' ||')
    elif optimizer == 'rmsprop':
        print('|| Optimizer : '+optimizer, '|| lr : '+str(lr), '|| Decay : '+str(decay)+' ||')
    else:
        print('|| Optimizer : '+optimizer, '|| lr : '+str(lr)+' ||')
    print()
    print('=== Architecture Parameters ===')
    print()
    print('Model Regularizer : '+ model_reg)
    print('Convolutional L2 Regularizer : '+ str(conv_reg))
    print('Convolutional Initializer : '+ conv_init)
    print('Convolutional Activation Fn : '+ activation_conv)
    print('Dense L2 Regularizer : '+ str(dense_reg))
    print('Dense Initializer : '+ dense_init)
    print('Dense Activation Fn : '+ activation_dense)
    print()
    print('======= Layer Parameters ======')
    print()
    print('Filter 1 || Size : {}, Pooling : {}, Pooling Size : {}'.format(filter_size_1, pooling_1, pool_size_1))
    print('Filter 2 || Size : {}, Pooling : {}, Pooling Size : {}'.format(filter_size_2, pooling_2, pool_size_2))
    print('Filter 3 || Size : {}, Pooling : {}, Pooling Size : {}'.format(filter_size_3, pooling_3, pool_size_3))
    print('Filter 4 || Size : {}'.format(filter_size_4))
    print('Filter 5 || Size : {}'.format(filter_size_5))
    print()
    print('===============================')
    print()

def format_labels(labels: list) -> list:
    """
    Takes a list of labels and returns the list with all words capitalized and underscores removed.
    Also replaces 'eta' with 'Learning Rate' and 'n_estimators' with 'Number of Trees'.
    
    Args:
        labels (list): A list of strings.
    
    Returns:
        Reformatted list.
    """

    new_labels = []
    for label in labels:
        label = label.replace("_", " ")
        if label == "lr":
            new_labels.append("Learning Rate")
            continue
        new_labels.append(label.title())

    return new_labels




