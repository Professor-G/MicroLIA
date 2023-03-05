#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  25 10:39:23 2023

@author: daniel
"""

import os, sys
import copy, gc
import tensorflow as tf
os.environ['PYTHONHASHSEED']= '0' #, os.environ["TF_DETERMINISTIC_OPS"] =1
import numpy as np
import random as python_random
##https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development##
np.random.seed(1909), python_random.seed(1909), tf.random.set_seed(1909)
from pandas import DataFrame
from warnings import filterwarnings
filterwarnings("ignore", category=FutureWarning)
from collections import Counter 

import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score

from skopt import BayesSearchCV, plots, gp_minimize
from skopt.plots import plot_convergence, plot_objective
from skopt.space import Real, Integer, Categorical 
from tensorflow.keras.backend import clear_session 
from tensorflow.keras.callbacks import EarlyStopping, Callback

import optuna
from boruta import BorutaPy
from BorutaShap import BorutaShap
from xgboost import XGBClassifier, DMatrix, train
from optuna.integration import TFKerasPruningCallback
optuna.logging.set_verbosity(optuna.logging.WARNING)
from MicroLIA.data_augmentation import augmentation, resize
from MicroLIA import data_processing, cnn_model


class objective_cnn(object):
    """
    Optimization objective function for MicroLIA's convolutional neural network. 
    This is passed through the hyper_opt() function when optimizing with
    Optuna. The Optuna software for hyperparameter optimization was published in 
    2019 by Akiba et al. Paper: https://arxiv.org/abs/1907.10902

    Unlike the objective functions for the ensemble algorithms, this takes as input
    the two classes directly, positive_class & negative_class, instead of the traditional 
    data (data_x) and accompanying label array (data_y). This is because the cnn_model.AlexNet() 
    function takes as input the two classes separately, after which it automatically assigns the 1 
    and 0 labels, respectively.

    By default, opt_model=True and opt_aug=False. If opt_aug=True, the image size and
    number of augmentations to perform on each image will be included as optimizable variables.
    If the opt_max_min_pix and opt_max_max_pix are set, the optimal maximum pixel value to use
    when min-max normalizing each band will also be optimizable variables that will be tuned and returned.
    If opt_model=True, additional architecture parameters will also be tuned. In both instances the training parameters 
    will be optimized, which include the optimizer, learning rate, decay, and momentum, if applicable.

    Note:
        If opt_aug is enabled, then the positive_class sample will be the data that will augmented.
        It is best to keep both class sizes the same after augmentation, therefore balance=True
        by default, which will truncate the negative_class sample to match the augmented positive_class size.
        The other class will also undergo the same data augmentation technique but only one image will be returned
        per sample, this is to ensure both classes contain the same level of variation, especially important
        when applying the mask-cutout technique to the augmentation procedure.

        Since the maximum number of augmentations allowed is batch_max per each sample, in practice negative_class 
        should contain batch_max times the size of positive_class. During the optimization procedure, if an 
        augmentation batch size of 100 is assesed and positive class contains n samples, then 100*n augmented images will be created, 
        and therefore during that particular trial the first 100*n samples from negative_class will be used, if available. 
        To use the entire negative_class sample regardless of the number augmentations performed, set balance=False.
    
        The min_pixel and max_pixel value will be used to min-max normalize the images, if normalize=True. The opt_max_min_pix
        and opt_max_max_pix, when set, will be used instead during the optimization procedure, so as to determine
        what optimal maximum value to use when normalizing the images. This is important when dealing with deep-survey data. 
        If optimizing the normalization scheme, the default min_pixel will be set to zero, as such only the optimal max_pixel 
        for every band will be output.

    Args:
        positive_class (ndarray): The samples for the first class should be passed, which will automatically 
            be assigned the positive label '1'.
        negative_class (ndarray, str): The samples for the second class should be passed, which will automatically 
            be assigned the negative label '0'.
        img_num_channels (int): The number of filters. Defaults to 1.
        normalize (bool, optional): If True the data will be min-max normalized using the 
            input min and max pixels. Defaults to True.
        min_pixel (int, optional): The minimum pixel count, pixels with counts 
            below this threshold will be set to this limit. Defaults to 0.
        max_pixel (int, list, optional): The maximum pixel count, pixels with counts 
            above this threshold will be set to this limit. Defaults to 100. If img_num_channels
            is not 1, the max_pixel should be a list containing two values, one for each band.
        val_positive (ndarray, optional): Positive class data to be used for validation. Defaults to None.
        val_negative (ndarray, optional): Negative class data to be used for validation. Defaults to None.
        test_positive (ndarray, optional): Positive class data to be used for post-trial testing. Defaults to None.
        test_negative (ndarray, optional): Negative class data to be used for post-trial testing. Defaults to None.
        train_epochs (int): Number of epochs to the train the CNN to during the optimization trials. Defaults to 25.
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
            regularizer. Only applicable if opt_aug=True. Defaults to None.
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
        batch_size_min (int): The minimum batch size to use during training. Should be multiples of 16 for optimal hardware use?? Defaults to 16.
        batch_size_max (int): The Maximum batch size to use during training. Should be multiples of 16 for optimal hardware use?? Defaults to 64.
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
        
    Returns:
        The performance metric.
    """

    def __init__(self, positive_class, negative_class, img_num_channels=1, normalize=True, min_pixel=0, max_pixel=100, 
        val_positive=None, val_negative=None, test_positive=None, test_negative=None, train_epochs=25, patience=20, 
        opt_model=True, opt_aug=False, batch_min=2, batch_max=25, image_size_min=50, image_size_max=100, 
        balance=True, opt_max_min_pix=None, opt_max_max_pix=None, metric='loss', average=True, shift=10,
        mask_size=None, num_masks=None, opt_cv=None, verbose=0, train_acc_threshold=None, 
        limit_search=True, batch_size_min=16, batch_size_max=64, monitor1=None, monitor2=None, monitor1_thresh=None, monitor2_thresh=None):

        self.positive_class = positive_class
        self.negative_class = negative_class
        self.img_num_channels = img_num_channels
        self.normalize = normalize 
        self.min_pixel = min_pixel
        self.max_pixel = max_pixel
        self.val_positive = val_positive
        self.val_negative = val_negative
        self.test_positive = test_positive
        self.test_negative = test_negative
        self.batch_size_min = batch_size_min
        self.batch_size_max = batch_size_max

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
        self.metric = metric 
        self.average = average
        self.shift = shift 
        self.opt_cv = opt_cv
        self.verbose = verbose
        self.mask_size = mask_size
        self.num_masks = num_masks
        self.train_acc_threshold = train_acc_threshold
        self.limit_search = limit_search

        self.monitor1 = monitor1
        self.monitor2 = monitor2
        self.monitor1_thresh = monitor1_thresh
        self.monitor2_thresh = monitor2_thresh

        if 'all' not in self.metric and 'loss' not in self.metric and 'f1_score' not in self.metric and 'binary_accuracy' not in self.metric:
            raise ValueError("Invalid metric input, options are: 'loss', 'binary_accuracy', 'f1_score', or 'all', and the validation equivalents (add val_ at the beginning).")
        
        if self.metric == 'val_loss' or self.metric == 'val_binary_accuracy':
            if self.val_positive is None and self.val_negative is None:
                raise ValueError('No validation data input, change the metric to either "loss" or "binary_accuracy".')

        if self.opt_max_min_pix is not None:
            if self.opt_max_max_pix is None:
                raise ValueError('To optimize min/max normalization pixel value, both opt_min_pix and opt_max_pix must be input')

        if self.opt_max_max_pix is not None:
            if self.opt_max_min_pix is None:
                raise ValueError('To optimize min/max normalization pixel value, both opt_min_pix and opt_max_pix must be input')

    def __call__(self, trial):

        if self.opt_aug:
            if self.img_num_channels == 1:
                channel1, channel2, channel3 = copy.deepcopy(self.positive_class), None, None 
            elif self.img_num_channels == 2:
                channel1, channel2, channel3 = copy.deepcopy(self.positive_class[:,:,:,0]), copy.deepcopy(self.positive_class[:,:,:,1]), None 
            elif self.img_num_channels == 3:
                channel1, channel2, channel3 = copy.deepcopy(self.positive_class[:,:,:,0]), copy.deepcopy(self.positive_class[:,:,:,1]), copy.deepcopy(self.positive_class[:,:,:,2])
            else:
                raise ValueError('Only three filters are supported!')

        if 'loss' in self.metric: #This sets the model patience during individual training runs
            mode = 'min'
        else:
            mode = 'max'

        if self.opt_aug:
            num_aug = trial.suggest_int('num_aug', self.batch_min, self.batch_max, step=1)
            image_size = trial.suggest_int('image_size', self.image_size_min, self.image_size_max, step=1)
            #rotation = trial.suggest_categorical('rotation', [True, False])
            rotation = False
            horizontal = vertical = True 

            augmented_images = augmentation(channel1=channel1, channel2=channel2, channel3=channel3, batch=num_aug, 
                width_shift=self.shift, height_shift=self.shift, horizontal=horizontal, vertical=vertical, rotation=rotation, 
                image_size=image_size, mask_size=self.mask_size, num_masks=self.num_masks)

            #The augmentation routine returns an output for each filter, e.g. 3 outputs for RGB
            if self.img_num_channels > 1:
                class_1=[]
                if self.img_num_channels == 2:
                    for i in range(len(augmented_images[0])):
                        class_1.append(data_processing.concat_channels(augmented_images[0][i], augmented_images[1][i]))
                else:
                    for i in range(len(augmented_images[0])):
                        class_1.append(data_processing.concat_channels(augmented_images[0][i], augmented_images[1][i], augmented_images[2][i]))
                class_1 = np.array(class_1)
            else:
                class_1 = augmented_images

            #Perform same augmentation techniques on negative class data for balance but only use batch=1
            #This is done so that the training data also includes the random cutout, if configured.
            if self.img_num_channels == 1:
                channel1, channel2, channel3 = copy.deepcopy(self.negative_class), None, None 
            elif self.img_num_channels == 2:
                channel1, channel2, channel3 = copy.deepcopy(self.negative_class[:,:,:,0]), copy.deepcopy(self.negative_class[:,:,:,1]), None 
            elif self.img_num_channels == 3:
                channel1, channel2, channel3 = copy.deepcopy(self.negative_class[:,:,:,0]), copy.deepcopy(self.negative_class[:,:,:,1]), copy.deepcopy(self.negative_class[:,:,:,2])
            
            augmented_images_negative = augmentation(channel1=channel1, channel2=channel2, channel3=channel3, batch=1, 
                width_shift=self.shift, height_shift=self.shift, horizontal=horizontal, vertical=vertical, rotation=rotation, 
                image_size=image_size, mask_size=self.mask_size, num_masks=self.num_masks)

            #The augmentation routine returns an output for each filter, e.g. 3 outputs for RGB
            if self.img_num_channels > 1:
                class_2=[]
                if self.img_num_channels == 2:
                    for i in range(len(augmented_images_negative[0])):
                        class_2.append(data_processing.concat_channels(augmented_images_negative[0][i], augmented_images_negative[1][i]))
                else:
                    for i in range(len(augmented_images_negative[0])):
                        class_2.append(data_processing.concat_channels(augmented_images_negative[0][i], augmented_images_negative[1][i], augmented_images_negative[2][i]))
                class_2 = np.array(class_2)
            else:
                class_2 = augmented_images_negative

            #Balance the class sizes if necessary
            if self.balance:
                class_2 = class_2[:len(class_1)]
  
            if self.img_num_channels == 1:
                class_2 = resize(class_2, size=image_size)
            else:
                channel1 = resize(class_2[:,:,:,0], size=image_size)
                channel2 = resize(class_2[:,:,:,1], size=image_size)
                if self.img_num_channels == 2:
                    class_2 = data_processing.concat_channels(channel1, channel2)
                else:
                    channel3 = resize(class_2[:,:,:,2], size=image_size)
                    class_2 = data_processing.concat_channels(channel1, channel2, channel3)

            #Need to also crop the validation images
            if self.val_positive is not None:
                if self.img_num_channels == 1:
                    val_class_1 = resize(self.val_positive, size=image_size)
                else:
                    val_channel1 = resize(self.val_positive[:,:,:,0], size=image_size)
                    val_channel2 = resize(self.val_positive[:,:,:,1], size=image_size)
                    if self.img_num_channels == 2:
                        val_class_1 = data_processing.concat_channels(val_channel1, val_channel2)
                    else:
                        val_channel3 = resize(self.val_positive[:,:,:,2], size=image_size)
                        val_class_1 = data_processing.concat_channels(val_channel1, val_channel2, val_channel3)
            else:
                val_class_1 = None 

            if self.val_negative is not None:
                if self.img_num_channels == 1:
                    val_class_2 = resize(self.val_negative, size=image_size)
                elif self.img_num_channels > 1:
                    val_channel1 = resize(self.val_negative[:,:,:,0], size=image_size)
                    val_channel2 = resize(self.val_negative[:,:,:,1], size=image_size)
                    if self.img_num_channels == 2:
                        val_class_2 = data_processing.concat_channels(val_channel1, val_channel2)
                    else:
                        val_channel3 = resize(self.val_negative[:,:,:,2], size=image_size)
                        val_class_2 = data_processing.concat_channels(val_channel1, val_channel2, val_channel3)
            else:
                val_class_2 = None 

        else:
            class_1, class_2 = self.positive_class, self.negative_class
            val_class_1, val_class_2 = self.val_positive, self.val_negative

        ### Optimize the max pixel to use for the min-max normalization -- per band ###
        if self.opt_max_min_pix is not None:
            self.normalize = True #Just in case it's set to False by the user 
            min_pix, max_pix = 0.0, []
            if self.img_num_channels >= 1:
                max_pix_1 = trial.suggest_int('max_pixel_1', self.opt_max_min_pix, self.opt_max_max_pix, step=1)
                max_pix.append(max_pix_1)
            if self.img_num_channels >= 2:
                max_pix_2 = trial.suggest_int('max_pixel_2', self.opt_max_min_pix, self.opt_max_max_pix, step=1)
                max_pix.append(max_pix_2)
            if self.img_num_channels == 3:
                max_pix_3 = trial.suggest_int('max_pixel_3', self.opt_max_min_pix, self.opt_max_max_pix, step=1)
                max_pix.append(max_pix_3)
            elif self.img_num_channels > 3:
                raise ValueError('Only up to three channels are currently supported!')
        else:
            min_pix, max_pix = self.min_pixel, self.max_pixel

        ### Early Stopping and Pruning Callbacks ###
        if self.patience != 0:
            if self.metric == 'all' or self.metric == 'val_all':
                print('Cannot use callbacks if averaging out all performance metrics for evaluation, setting patience=0.')
                callbacks = []#TFKerasPruningCallback(trial, monitor=self.metric),]
            else:
                if self.train_acc_threshold is None:
                    callbacks = [
                        EarlyStopping(monitor=self.metric, mode=mode, patience=self.patience),
                        #TFKerasPruningCallback(trial, monitor=self.metric),
                        ]
                else:
                    callbacks = [
                        EarlyStopping(monitor='binary_accuracy', mode='max', patience=0, baseline=self.train_acc_threshold),
                        EarlyStopping(monitor=self.metric, mode=mode, patience=self.patience), 
                        #TFKerasPruningCallback(trial, monitor=self.metric),
                        ]
            if self.monitor1 is not None:
                callbacks.append(Monitor_Tracker(monitor1=self.monitor1, monitor2=self.monitor2, monitor1_thresh=self.monitor1_thresh, monitor2_thresh=self.monitor2_thresh))
        else:
            callbacks = None

        ### ### ### ### ### ### ### ### ### 
        ## Optimize CNN -- AlexNet only! ##
        ### ### ### ### ### ### ### ### ### 

        ### Batch Size, Learning Rate & Optimizer ###
        batch_size = trial.suggest_int('batch_size', self.batch_size_min, self.batch_size_max, step=1)
        lr = trial.suggest_float('lr', 1e-6, 1e-3, step=1e-4) 
        optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop', 'adagrad']) 

        #Adam and Adagrad don't need decay nor momentum
        if optimizer != 'adam' and optimizer != 'adagrad':
            decay = trial.suggest_float('decay', 0.0, 0.1, step=1e-3)
            if optimizer == 'sgd': #Only SGD requires momentum and nesterov, rmsprop only needs lr & decay
                momentum = trial.suggest_float('momentum', 0.0, 1.0, step=0.01)
                nesterov = trial.suggest_categorical('nesterov', [True, False])
            else:
                momentum, nesterov = 0.0, False
        else:
            decay, momentum, nesterov = 0.0, 0.0, False

        clear_session()
        if self.opt_model:
            """AlexNet Hyperparameter Search Space"""

            ### Regularization Technique (BN is newer) ###
            model_reg = trial.suggest_categorical('model_reg', ['local_response', 'batch_norm'])

            ### Activation and Loss Functions ### 
            activation_conv = trial.suggest_categorical('activation_conv', ['relu',  'sigmoid', 'tanh', 'elu', 'selu'])            
            activation_dense = trial.suggest_categorical('activation_dense', ['relu', 'sigmoid', 'tanh', 'elu', 'selu'])
            loss = trial.suggest_categorical('loss', ['binary_crossentropy', 'hinge', 'squared_hinge', 'kld', 'logcosh'])#, 'focal_loss', 'dice_loss', 'jaccard_loss'])

            ### Kernel Initializers ###
            conv_init = trial.suggest_categorical('conv_init', ['uniform_scaling', 'TruncatedNormal', 'he_normal', 'lecun_uniform', 'glorot_uniform']) 
            dense_init = trial.suggest_categorical('dense_init', ['uniform_scaling', 'TruncatedNormal','he_normal', 'lecun_uniform', 'glorot_uniform']) 

            ### Train the Model ###
            print()
            if self.verbose == 1:
                if self.opt_cv is not None:
                    print()
                    print('***********  CV - 1 ***********')
                    print()
                if self.opt_aug:
                    print()
                    print('======= Image Parameters ======')
                    print()
                    print('Num Augmentations :', num_aug)
                    print('Image Size : ', image_size)
                    print('Max Pixel(s) :', max_pix)

            ### Pooling Type ###
            if self.limit_search:
                pooling_1 = trial.suggest_categorical('pooling_1', ['min', 'max', 'average'])
                pooling_2 = trial.suggest_categorical('pooling_2', ['min', 'max', 'average'])
                pooling_3 = trial.suggest_categorical('pooling_3', ['min', 'max', 'average'])

                model, history = cnn_model.AlexNet(class_1, class_2, img_num_channels=self.img_num_channels, 
                    normalize=self.normalize, min_pixel=min_pix, max_pixel=max_pix, val_positive=val_class_1, val_negative=val_class_2, 
                    epochs=self.train_epochs, batch_size=batch_size, lr=lr, momentum=momentum, decay=decay, nesterov=nesterov, 
                    loss=loss, activation_conv=activation_conv, activation_dense=activation_dense, model_reg=model_reg, 
                    conv_init=conv_init, dense_init=dense_init, optimizer=optimizer, early_stop_callback=callbacks, checkpoint=False, verbose=self.verbose, 
                    pooling_1=pooling_1, pooling_2=pooling_2, pooling_3=pooling_3)
            else:
                ### Filter and Layer Characterstics ###
                filter_1 = trial.suggest_int('filter_1', 12, 240, step=12)
                filter_size_1 = trial.suggest_int('filter_size_1', 1, 11, step=2)
                strides_1 = trial.suggest_int('strides_1', 1, 3, step=1)
                pooling_1 = trial.suggest_categorical('pooling_1', ['max', 'average'])
                pool_size_1 = trial.suggest_int('pool_size_1', 1, 7, step=1)
                pool_stride_1 = trial.suggest_int('pool_stride_1', 1, 3, step=1)

                filter_2 = trial.suggest_int('filter_2', 12, 240, step=12)
                filter_size_2 = trial.suggest_int('filter_size_2', 1, 7, step=2)
                strides_2 = trial.suggest_int('strides_2', 1, 3, step=1)
                pooling_2 = trial.suggest_categorical('pooling_2', ['max', 'average'])
                pool_size_2 = trial.suggest_int('pool_size_2', 1, 7, step=1)
                pool_stride_2 = trial.suggest_int('pool_stride_2', 1, 3, step=1)

                filter_3 = trial.suggest_int('filter_3', 12, 240, step=12)
                filter_size_3 = trial.suggest_int('filter_size_3', 1, 7, step=2)
                strides_3 = trial.suggest_int('strides_3', 1, 3, step=1)
                pooling_3 = trial.suggest_categorical('pooling_3', ['max', 'average'])
                pool_size_3 = trial.suggest_int('pool_size_3', 1, 7, step=1)
                pool_stride_3 = trial.suggest_int('pool_stride_3', 1, 3, step=1)

                filter_4 = trial.suggest_int('filter_4', 12, 240, step=12)
                filter_size_4 = trial.suggest_int('filter_size_4', 1, 7, step=2)
                strides_4 = trial.suggest_int('strides_4', 1, 3, step=1)

                filter_5 = trial.suggest_int('filter_5', 12, 240, step=12)
                filter_size_5 = trial.suggest_int('filter_size_5', 1, 7, step=2)
                strides_5 = trial.suggest_int('strides_5', 1, 3, step=1) 

                ### Dense Layers ###
                dense_neurons_1 = trial.suggest_int('dense_neurons_1', 128, 6400, step=128)
                dense_neurons_2 = trial.suggest_int('dense_neurons_2', 128, 6400, step=128)
                dropout_1 = trial.suggest_float('dropout_1', 0.0, 0.5, step=0.01)
                dropout_2 = trial.suggest_float('dropout_2', 0.0, 0.5, step=0.01) 

                model, history = cnn_model.AlexNet(class_1, class_2, img_num_channels=self.img_num_channels, 
                    normalize=self.normalize, min_pixel=min_pix, max_pixel=max_pix, val_positive=val_class_1, val_negative=val_class_2, 
                    epochs=self.train_epochs, batch_size=batch_size, lr=lr, momentum=momentum, decay=decay, nesterov=nesterov, 
                    loss=loss, activation_conv=activation_conv, activation_dense=activation_dense, model_reg=model_reg, 
                    conv_init=conv_init, dense_init=dense_init, optimizer=optimizer, pooling_1=pooling_1, pooling_2=pooling_2, pooling_3=pooling_3, 
                    pool_stride_1=pool_stride_1, pool_size_2=pool_size_2, pool_stride_2=pool_stride_2, 
                    pool_size_3=pool_size_3, pool_stride_3=pool_stride_3, filter_1=filter_1, filter_size_1=filter_size_1, 
                    strides_1=strides_1, filter_2=filter_2, filter_size_2=filter_size_2, strides_2=strides_2, 
                    filter_3=filter_3, filter_size_3=filter_size_3, strides_3=strides_3, filter_4=filter_4, 
                    filter_size_4=filter_size_4, strides_4=strides_4, filter_5=filter_5, filter_size_5=filter_size_5, 
                    strides_5=strides_5, dense_neurons_1=dense_neurons_1, dense_neurons_2=dense_neurons_2, 
                    dropout_1=dropout_1, dropout_2=dropout_2, early_stop_callback=callbacks, checkpoint=False, verbose=self.verbose)
        else:
            if self.opt_cv is not None:
                print()
                print('***********  CV - 1 ***********')
                print()
            if self.opt_aug:
                print()
                print('======= Image Parameters ======')
                print()
                print('Num Augmentations : ', num_aug)
                print('Image Size : ', image_size)
                print('Max Pixel(s) : ', max_pix)

            model, history = cnn_model.AlexNet(class_1, class_2, img_num_channels=self.img_num_channels, normalize=self.normalize, 
                min_pixel=min_pix, max_pixel=max_pix, val_positive=val_class_1, val_negative=val_class_2, epochs=self.train_epochs, 
                batch_size=batch_size, lr=lr, momentum=momentum, decay=decay, nesterov=nesterov, early_stop_callback=callbacks, 
                checkpoint=False, verbose=self.verbose, optimizer=optimizer)

        #If the patience is reached, return nan#
        if len(history.history['loss']) != self.train_epochs:
            print()
            print('The trial was pruned or the training patience was reached...')#, returning 0.001 times the number of completed epochs...')
            print()
            return (len(history.history['loss']) * 0.001) - 999.0

        #If the training fails return NaN right away#
        if np.isfinite(history.history['loss'][-1]):
            #If the train acc threshold is set#
            if self.train_acc_threshold is not None:
                if history.history['binary_accuracy'][-1] > self.train_acc_threshold:
                    print()
                    print('The trial was pruned or the training patience was reached...')#, returning 0.001 times the number of completed epochs...')
                    print()
                    return (len(history.history['loss']) * 0.001) - 999.0


            models, histories = [], []
            models.append(model), histories.append(history)

        else:
            print('Training failed due to numerical instability, returning nan...')
            return np.nan 

        ##### Cross-Validation Routine - very specific implementation #####
        if self.opt_cv is not None:
            if self.val_positive is None and self.val_negative is not None:
                raise ValueError('CNN cross-validation is currently supported only if validation data is input.')
            if self.val_positive is not None:
                if len(self.positive_class) / len(self.val_positive) < self.opt_cv-1:
                    raise ValueError('Cannot partition the training/validation data, refer to the API documentation for instructions on how to use the opt_cv parameter.')
            if self.val_negative is not None:
                if len(self.negative_class) / len(self.val_negative) < self.opt_cv-1:
                    raise ValueError('Cannot partition the training/validation data, refer to the API documentation for instructions on how to use the opt_cv parameter.')
            
            #The first model already ran therefore sutbract 1      
            for k in range(self.opt_cv-1):          
                #Make deep copies to avoid overwriting arrays
                class_1, class_2 = copy.deepcopy(self.positive_class), copy.deepcopy(self.negative_class)
                val_class_1, val_class_2 = copy.deepcopy(self.val_positive), copy.deepcopy(self.val_negative)

                #Sort the new data samples, no random shuffling, just a linear sequence
                if val_class_1 is not None:
                    val_hold_1 = copy.deepcopy(class_1[k*len(val_class_1):len(val_class_1)*(k+1)]) #The new validation data
                    class_1[k*len(val_class_1):len(val_class_1)*(k+1)] = copy.deepcopy(val_class_1)
                    val_class_1 = val_hold_1 #The new validation data
                if val_class_2 is not None:
                    val_hold_2 = copy.deepcopy(class_2[k*len(val_class_2):len(val_class_2)*(k+1)]) #The new validation data
                    class_2[k*len(val_class_2):len(val_class_2)*(k+1)] = copy.deepcopy(val_class_2)
                    val_class_2 = val_hold_2 #The new validation data

                if self.opt_aug:
                    if self.img_num_channels == 1:
                        channel1, channel2, channel3 = copy.deepcopy(class_1), None, None 
                    elif self.img_num_channels == 2:
                        channel1, channel2, channel3 = copy.deepcopy(class_1[:,:,:,0]), copy.deepcopy(class_1[:,:,:,1]), None 
                    else:
                        channel1, channel2, channel3 = copy.deepcopy(class_1[:,:,:,0]), copy.deepcopy(class_1[:,:,:,1]), copy.deepcopy(class_1[:,:,:,2])
            
                    augmented_images = augmentation(channel1=channel1, channel2=channel2, channel3=channel3, batch=num_aug, 
                        width_shift=self.shift, height_shift=self.shift, horizontal=horizontal, vertical=vertical, rotation=rotation, 
                        image_size=image_size, mask_size=self.mask_size, num_masks=self.num_masks)

                    if self.img_num_channels > 1:
                        class_1=[]
                        if self.img_num_channels == 2:
                            for i in range(len(augmented_images[0])):
                                class_1.append(data_processing.concat_channels(augmented_images[0][i], augmented_images[1][i]))
                        else:
                            for i in range(len(augmented_images[0])):
                                class_1.append(data_processing.concat_channels(augmented_images[0][i], augmented_images[1][i], augmented_images[2][i]))
                        class_1 = np.array(class_1)
                    else:
                        class_1 = augmented_images

                    #Perform same augmentation techniques on negative class data, batch=1
                    if self.img_num_channels == 1:
                        channel1, channel2, channel3 = copy.deepcopy(class_2), None, None 
                    elif self.img_num_channels == 2:
                        channel1, channel2, channel3 = copy.deepcopy(class_2[:,:,:,0]), copy.deepcopy(class_2[:,:,:,1]), None 
                    elif self.img_num_channels == 3:
                        channel1, channel2, channel3 = copy.deepcopy(class_2[:,:,:,0]), copy.deepcopy(class_2[:,:,:,1]), copy.deepcopy(class_2[:,:,:,2])
                    
                    augmented_images_negative = augmentation(channel1=channel1, channel2=channel2, channel3=channel3, batch=1, 
                        width_shift=self.shift, height_shift=self.shift, horizontal=horizontal, vertical=vertical, rotation=rotation, 
                        image_size=image_size, mask_size=self.mask_size, num_masks=self.num_masks)

                    #The augmentation routine returns an output for each filter, e.g. 3 outputs for RGB
                    if self.img_num_channels > 1:
                        class_2=[]
                        if self.img_num_channels == 2:
                            for i in range(len(augmented_images_negative[0])):
                                class_2.append(data_processing.concat_channels(augmented_images_negative[0][i], augmented_images_negative[1][i]))
                        else:
                            for i in range(len(augmented_images_negative[0])):
                                class_2.append(data_processing.concat_channels(augmented_images_negative[0][i], augmented_images_negative[1][i], augmented_images_negative[2][i]))
                        class_2 = np.array(class_2)
                    else:
                        class_2 = augmented_images_negative

                    #Balance the class sizes
                    if self.balance:
                        class_2 = class_2[:len(class_1)]     

                    if self.img_num_channels == 1:
                        class_2 = resize(class_2, size=image_size)
                    else:
                        channel1 = resize(class_2[:,:,:,0], size=image_size)
                        channel2 = resize(class_2[:,:,:,1], size=image_size)
                        if self.img_num_channels == 2:
                            class_2 = data_processing.concat_channels(channel1, channel2)
                        else:
                            channel3 = resize(class_2[:,:,:,2], size=image_size)
                            class_2 = data_processing.concat_channels(channel1, channel2, channel3)

                    if val_class_1 is not None:
                        if self.img_num_channels == 1:
                            val_class_1 = resize(val_class_1, size=image_size)
                        else:
                            val_channel1 = resize(val_class_1[:,:,:,0], size=image_size)
                            val_channel2 = resize(val_class_1[:,:,:,1], size=image_size)
                            if self.img_num_channels == 2:
                                val_class_1 = data_processing.concat_channels(val_channel1, val_channel2)
                            else:
                                val_channel3 = resize(val_class_1[:,:,:,2], size=image_size)
                                val_class_1 = data_processing.concat_channels(val_channel1, val_channel2, val_channel3)

                    if val_class_2 is not None:
                        if self.img_num_channels == 1:
                            val_class_2 = resize(val_class_2, size=image_size)
                        elif self.img_num_channels > 1:
                            val_channel1 = resize(val_class_2[:,:,:,0], size=image_size)
                            val_channel2 = resize(val_class_2[:,:,:,1], size=image_size)
                            if self.img_num_channels == 2:
                                val_class_2 = data_processing.concat_channels(val_channel1, val_channel2)
                            else:
                                val_channel3 = resize(val_class_2[:,:,:,2], size=image_size)
                                val_class_2 = data_processing.concat_channels(val_channel1, val_channel2, val_channel3)

                if self.verbose == 1:
                    print()
                    print('***********  CV - {} ***********'.format(k+2))
                    print()

                clear_session()
                if self.opt_model is False:
                    model, history = cnn_model.AlexNet(class_1, class_2, img_num_channels=self.img_num_channels, normalize=self.normalize, 
                        min_pixel=min_pix, max_pixel=max_pix, val_positive=val_class_1, val_negative=val_class_2, epochs=self.train_epochs, 
                        batch_size=batch_size, lr=lr, momentum=momentum, decay=decay, nesterov=nesterov, early_stop_callback=callbacks, 
                        conv_init=conv_init, dense_init=dense_init, checkpoint=False, verbose=self.verbose, optimizer=optimizer)
                else:
                    if self.limit_search:
                        model, history = cnn_model.AlexNet(class_1, class_2, img_num_channels=self.img_num_channels, 
                            normalize=self.normalize, min_pixel=min_pix, max_pixel=max_pix, val_positive=val_class_1, val_negative=val_class_2, 
                            epochs=self.train_epochs, batch_size=batch_size, lr=lr, momentum=momentum, decay=decay, nesterov=nesterov, 
                            loss=loss, activation_conv=activation_conv, activation_dense=activation_dense, model_reg=model_reg, 
                            conv_init=conv_init, dense_init=dense_init, optimizer=optimizer, early_stop_callback=callbacks, checkpoint=False, verbose=self.verbose, 
                            pooling_1=pooling_1, pooling_2=pooling_2, pooling_3=pooling_3)
                    else:
                        model, history = cnn_model.AlexNet(class_1, class_2, img_num_channels=self.img_num_channels, 
                            normalize=self.normalize, min_pixel=min_pix, max_pixel=max_pix, val_positive=val_class_1, val_negative=val_class_2, 
                            epochs=self.train_epochs, batch_size=batch_size, lr=lr, momentum=momentum, decay=decay, nesterov=nesterov, 
                            loss=loss, activation_conv=activation_conv, activation_dense=activation_dense, model_reg=model_reg, 
                            conv_init=conv_init, dense_init=dense_init, optimizer=optimizer, pooling_1=pooling_1, pooling_2=pooling_2, pooling_3=pooling_3, 
                            pool_stride_1=pool_stride_1, pool_size_2=pool_size_2, pool_stride_2=pool_stride_2, 
                            pool_size_3=pool_size_3, pool_stride_3=pool_stride_3, filter_1=filter_1, filter_size_1=filter_size_1, 
                            strides_1=strides_1, filter_2=filter_2, filter_size_2=filter_size_2, strides_2=strides_2, 
                            filter_3=filter_3, filter_size_3=filter_size_3, strides_3=strides_3, filter_4=filter_4, 
                            filter_size_4=filter_size_4, strides_4=strides_4, filter_5=filter_5, filter_size_5=filter_size_5, 
                            strides_5=strides_5, dense_neurons_1=dense_neurons_1, dense_neurons_2=dense_neurons_2, 
                            dropout_1=dropout_1, dropout_2=dropout_2, early_stop_callback=callbacks, checkpoint=False, verbose=self.verbose)

                #If the trial is pruned or the patience is reached, return nan#
                if len(history.history['loss']) != self.train_epochs:
                    print()
                    print('The trial was pruned or the training patience was reached...')#, returning 0.001 times the number of completed epochs...')
                    print()
                    return (len(history.history['loss']) * 0.001) - 999.0

                #If the training fails return NaN right away#
                if np.isfinite(history.history['loss'][-1]):
                    #If the train acc threshold is set#
                    if self.train_acc_threshold is not None:
                        if history.history['binary_accuracy'][-1] > self.train_acc_threshold:
                            print()
                            print('The trial was pruned or the training patience was reached...')#, returning 0.001 times the number of completed epochs...')
                            print()
                            return (len(history.history['loss']) * 0.001) - 999.0
                    models.append(model), histories.append(history)
                else:
                    print('Training failed due to numerical instability, returning nan...')
                    return np.nan 

        ###### Additional test data metric ######
        if self.test_positive is not None or self.test_negative is not None:
            if self.test_positive is not None and self.test_negative is not None:
                positive_test_crop, negative_test_crop = resize(self.test_positive, size=image_size), resize(self.test_negative, size=image_size)
                X_test, Y_test = create_training_set(positive_test_crop, negative_test_crop, normalize=self.normalize, min_pixel=min_pix, max_pixel=max_pix, img_num_channels=self.img_num_channels)
            elif self.test_positive is not None:
                positive_test_crop = resize(self.test_positive, size=image_size)
                positive_class_data, positive_class_label = data_processing.process_class(positive_test_crop, label=1, normalize=self.normalize, min_pixel=min_pix, max_pixel=max_pix, img_num_channels=self.img_num_channels)
                if self.test_negative is not None:
                    negative_test_crop = resize(self.test_negative, size=image_size)
                    negative_class_data, negative_class_label = data_processing.process_class(negative_test_crop, label=0, normalize=self.normalize, min_pixel=min_pix, max_pixel=max_pix, img_num_channels=self.img_num_channels)
                    X_test, Y_test = np.r_[positive_class_data, negative_class_data], np.r_[positive_class_label, negative_class_label]
                else:
                    X_test, Y_test = positive_class_data, positive_class_label
            else:
                if self.test_negative is not None:
                    X_test, Y_test = data_processing.process_class(negative_test_crop, label=0, normalize=self.normalize, min_pixel=min_pix, max_pixel=max_pix, img_num_channels=self.img_num_channels)
             
            ###Loop through all the models###
            test = []
            for _model_ in models:
                test_loss, test_acc, test_f1_score = _model_.evaluate(X_test, Y_test, batch_size=len(X_test))
                if 'loss' in self.metric:
                    test_metric = 1 - test_loss
                elif 'acc' in self.metric:
                    test_metric = test_acc 
                elif 'f1' in self.metric:
                    test_metric = test_f1_score
                elif 'all' in self.metric:
                    test_metric = np.mean([1 - test_loss, test_acc, test_f1_score])
                test.append(test_metric)

            test_metric = np.mean(test)
            if self.verbose == 1:
                print('Post-Trial Metric: '+str(test_metric))
        else:
            test_metric = None

        ### Calculate the final optimization metric ###
        metrics = ['loss', 'binary_accuracy', 'f1_score']
        if self.metric == 'all': #Average all the training metrics
            training_metrics_mean, training_loss_mean = [], []
            if self.average:
                for _history_ in histories:
                    training_metrics_mean.append(np.mean([np.mean(_history_.history[metric]) for metric in metrics if 'loss' not in metric]))
                    training_loss_mean.append(1 - np.mean(_history_.history['loss']))
            else:
                for _history_ in histories:
                    training_metrics_mean.append(np.mean([_history_.history[metric][-1] for metric in metrics if 'loss' not in metric]))
                    training_loss_mean.append(1 - _history_.history['loss'][-1])
            if test_metric is None:
                final_score = np.mean([training_metrics_mean, training_loss_mean])
            else:
                final_score = np.mean([training_metrics_mean, training_loss_mean, test_metric])
        elif self.metric == 'val_all': #Average all the validation metrics
            val_metrics_mean, val_loss_mean = [], [] 
            if self.average:
                for _history_ in histories:
                    val_metrics_mean.append(np.mean([np.mean(_history_.history['val_'+metric]) for metric in metrics if 'loss' not in metric]))
                    val_loss_mean.append(1 - np.mean(_history_.history['val_loss']))
            else:
                for _history_ in histories:
                    val_metrics_mean.append(np.mean([_history_.history['val_'+metric][-1] for metric in metrics if 'loss' not in metric]))
                    val_loss_mean.append(1 - _history_.history['val_loss'][-1])
            if test_metric is None:
                final_score = np.mean([val_metrics_mean, val_loss_mean])
            else:
                final_score = np.mean([val_metrics_mean, val_loss_mean, test_metric])
        else:
            scores = []
            if self.average:
                for _history_ in histories:
                    scores.append(np.mean(_history_.history[self.metric]))
            else:
                for _history_ in histories:
                    scores.append(_history_.history[self.metric][-1])

            final_score = np.mean(scores)

            if 'loss' in self.metric: 
                final_score = 1 - final_score
            if test_metric is not None:
                final_score = (final_score + test_metric) / 2.0
        
        return final_score

class objective_xgb(object):
    """
    Optimization objective function for the tree-based XGBoost classifier. 
    The Optuna software for hyperparameter optimization was published in 
    2019 by Akiba et al. Paper: https://arxiv.org/abs/1907.10902

    test_size is an optional parameter to speed up the XGB optimization training.
    If input, a random validation data will be generated according to this test size,
    which will replace the cross-validation method used by default during. 
    Need more testing to make this more robust, recommended option is test_size=None. 
    The opt_cv parameter should be used instead to set the number of folds to use 
    when assessing optimization trial performance. 

    Args:
        data_x (ndarray): 2D array of size (n x m), where n is the
            number of samples, and m the number of features.
        data_y (ndarray, str): 1D array containing the corresponing labels. 
        limit_search (bool): If True, the search space for the parameters will be expanded,
            as there are some hyperparameters that can range from 0 to inf. Defaults to False.
        opt_cv (int): Cross-validations to perform when assesing the performance at each
            hyperparameter optimization trial. For example, if cv=3, then each optimization trial
            will be assessed according to the 3-fold cross validation accuracy. 
        test_size (float, optional): The size of the validation data, will be chosen
            randomly each trial. Must be between 0 and 1. Defaults to None, in which
            case the opt_cv parameter will be used instead.

    Returns:
        The performance metric.
    """

    def __init__(self, data_x, data_y, limit_search=False, opt_cv=3, test_size=None):
        self.data_x = data_x
        self.data_y = data_y
        self.limit_search = limit_search
        self.opt_cv = opt_cv 
        self.test_size = test_size 

    def __call__(self, trial):

        params = {"objective": "binary:logistic", "eval_metric": "auc"}
    
        if self.test_size is not None:
            train_x, valid_x, train_y, valid_y = train_test_split(self.data_x, self.data_y, test_size=self.test_size, random_state=np.random.randint(1, 1e9))
            dtrain, dvalid = DMatrix(train_x, label=train_y), DMatrix(valid_x, label=valid_y)
            #print('Initializing XGBoost Pruner...')
            pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-auc")

        if self.limit_search:
            params['n_estimators'] = trial.suggest_int('n_estimators', 100, 250)
            params['booster'] = trial.suggest_categorical('booster', ['gbtree', 'dart'])
            params['reg_lambda'] = trial.suggest_loguniform('reg_lambda', 1e-8, 1)
            params['reg_alpha'] = trial.suggest_loguniform('reg_alpha', 1e-8, 1)
            params['max_depth'] = trial.suggest_int('max_depth', 2, 25)
            params['eta'] = trial.suggest_loguniform('eta', 1e-8, 1)
            params['gamma'] = trial.suggest_loguniform('gamma', 1e-8, 1)
            params['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])

            if params['booster'] == "dart":
                params['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
                params['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
                params['rate_drop'] = trial.suggest_loguniform('rate_drop', 1e-8, 1)
                params['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1)
                if self.test_size is None:
                    clf = XGBClassifier(booster=params['booster'], n_estimators=params['n_estimators'], reg_lambda=params['reg_lambda'], 
                        reg_alpha=params['reg_alpha'], max_depth=params['max_depth'], eta=params['eta'], gamma=params['gamma'], 
                        grow_policy=params['grow_policy'], sample_type=params['sample_type'], normalize_type=params['normalize_type'],
                        rate_drop=params['rate_drop'], skip_drop=params['skip_drop'], random_state=1909)#, tree_method='hist')
            
            elif params['booster'] == 'gbtree':
                params['subsample'] = trial.suggest_loguniform('subsample', 1e-6, 1.0)
                if self.test_size is None:
                    clf = XGBClassifier(booster=params['booster'], n_estimators=params['n_estimators'], reg_lambda=params['reg_lambda'], 
                        reg_alpha=params['reg_alpha'], max_depth=params['max_depth'], eta=params['eta'], gamma=params['gamma'], 
                        grow_policy=params['grow_policy'], subsample=params['subsample'], random_state=1909)#, tree_method='hist')

            if self.test_size is not None:
                bst = train(params, dtrain, evals=[(dvalid, "validation")], callbacks=[pruning_callback])
                preds = bst.predict(dvalid)
                pred_labels = np.rint(preds)
                accuracy = accuracy_score(valid_y, pred_labels)
            else:
                #FROM SKLEARN DOCUMENTATION: For int/None inputs, if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used. In all other cases, Fold is used.
                cv = cross_validate(clf, self.data_x, self.data_y, cv=self.opt_cv) 
                accuracy = np.mean(cv['test_score'])

            return accuracy

        params['booster'] = trial.suggest_categorical('booster', ['gbtree', 'dart'])
        params['n_estimators'] = trial.suggest_int('n_estimators', 100, 500)
        params['reg_lambda'] = trial.suggest_float('reg_lambda', 0, 100)
        params['reg_alpha'] = trial.suggest_int('reg_alpha', 0, 100)
        params['max_depth'] = trial.suggest_int('max_depth', 2, 25)
        params['eta'] = trial.suggest_float('eta', 1e-8, 1)
        params['gamma'] = trial.suggest_int('gamma', 1, 100)
        params['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
        params['min_child_weight'] = trial.suggest_int('min_child_weight', 1, 100)
        params['max_delta_step'] = trial.suggest_int('max_delta_step', 1, 100)
        params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
        params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.5, 1)

        if params['booster'] == "dart":
            params['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
            params['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
            params['rate_drop'] = trial.suggest_float('rate_drop', 1e-8, 1)
            params['skip_drop'] = trial.suggest_float('skip_drop', 1e-8, 1)
            if self.test_size is None:
                clf = XGBClassifier(booster=params['booster'], n_estimators=params['n_estimators'], colsample_bytree=params['colsample_bytree'], 
                    reg_lambda=params['reg_lambda'], reg_alpha=params['reg_alpha'], max_depth=params['max_depth'], eta=params['eta'], 
                    gamma=params['gamma'], grow_policy=params['grow_policy'], min_child_weight=params['min_child_weight'], 
                    max_delta_step=params['max_delta_step'], subsample=params['subsample'], sample_type=params['sample_type'], 
                    normalize_type=params['normalize_type'], rate_drop=params['rate_drop'], skip_drop=params['skip_drop'], random_state=1909)#, tree_method='hist')
        elif params['booster'] == 'gbtree':
            if self.test_size is None:
                clf = XGBClassifier(booster=params['booster'], n_estimators=params['n_estimators'], colsample_bytree=params['colsample_bytree'],  reg_lambda=params['reg_lambda'], 
                    reg_alpha=params['reg_alpha'], max_depth=params['max_depth'], eta=params['eta'], gamma=params['gamma'], grow_policy=params['grow_policy'], 
                    min_child_weight=params['min_child_weight'], max_delta_step=params['max_delta_step'], subsample=params['subsample'], random_state=1909)#, tree_method='hist')
            
        if self.test_size is not None:
            bst = train(params, dtrain, evals=[(dvalid, "validation")], callbacks=[pruning_callback])
            preds = bst.predict(dvalid)
            pred_labels = np.rint(preds)
            accuracy = accuracy_score(valid_y, pred_labels)
        else:
            cv = cross_validate(clf, self.data_x, self.data_y, cv=self.opt_cv)
            accuracy = np.mean(cv['test_score'])
        
        return accuracy

class objective_nn(object):
    """
    Optimization objective function for the scikit-learn implementatin of the
    MLP classifier. The Optuna software for hyperparameter optimization
    was published in 2019 by Akiba et al. Paper: https://arxiv.org/abs/1907.10902.

    The total number of hidden layers to test is limited to 10, with 100-5000 possible 
    number of neurons in each.

    Args:
        data_x (ndarray): 2D array of size (n x m), where n is the
            number of samples, and m the number of features.
        data_y (ndarray, str): 1D array containing the corresponing labels. 
        opt_cv (int): Cross-validations to perform when assesing the performance at each
            hyperparameter optimization trial. For example, if cv=3, then each optimization trial
            will be assessed according to the 3-fold cross validation accuracy. 

    Returns:
        The performance metric, determined using the cross-fold validation method.
    """

    def __init__(self, data_x, data_y, opt_cv):
        self.data_x = data_x
        self.data_y = data_y
        self.opt_cv = opt_cv

    def __call__(self, trial):
        learning_rate_init = trial.suggest_float('learning_rate_init', 1e-5, 0.1, step=1e-5)
        solver = trial.suggest_categorical("solver", ["sgd", "adam"]) #"lbfgs"
        activation = trial.suggest_categorical("activation", ["logistic", "tanh", "relu"])
        learning_rate = trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"])
        alpha = trial.suggest_float("alpha", 1e-6, 1, step=1e-5)
        batch_size = trial.suggest_int('batch_size', 1, 1000)
        
        n_layers = trial.suggest_int('hidden_layer_sizes', 1, 10)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f'n_units_{i}', 100, 5000))

        try:
            clf = MLPClassifier(hidden_layer_sizes=tuple(layers),learning_rate_init=learning_rate_init, 
                solver=solver, activation=activation, alpha=alpha, batch_size=batch_size, max_iter=2500, random_state=1909)
        except:
            print("Invalid hyperparameter combination, skipping trial")
            return 0.0

        cv = cross_validate(clf, self.data_x, self.data_y, cv=self.opt_cv)
        final_score = np.mean(cv['test_score'])

        return final_score

class objective_rf(object):
    """
    Optimization objective function for the scikit-learn implementatin of the
    Random Forest classifier. The Optuna software for hyperparameter optimization
    was published in 2019 by Akiba et al. Paper: https://arxiv.org/abs/1907.10902

    Args:
        data_x (ndarray): 2D array of size (n x m), where n is the
            number of samples, and m the number of features.
        data_y (ndarray, str): 1D array containing the corresponing labels. 
        opt_cv (int): Cross-validations to perform when assesing the performance at each
            hyperparameter optimization trial. For example, if cv=3, then each optimization trial
            will be assessed according to the 3-fold cross validation accuracy. 

    Returns:
        The performance metric, determined using the cross-fold validation method.
    """
    
    def __init__(self, data_x, data_y, opt_cv):

        self.data_x = data_x
        self.data_y = data_y
        self.opt_cv = opt_cv

    def __call__(self, trial):

        n_estimators = trial.suggest_int('n_estimators', 100, 3000)
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
        max_depth = trial.suggest_int('max_depth', 2, 25)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 25)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 15)
        max_features = trial.suggest_int('max_features', 1, self.data_x.shape[1])
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
        
        try:
            clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                max_features=max_features, bootstrap=bootstrap, random_state=1909)
        except:
            print("Invalid hyperparameter combination, skipping trial")
            return 0.0

        cv = cross_validate(clf, self.data_x, self.data_y, cv=self.opt_cv)
        final_score = np.mean(cv['test_score'])

        return final_score


class Monitor_Tracker(Callback):
    """
    Custom callback that allows for two different monitors.
    This class is used by the `objective_cnn` routine as it inherits the `monitor1` 
    and `monitor2` attributes, as well as their corresponding thresholds. This early stopping
    custom callback will terminate the training early if either monitors cross their respective
    threshold, therefore the patience input is not used. Whether the threshold should be a ceiling
    or floor value will be determined according to the metric, by checking whether the monitor name
    contains the word 'loss'.

    Args:
        monitor1 (str): Name of the metric to monitor. If `None`, this monitor will not be used.
        monitor2 (str): Name of the metric to monitor. If `None`, this monitor will not be used.
        monitor1_thresh (float): Threshold value for `monitor1`.
        monitor2_thresh (float): Threshold value for `monitor2`.

    Attributes:
        stopped_epoch (int): The epoch at which the training was stopped.
        best_metric (float): The best metric value achieved during the training.
        best_weights (ndarray): The weights of the model at the epoch with the best metric value.

    Returns:
        None
    """
    
    def __init__(self, monitor1=None, monitor2=None, monitor1_thresh=None, monitor2_thresh=None):
        super().__init__() ##Initializes the inherited attributes and methods from the parent class as it will inherit from Callback
        self.monitor1 = monitor1
        self.monitor2 = monitor2
        self.monitor1_thresh = monitor1_thresh
        self.monitor2_thresh = monitor2_thresh
        self.stopped_epoch = 0
        self.best_metric = float('inf') 
        self.best_weights = None #Will use this to store the weights with the best metric value

    def on_epoch_end(self, epoch, logs={}):
        """
        This method will be called at the end of each epoch during training.

        Args:
            epoch (int): Current epoch number.
            logs (dict): Dictionary containing the metrics for the current epoch.

        Returns:
            None
        """

        if self.monitor1 is not None and self.monitor2 is not None:
            current_1 = logs.get(self.monitor1)
            current_2 = logs.get(self.monitor2)
            if 'loss' not in self.monitor1 and 'loss' in self.monitor2:
                if np.greater(current_1, self.monitor1_thresh) or np.less_equal(current_2, self.monitor2_thresh):
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                current_metric = current_1
            elif 'loss' not in self.monitor1 and 'loss' not in self.monitor2:
                if np.greater(current_1, self.monitor1_thresh) or np.greater(current_2, self.monitor2_thresh):
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                current_metric = current_1
            elif 'loss' in self.monitor1 and 'loss' in self.monitor2:
                if np.less_equal(current_1, self.monitor1_thresh) or np.less_equal(current_2, self.monitor2_thresh):
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                current_metric = current_1
            elif 'loss' in self.monitor1 and 'loss' not in self.monitor2:
                if np.less_equal(current_1, self.monitor1_thresh) or np.greater(current_2, self.monitor2_thresh):
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                current_metric = current_1
            else:
                raise ValueError('Invalid monitor input.')
            
            if current_metric < self.best_metric:
                self.best_metric = current_metric
                self.best_weights = self.model.get_weights()
                
        else:
            current_metric = logs.get(self.monitor1)
            if 'loss' not in self.monitor1:
                if np.greater(current_metric, self.monitor1_thresh):
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                if current_metric < self.best_metric:
                    self.best_metric = current_metric
                    self.best_weights = self.model.get_weights()
            else:
                if np.less_equal(current_metric, self.monitor1_thresh):
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                if current_metric < self.best_metric:
                    self.best_metric = current_metric
                    self.best_weights = self.model.get_weights()


def hyper_opt(data_x, data_y, clf='rf', n_iter=25, return_study=True, balance=True, img_num_channels=1, 
    normalize=True, min_pixel=0, max_pixel=100, val_X=None, val_Y=None, train_epochs=25, patience=0, metric='loss', 
    limit_search=True, opt_model=True, opt_aug=False, batch_min=2, batch_max=25, image_size_min=50, image_size_max=100,
    opt_max_min_pix=None, opt_max_max_pix=None, opt_cv=10, test_size=None, average=True, test_positive=None, test_negative=None, shift=10, 
    mask_size=None, num_masks=None, verbose=1, train_acc_threshold=None, monitor1=None, monitor2=None, monitor1_thresh=None, monitor2_thresh=None):
    """
    Optimizes hyperparameters using a k-fold cross validation splitting strategy, unless a CNN
    is being optimized, in which case no cross-validation is performed during trial assesment.

    **IMPORTANT** In the case of CNN optimization, data_x and data_y are not the standard
    data plus labels -- if optimizing a CNN the samples for the first class should be passed
    through the data_x parameter, and the samples for the second class should be given as data_y.
    These two classes will automatically be assigned the positive and negative labels 1 and 0, respectively.
    Likewise, if optimizing a CNN model, val_X corresponds to the images of the first class, and val_Y the images of the second class. 
    
    If save_study=True, the Optuna study object will be the third output. This
    object can be used for various analysis, including optimization visualizations.
    See: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html

    Example:
        The function will create the classification engine and optimize the hyperparameters
        using an iterative approach:

        >>> model, params = hyper_opt(data_x, data_y, clf='rf') 
        
        The first output is our optimal classifier, and will be used to make predictions:
        
        >>> prediction = model.predict(new_data)
        
        The second output of the optimize function is the dictionary containing
        the hyperparameter combination that yielded the highest mean accuracy.

        If save_study = True, the Optuna study object will also be returned as the third output.
        This can be used to plot the optimization results, see: https://optuna.readthedocs.io/en/latest/tutorial/10_key_features/005_visualization.html#sphx-glr-tutorial-10-key-features-005-visualization-py

        >>> from optuna.visualization.matplotlib import plot_contour
        >>> 
        >>> model, params, study = hyper_opt(data_x, data_y, clf='rf', save_study=True) 
        >>> plot_contour(study)

        If clf='cnn', the model is not returned.
        
    Args:
        data_x (ndarray): 2D array of size (n x m), where n is the
            number of samples, and m the number of features. If clf='cnn', 
            the samples for the first class should be passed, which will automatically 
            be assigned the positive label '1'.
        data_y (ndarray, str): 1D array containing the corresponing labels. If clf='cnn',
            the samples for the second class should be passed, which will automatically
            be assigned the negative label '0'.
        clf (str): The machine learning classifier to optimize. Can either be
            'rf' for Random Forest, 'nn' for Neural Network, 'xgb' for eXtreme Gradient Boosting,
            or 'cnn' for Convolutional Neural Network. Defaults to 'rf'.
        n_iter (int, optional): The maximum number of iterations to perform during 
            the hyperparameter search. Defaults to 25.
        opt_cv (int): Cross-validations to perform when assesing the performance at each
            hyperparameter optimization trial. For example, if cv=3, then each optimization trial
            will be assessed according to the 3-fold cross validation accuracy. Defaults to 10.
            NOTE: The higher this number, the longer the optimization will take.
        test_size (float, optional): Can use this in lieu of opt_cv, only if clf='xgb'!
            Must be between 0 and 1 as this sets the size of the validation data, which will be chosen randomly each trial. 
            Defaults to None, in which case the opt_cv parameter will be used. This is used 
            to enable an early stopping callback which is not possible with the cross-validation method.
        limit_search (bool): If True the optimization search spaces will be limited, for computational and time purposes. 
            This is especially important if clf='cnn' as memory errors can be encountered. Defaults to True.
        balance (bool, optional): If True, a weights array will be calculated and used
            when fitting the classifier. This can improve classification when classes
            are imbalanced. This is only applied if the classification is a binary task. 
            Defaults to True. If clf='cnn', this will determine whether the two classes
            are kept the same size during optimization, applicable if tuning the augmentation
            parameters. Defaults to True.
        return_study (bool, optional): If True the Optuna study object will be returned. This
            can be used to review the method attributes, such as optimization plots. Defaults to True.

    The folllowing arguments are only used when optimizing the CNN network:

    Args:
        img_num_channels (int): The number of filters. Defaults to 1.
        normalize (bool, optional): If True the data will be min-max normalized using the 
            input min and max pixels. Defaults to True.
        min_pixel (int, optional): The minimum pixel count, pixels with counts 
            below this threshold will be set to this limit. Defaults to 0.
        max_pixel (int, list, optional): The maximum pixel count, pixels with counts 
            above this threshold will be set to this limit. Defaults to 100. If img_num_channels
            is not 1, the max_pixel should be a list containing two values, one for each band.
        val_positive (ndarray, optional): Positive class data to be used for validation. Defaults to None.
        val_negative (ndarray, optional): Negative class data to be used for validation. Defaults to None.
        test_positive (ndarray, optional): Positive class data to be used for post-trial testing. Defaults to None.
        test_negative (ndarray, optional): Negative class data to be used for post-trial testing. Defaults to None.
        train_epochs (int): Number of epochs to the train the CNN to during the optimization trials. Defaults to 25.
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
            regularizer. Only applicable if opt_aug=True. Defaults to None.
        num_masks (int, optional): The number of masks to create, to be used alongside the mask_size parameter. If 
            this is set to a value greater than one, overlap may occur. 
        verbose (int): Controls the amount of output printed during the training process. A value of 0 is for silent mode, 
            a value of 1 is used for progress bar mode, and 2 for one line per epoch mode. Defaults to 1.

    The folllowing arguments can be used to set early-stopping callbacks. These can be used to terminate trials that exceed
    pre-determined thresholds, which may be indicative of an overfit model.

    Args:
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
        
    Returns:
        The first output is the classifier with the optimal hyperparameters.
        Second output is a dictionary containing the optimal hyperparameters.
        If save_study=True, the Optuna study object will be the third output.
    """

    if clf == 'rf':
        model_0 = RandomForestClassifier(random_state=1909)
    elif clf == 'nn':
        model_0 = MLPClassifier(random_state=1909)
    elif clf == 'xgb':
        model_0 = XGBClassifier(random_state=1909)
        if all(isinstance(val, (int, str)) for val in data_y):
            print('XGBoost classifier requires numerical class labels! Converting class labels as follows:')
            print('____________________________________')
            y = np.zeros(len(data_y))
            for i in range(len(np.unique(data_y))):
                print(str(np.unique(data_y)[i]).ljust(10)+'  ------------->     '+str(i))
                index = np.where(data_y == np.unique(data_y)[i])[0]
                y[index] = i
            data_y = y 
            print('------------------------------------')
    elif clf == 'cnn':
        pass 
    else:
        raise ValueError('clf argument must either be "rf", "xgb", "nn", or "cnn".')

    if clf != 'cnn':
        cv = cross_validate(model_0, data_x, data_y, cv=opt_cv)
        initial_score = np.mean(cv['test_score'])

    sampler = optuna.samplers.TPESampler(seed=1909)
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10))
    print('Starting hyperparameter optimization, this will take a while...')
    #If binary classification task, can deal with imbalance classes with weights hyperparameter
    if len(np.unique(data_y)) == 2 and clf != 'cnn':
        counter = Counter(data_y)
        if counter[np.unique(data_y)[0]] != counter[np.unique(data_y)[1]]:
            if balance:
                print('Unbalanced dataset detected, will train classifier with weights! To disable, set balance=False')
                if clf == 'xgb':
                    total_negative = len(np.where(data_y == counter.most_common(1)[0][0])[0])
                    total_positive = len(data_y) - total_negative
                    sample_weight = total_negative / total_positive
                elif clf == 'rf':
                    sample_weight = 'auto'
                elif clf == 'nn':
                    print('WARNING: Unbalanced dataset detected but MLPClassifier() does not support sample weights.')
            else:
                sample_weight = None
        else:
            sample_weight = None
    else:
        sample_weight = None

    if clf == 'rf':
        try:
            objective = objective_rf(data_x, data_y, opt_cv=opt_cv)
            study.optimize(objective, n_trials=n_iter, show_progress_bar=True)#, gc_after_trial=True)
            params = study.best_trial.params
            model = RandomForestClassifier(n_estimators=params['n_estimators'], criterion=params['criterion'], 
                max_depth=params['max_depth'], min_samples_split=params['min_samples_split'], 
                min_samples_leaf=params['min_samples_leaf'], max_features=params['max_features'], 
                bootstrap=params['bootstrap'], class_weight=sample_weight, random_state=1909)
        except:
            print('Failed to optimize with Optuna, switching over to BayesSearchCV...')
            params = {
                'criterion': ["gini", "entropy"],
                'n_estimators': [int(x) for x in np.linspace(50,1000, num=20)], 
                'max_features': [data_x.shape[1], "sqrt", "log2"],
                'max_depth': [int(x) for x in np.linspace(5,50,num=5)],
                'min_samples_split': [3,4,6,7,8,9,10],
                'min_samples_leaf': [1,3,5,7,9,10],
                'max_leaf_nodes': [int(x) for x in np.linspace(2,200)],
                'bootstrap': [True,False]   
            }
            gs = BayesSearchCV(n_iter=n_iter, estimator=model_0, search_spaces=params, 
                optimizer_kwargs={'base_estimator': 'RF'}, cv=opt_cv)
            gs.fit(data_x, data_y)
            best_est, best_score = gs.best_estimator_, np.round(gs.best_score_, 4)
            print('Highest mean accuracy: {}'.format(best_score))
            return gs.best_estimator_, gs.best_params_

    elif clf == 'nn':
        try:
            objective = objective_nn(data_x, data_y, opt_cv=opt_cv)
            study.optimize(objective, n_trials=n_iter, show_progress_bar=True)#, gc_after_trial=True)
            params = study.best_trial.params
            layers = [param for param in params if 'n_units_' in param]
            layers = tuple(params[layer] for layer in layers)
            model = MLPClassifier(hidden_layer_sizes=tuple(layers), learning_rate_init=params['learning_rate_init'], 
                activation=params['activation'], learning_rate=params['learning_rate'], alpha=params['alpha'], 
                batch_size=params['batch_size'], solver=params['solver'], max_iter=2500, random_state=1909)
        except:
            print('Failed to optimize with Optuna, switching over to BayesSearchCV...')
            params = {
                'hidden_layer_sizes': [(100,),(50,100,50),(75,50,20),(150,100,50),(120,80,40),(100,50,30)],
                'learning_rate': ['constant', 'invscaling', 'adaptive'],
                'alpha': [0.00001, 0.5], 
                'activation': ['tanh', 'logistic', 'relu'],
                'solver': ['sgd', 'adam'],
                'max_iter': [100, 150, 200] 
            }
            gs = BayesSearchCV(n_iter=n_iter, estimator=model_0, search_spaces=params, cv=opt_cv)
            gs.fit(data_x, data_y)
            best_est, best_score = gs.best_estimator_, np.round(gs.best_score_, 4)
            print('Highest mean accuracy: {}'.format(best_score))
            return gs.best_estimator_, gs.best_params_
          
    elif clf == 'xgb':
        objective = objective_xgb(data_x, data_y, limit_search=limit_search, opt_cv=opt_cv, test_size=test_size)
        if limit_search:
            print('NOTE: To expand hyperparameter search space, set limit_search=False, although this will increase the optimization time significantly.')
        study.optimize(objective, n_trials=n_iter, show_progress_bar=True)#, gc_after_trial=True)
        params = study.best_trial.params
        if limit_search:
            if params['booster'] == 'dart':
                model = XGBClassifier(booster=params['booster'],  n_estimators=params['n_estimators'], reg_lambda=params['reg_lambda'], 
                    reg_alpha=params['reg_alpha'], max_depth=params['max_depth'], eta=params['eta'], gamma=params['gamma'], 
                    grow_policy=params['grow_policy'], sample_type=params['sample_type'], normalize_type=params['normalize_type'],
                    rate_drop=params['rate_drop'], skip_drop=params['skip_drop'], scale_pos_weight=sample_weight, random_state=1909)
            elif params['booster'] == 'gbtree':
                model = XGBClassifier(booster=params['booster'],  n_estimators=params['n_estimators'], reg_lambda=params['reg_lambda'], 
                    reg_alpha=params['reg_alpha'], max_depth=params['max_depth'], eta=params['eta'], gamma=params['gamma'], 
                    grow_policy=params['grow_policy'], subsample=params['subsample'], scale_pos_weight=sample_weight, random_state=1909)
        else:
            if params['booster'] == 'dart':
                model = XGBClassifier(booster=params['booster'], n_estimators=params['n_estimators'], colsample_bytree=params['colsample_bytree'], 
                    reg_lambda=params['reg_lambda'], reg_alpha=params['reg_alpha'], max_depth=params['max_depth'], eta=params['eta'], gamma=params['gamma'], 
                    grow_policy=params['grow_policy'], sample_type=params['sample_type'], normalize_type=params['normalize_type'],rate_drop=params['rate_drop'], 
                    skip_drop=params['skip_drop'], min_child_weight=params['min_child_weight'], max_delta_step=params['max_delta_step'], subsample=params['subsample'],
                    scale_pos_weight=sample_weight, random_state=1909)
            elif params['booster'] == 'gbtree':
                model = XGBClassifier(booster=params['booster'], n_estimators=params['n_estimators'], colsample_bytree=params['colsample_bytree'], 
                    reg_lambda=params['reg_lambda'], reg_alpha=params['reg_alpha'], max_depth=params['max_depth'], eta=params['eta'], gamma=params['gamma'], 
                    grow_policy=params['grow_policy'], subsample=params['subsample'], min_child_weight=params['min_child_weight'], max_delta_step=params['max_delta_step'],
                    scale_pos_weight=sample_weight, random_state=1909)
       
    else:
        objective = objective_cnn(data_x, data_y, img_num_channels=img_num_channels, normalize=normalize, min_pixel=min_pixel, max_pixel=max_pixel, 
            val_positive=val_X, val_negative=val_Y, train_epochs=train_epochs, patience=patience, metric=metric, average=average, test_positive=test_positive, test_negative=test_negative,
            opt_model=opt_model, opt_aug=opt_aug, batch_min=batch_min, batch_max=batch_max, image_size_min=image_size_min, image_size_max=image_size_max, balance=balance, 
            opt_max_min_pix=opt_max_min_pix, opt_max_max_pix=opt_max_max_pix, shift=shift, opt_cv=opt_cv, mask_size=mask_size, num_masks=num_masks, train_acc_threshold=train_acc_threshold,
            verbose=verbose, limit_search=limit_search, monitor1=monitor1, monitor2=monitor2, monitor1_thresh=monitor1_thresh, monitor2_thresh=monitor2_thresh)
        study.optimize(objective, n_trials=n_iter, show_progress_bar=True, gc_after_trial=True)#, n_jobs=1)
        params = study.best_trial.params

    final_score = study.best_value

    if clf != 'cnn':
        if initial_score > final_score:
            print('Hyperparameter optimization complete! Optimal performance of {} is LOWER than the base performance of {}, try increasing the value of n_iter and run again.'.format(np.round(final_score, 6), np.round(initial_score, 6)))
        else:
            print('Hyperparameter optimization complete! Optimal performance of {} is HIGHER than the base performance of {}.'.format(np.round(final_score, 6), np.round(initial_score, 6)))
        if return_study:
            return model, params, study
        return model, params
    else:
        print('Hyperparameter optimization complete! Optimal performance: {}'.format(np.round(final_score, 6)))
        if return_study:
            return params, study
        return params

def borutashap_opt(data_x, data_y, boruta_trials=50, model='rf', importance_type='gain'):
    """
    Applies a combination of the Boruta algorithm and
    Shapley values, a method developed by Eoghan Keany (2020).

    See: https://doi.org/10.5281/zenodo.4247618

    Args:
        data_x (ndarray): 2D array of size (n x m), where n is the
            number of samples, and m the number of features.
        data_y (ndarray, str): 1D array containing the corresponing labels.
        boruta_trials (int): The number of trials to run. A larger number is
            better as the distribution will be more robust to random fluctuations. 
            Defaults to 50.
        model (str): The ensemble method to use when fitting and calculating
            the feature importance metric. Only two options are currently
            supported, 'rf' for Random Forest and 'xgb' for Extreme Gradient Boosting.
            Defaults to 'rf'.
        importance_type (str): The feature importance type to use, only applicable
            when using clf='xgb'. The options include “gain”, “weight”, “cover”,
            “total_gain” or “total_cover”. Defaults to 'gain'.

    Returns:
        First output is a 1D array containing the indices of the selected features. 
        These indices can then be used to select the columns in the data_x array.
        Second output is the feature selection object, which contains feature selection
        history information and visualization options.
    """
    
    if boruta_trials == 0: #This is the flag that the ensemble_model.Classifier class uses to disable feature selection
        return np.arange(data_x.shape[1]), None

    if boruta_trials < 20:
        print('WARNING: Results are unstable if boruta_trials is too low!')
    if np.any(np.isnan(data_x)):
        #print('NaN values detected, applying Strawman imputation...')
        data_x = Strawman_imputation(data_x)

    if model == 'rf':
        classifier = RandomForestClassifier(random_state=1909)
    elif model == 'xgb':
        classifier = XGBClassifier(tree_method='exact', max_depth=20, importance_type=importance_type, random_state=1909)
    else:
        raise ValueError('Model argument must either be "rf" or "xgb".')
    
    try:
        #BorutaShap program requires input to have the columns attribute
        #Converting to Pandas dataframe
        cols = [str(i) for i in np.arange(data_x.shape[1])]
        X = DataFrame(data_x, columns=cols)
        y = np.zeros(len(data_y))

        #Below is to convert categorical labels to numerical, as per BorutaShap requirements
        for i, label in enumerate(np.unique(data_y)):
            mask = np.where(data_y == label)[0]
            y[mask] = i

        feat_selector = BorutaShap(model=classifier, importance_measure='shap', classification=True)
        print('Running feature selection...')
        feat_selector.fit(X=X, y=y, n_trials=boruta_trials, verbose=False, random_state=1909)

        index = np.array([int(feat) for feat in feat_selector.accepted])
        index.sort()
        print('Feature selection complete, {} selected out of {}!'.format(len(index), data_x.shape[1]))
    except:
        print('Boruta with Shapley values failed, switching to original Boruta...')
        index = boruta_opt(data_x, data_y)

    return index, feat_selector

def boruta_opt(data_x, data_y):
    """
    Applies the Boruta algorithm (Kursa & Rudnicki 2011) to identify features
    that perform worse than random.

    See: https://arxiv.org/pdf/1106.5112.pdf

    Args:
        data_x (ndarray): 2D array of size (n x m), where n is the
            number of samples, and m the number of features.
        data_y (ndarray, str): 1D array containing the corresponing labels.
            
    Returns:
        1D array containing the indices of the selected features. This can then
        be used to index the columns in the data_x array.
    """

    classifier = RandomForestClassifier(random_state=1909)

    feat_selector = BorutaPy(classifier, n_estimators='auto', random_state=1909)
    print('Running feature selection...')
    feat_selector.fit(data_x, data_y)

    feats = np.array([str(feat) for feat in feat_selector.support_])
    index = np.where(feats == 'True')[0]
    print('Feature selection complete, {} selected out of {}!'.format(len(index),len(feat_selector.support)))
    return index

def Strawman_imputation(data):
    """
    Perform Strawman imputation, a time-efficient algorithm
    in which missing data values are replaced with the median
    value of the entire, non-NaN sample. If the data is a hot-encoded
    boolean (as the RF does not allow True or False), then the 
    instance that is used the most will be computed as the median. 

    This is the baseline algorithm used by (Tang & Ishwaran 2017).
    See: https://arxiv.org/pdf/1701.05305.pdf

    Note:
        This function assumes each row corresponds to one sample, and 
        that missing values are masked as either NaN or inf. 

    Args:
        data (ndarray): 1D array if single parameter is input. If
            data is 2-dimensional, the medians will be calculated
            using the non-missing values in each corresponding column.

    Returns:
        The data array with the missing values filled in. 
    """

    if np.all(np.isfinite(data)):
        print('No missing values in data, returning original array.')
        return data 

    if len(data.shape) == 1:
        mask = np.where(np.isfinite(data))[0]
        median = np.median(data[mask])
        data[np.isnan(data)] = median 

        return data

    Ny, Nx = data.shape
    imputed_data = np.zeros((Ny,Nx))

    for i in range(Nx):
        mask = np.where(np.isfinite(data[:,i]))[0]
        median = np.median(data[:,i][mask])

        for j in range(Ny):
            if np.isnan(data[j,i]) == True or np.isinf(data[j,i]) == True:
                imputed_data[j,i] = median
            else:
                imputed_data[j,i] = data[j,i]

    return imputed_data 

def KNN_imputation(data, imputer=None, k=3):
    """
    Performs k-Nearest Neighbor imputation and transformation.
    By default the imputer will be created and returned, unless
    the imputer argument is set, in which case only the transformed
    data is output. 

    As this bundles neighbors according to their eucledian distance,
    it is sensitive to outliers. Can also yield weak predictions if the
    training features are heaviliy correlated.
    
    Args:
        imputer (optional): A KNNImputer class instance, configured using sklearn.impute.KNNImputer.
            Defaults to None, in which case the transformation is created using
            the data itself. 
        data (ndarray): 1D array if single parameter is input. If
            data is 2-dimensional, the medians will be calculated
            using the non-missing values in each corresponding column.
        k (int, optional): If imputer is None, this is the number
            of nearest neighbors to consider when computing the imputation.
            Defaults to 3. If imputer argument is set, this variable is ignored.

    Note:
        Tang & Ishwaran 2017 reported that if there is low to medium
        correlation in the dataset, RF imputation algorithms perform 
        better than KNN imputation

    Example:
        If we have our training data in an array called training_set, we 
        can create the imputer so that we can call it to transform new data
        when making on-the-field predictions.

        >>> imputed_data, knn_imputer = KNN_imputation(data=training_set, imputer=None)
        
        Now we can use the imputed data to create our machine learning model.
        Afterwards, when new data is input for prediction, we will insert our 
        imputer into the pipelinen by calling this function again, but this time
        with the imputer argument set:

        >>> new_data = knn_imputation(new_data, imputer=knn_imputer)

    Returns:
        The first output is the data array with with the missing values filled in. 
        The second output is the KNN Imputer that should be used to transform
        new data, prior to predictions. 
    """

    if imputer is None:
        imputer = KNNImputer(n_neighbors=k)
        imputer.fit(data)
        imputed_data = imputer.transform(data)
        return imputed_data, imputer

    return imputer.transform(data) 


















