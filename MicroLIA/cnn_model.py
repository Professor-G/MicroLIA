#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 10:37:01 2023

@author: daniel
"""
import os
import tensorflow as tf
os.environ['PYTHONHASHSEED'], os.environ["TF_DETERMINISTIC_OPS"] = '0', '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import copy 
import joblib
import pickle 
import numpy as np
import pkg_resources
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from cycler import cycler

from sklearn.manifold import TSNE
from tensorflow.keras.utils import to_categorical
import random as python_random
##https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development##
np.random.seed(1909), python_random.seed(1909), tf.random.set_seed(1909)

from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.backend import clear_session 
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.models import Sequential, save_model, load_model, Model
from tensorflow.keras.initializers import VarianceScaling, HeNormal

from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from tensorflow.keras.losses import categorical_crossentropy, Hinge, SquaredHinge, KLDivergence, LogCosh
from tensorflow.keras.layers import Input, Activation, Dense, Dropout, Conv2D, MaxPool2D, Add, ZeroPadding2D, \
    AveragePooling2D, GlobalAveragePooling2D, Flatten, BatchNormalization, Lambda, concatenate
from optuna.importance import get_param_importances, FanovaImportanceEvaluator
from MicroLIA.data_processing import process_class, create_training_set, concat_channels
from MicroLIA.data_augmentation import augmentation, resize, smote_oversampling, plot
from MicroLIA import optimization

class Classifier:
    """
    Creates and trains the convolutional neural network. The built-in methods can be used predict new samples, and also optimize the engine and output visualizations.
    REMINDER: horizontal = vertical = rotation = True 

    Note:
        The smote_sampling parameter is used in the SMOTE algorithm to specify the desired 
        ratio of the minority class to the majority class after oversampling the majority.

        If smote_sampling=1.0, the minority class will be oversampled to have the same number 
        of instances as the majority class, resulting in a balanced dataset.

        If smote_sampling=0.5, the minority class will be oversampled to have 50% of the number 
        of instances of the majority class.

        If smote_sampling='auto', SMOTE will automatically set the desired ratio between the minority 
        and majority classes to balance the dataset.

    Args:
        positive_class (ndarray): The samples for the first class should be passed, which will automatically 
            be assigned the positive label '1'.
        negative_class (ndarray, str): The samples for the second class should be passed, which will automatically 
            be assigned the negative label '0'.
        val_positive (ndarray, optional): Positive class data to be used for validation. Defaults to None.
        val_negative (ndarray, optional): Negative class data to be used for validation. Defaults to None.
        test_positive (ndarray, optional): Positive class data to be used for post-trial testing. Defaults to None.
        test_negative (ndarray, optional): Negative class data to be used for post-trial testing. Defaults to None.
        test_acc_threshold (float, optional): If input, models that yield test accuracies lower than the threshold will
            be rejected by the optimizer. The accuracy of both the test_positive and test_negative is asessed, if input.
            This is used to reject models that have over or under fit the training data. Defaults to None.
        post_metric (bool): If True, the test_positive and/or test_negative inputs will be included in the final optimization score.
            This will be the averaged out metric. Defaults to True. Can be set to False to only apply the test_acc_threshold.
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
        metric2 (str, optional): Additional metric to be used solely for early-stopping purposes. If input, the trial will stop if either
            metric or metric2 stop improving after the same patience number of epochs, but only the value of metric is used to assess
            the performance of the model after each trial. Defaults to None.        patience (int): Number of epochs without improvement before the optimization trial is terminated. Defaults to 0, which
            disables this feature.
        metric3 (str, optional): Additional metric to be used solely for early-stopping purposes. If input, the trial will stop if either
            metric or metric3 stop improving after the same patience number of epochs, but only the value of metric is used to assess
            the performance of the model after each trial. Defaults to None.        patience (int): Number of epochs without improvement before the optimization trial is terminated. Defaults to 0, which
            disables this feature.
        average (bool): If False, the designated metric will be calculated according to its value at the end of the train_epochs. 
            If True, the metric will be averaged out across all train_epochs. Defaults to True.
        opt_model (bool): If True, the architecture parameters will be optimized. Defaults to True.
        opt_aug (bool): If True, the augmentation procedure will be optimized. Defaults to False.
        batch_min (int): The minimum number of augmentations to perform per image on the positive class, only applicable 
            if opt_aug=True. Defaults to 2.
        batch_max (int): The maximum number of augmentations to perform per image on the positive class, only applicable 
            if opt_aug=True. Defaults to 25.
        batch_other (int): The number of augmentations to perform to the other class, presumed to be the majority class.
            Defaults to 1. This is done to ensure augmentation techniques are applied consistently across both classes.        
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
            regularizer. Only applicable if opt_aug=True. This value can either be an integer to hard-set the mask size everytime,
            or can be a tuple representing the lower and upper bounds, respectively, in which case the mask size will be optimized. 
            Defaults to None.
        num_masks (int, optional): The number of masks to create, to be used alongside the mask_size parameter. Note that if 
            this is set to a value greater than one, overlap may occur. This value can either be an integer to hard-set the number
            of masks everytime, or it can be a tuple representing the lower and upper bounds, respectively, in which case the number
            of masks will be optimized. Defaults to None.
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
        monitor1 (str, optional): The first metric to monitor, can take the same values as the metric argument. Defaults to None.
        monitor2 (str, optional): The second metric to monitor, can take the same values as the metric argument. Defaults to None.
        monitor1_thresh (float, optional): The threshold value of the first monitor metric. If the metric is loss-related
            the training will stop early if the value falls below this threshold. Similarly, if the metric is accuracy-related,
            then the training will stop early if the value falls above this threshold. Defaults to None.
        monitor2_thresh (float, optional): The threshold value of the second monitor metric. If the metric is loss-related
            the training will stop early if the value falls below this threshold. Similarly, if the metric is accuracy-related,
            then the training will stop early if the value falls above this threshold. Defaults to None.
        smote_sampling (float): The smote_sampling parameter is used in the SMOTE algorithm to specify the desired 
            ratio of the minority class to the majority class. Defaults to 0 which disables the procedure.
        clf (str): The designated cnn model to use, can either be 'alexnet', 'resnet18', 'vgg16', or 'custom_cnn'
        blend_max (float): A float greater than 1.1, corresponding to the increase in the minority class after the 
            blending augmentations, to be used if optimizing with opt_aug=True, then this parameter will be tuned and will be used as the 
            maximum increase to accept. For example, if opt_aug=True and blend_max=5, then the optimization will return
            an optimal value between 1 and 5. If 1, then the blending procedure is applied but the minority class size remains same the. If 5,
            then the minority class will be increased 500% via the blening routine. Defaults to 0 which disables this feature. To enable
            when opt_aug=True, set to to greater than or equal to 1.1 (a minimum of 10% increase), which would thus try different values for this
            during the optimization between 1 and 1.1.
        blend_other (float): Greater than or equal to 1. Can be zero to not apply augmentation to the majority class.
        path (str): Absolute path where the models and study object will be saved. Defaults to None, which saves the models and study
            in the home directory.
    """

    def __init__(self, positive_class=None, negative_class=None, val_positive=None, val_negative=None, img_num_channels=1, clf='alexnet', 
        normalize=False, min_pixel=0, max_pixel=100, optimize=False, n_iter=25, batch_size_min=16, batch_size_max=64, epochs=25, patience=5, metric='loss', metric2=None, metric3=None,
        average=True, test_positive=None, test_negative=None, test_acc_threshold=None, post_metric=True, opt_model=True, train_epochs=25, opt_cv=None,
        opt_aug=False, batch_min=2, batch_max=25, batch_other=1, balance=True, image_size_min=50, image_size_max=100, opt_max_min_pix=None, opt_max_max_pix=None, 
        shift=10, rotation=False, horizontal=False, vertical=False, mask_size=None, num_masks=None, smote_sampling=0, blend_max=0, blending_func='mean', num_images_to_blend=2, blend_other=1, zoom_range=None, skew_angle=0,
        limit_search=True, monitor1=None, monitor2=None, monitor1_thresh=None, monitor2_thresh=None, verbose=0, path=None, use_gpu=False):

        self.positive_class = positive_class
        self.negative_class = negative_class
        self.val_positive = val_positive
        self.val_negative = val_negative
        self.img_num_channels = img_num_channels
        self.clf = clf

        #Normalization parameters, will be used if opt_aug = False
        self.normalize = normalize
        self.min_pixel = min_pixel
        self.max_pixel = max_pixel

        #Optimization
        self.optimize = optimize
        self.n_iter = n_iter
        self.batch_size_min = batch_size_min
        self.batch_size_max = batch_size_max
        
        self.epochs = epochs
        self.patience = patience
        self.metric = metric
        self.metric2 = metric2
        self.metric3 = metric3
        self.average = average
        self.test_positive = test_positive
        self.test_negative = test_negative
        self.test_acc_threshold = test_acc_threshold
        self.post_metric = post_metric
        self.opt_model = opt_model
        self.train_epochs = train_epochs
        self.opt_cv = opt_cv

        #Augmentation params including min and max pixels normalization
        self.opt_aug = opt_aug
        self.batch_min = batch_min
        self.batch_max = batch_max
        self.batch_other = batch_other
        self.balance = balance
        self.image_size_min = image_size_min
        self.image_size_max = image_size_max
        self.opt_max_min_pix = opt_max_min_pix
        self.opt_max_max_pix = opt_max_max_pix

        #Image augmentation procedures
        self.shift = shift
        self.rotation = rotation
        self.horizontal = horizontal
        self.vertical = vertical
        self.mask_size = mask_size
        self.num_masks = num_masks
        self.smote_sampling = smote_sampling
        self.blend_max = blend_max
        self.blending_func = blending_func
        self.num_images_to_blend = num_images_to_blend
        self.blend_other = blend_other
        self.zoom_range = zoom_range
        self.skew_angle = skew_angle

        #Limit search and optional early-stopping monitors to speed up the optimization
        self.limit_search = limit_search
        self.monitor1 = monitor1
        self.monitor2 = monitor2
        self.monitor1_thresh = monitor1_thresh
        self.monitor2_thresh = monitor2_thresh
        #Verbose following the tf.keras convention
        self.verbose = verbose
        #Path for saving & loading, will start as None and be updated when objects are loaded/saved
        self.path = path
        #Whether to turn off GPU
        self.use_gpu = use_gpu

        if self.use_gpu is False:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

        if self.clf not in ['alexnet', 'vgg16', 'resnet18', 'custom_cnn']:
            raise ValueError('Invalid clf input, options are: "alexnet", "vgg16", "resnet18", or "custom_cnn".')

        if self.positive_class is not None:
            if len(self.positive_class.shape) == 4 and self.img_num_channels != self.positive_class.shape[-1]:
                print('NOTE: Detected {} filters but img_num_channels was set to {}, setting img_numg_channels={}'.format(self.positive_class.shape[-1], self.img_num_channels, self.positive_class.shape[-1]))
                self.img_num_channels = self.positive_class.shape[-1]
            if len(self.positive_class.shape) == 4 and self.img_num_channels == 1: #If it's just one filter convert to 3-D array
                self.positive_class = np.squeeze(self.positive_class)
            if len(self.negative_class.shape) == 4 and self.img_num_channels == 1: #If it's just one filter convert to 3-D array
                self.negative_class = np.squeeze(self.negative_class)
            if self.val_positive is not None:
                if len(self.val_positive.shape) == 4 and self.img_num_channels == 1: #If it's just one filter convert to 3-D array
                    self.val_positive = np.squeeze(self.val_positive)
                if len(self.val_positive.shape) == 2:
                    if self.img_num_channels != 1:
                        raise ValueError('Single image detected as the positive validation data, img_num_channels must be 1!')
                    else:
                        self.val_positive = np.reshape(self.val_positive, (1, self.val_positive.shape[0], self.val_positive.shape[1]))
            if self.val_negative is not None:
                if len(self.val_negative.shape) == 4 and self.img_num_channels == 1: #If it's just one filter convert to 3-D array
                    self.val_negative = np.squeeze(self.val_negative)
                if len(self.val_negative.shape) == 2:
                    if self.img_num_channels != 1:
                        raise ValueError('Single image detected as the negative validation data, img_num_channels must be 1!')
                    else:
                        self.val_negative = np.reshape(self.val_negative, (1, self.val_negative.shape[0], self.val_negative.shape[1]))

        if self.test_positive is not None or self.test_negative is not None:
            if self.post_metric is False and self.test_acc_threshold is None:
                raise ValueError('NOTE: Test data has been input but both post_metric=False and test_acc_threshold=None -- enable at least one option or set the test data to None!')


        #These will be the model attributes
        self.model = None
        self.history = None 
        self.best_params = None 
        self.optimization_results = None 

    def create(self, overwrite_training=False, save_training=False):
        """
        Generates the CNN machine learning model.

        Args:
            overwrite_training (bool)
            save_training (bool):
        
        Returns:
            Trained classifier.
        """

        if self.positive_class is None or self.negative_class is None:
            raise ValueError('No training data found! Input both the positive_class and the negative_class.')
            
        if self.optimize is False and self.best_params is None:
            if self.clf == 'alexnet':
                print("Returning base AlexNet model...")
                self.model, self.history = AlexNet(self.positive_class, self.negative_class, img_num_channels=self.img_num_channels, normalize=self.normalize,
                    min_pixel=self.min_pixel, max_pixel=self.max_pixel, val_positive=self.val_positive, val_negative=self.val_negative, epochs=self.epochs, 
                    smote_sampling=self.smote_sampling, patience=self.patience, metric=self.metric, save_training_data=save_training, path=self.path)
            elif self.clf == 'vgg16':
                print("Returning the base VGG16 model...")
                self.model, self.history = VGG16(self.positive_class, self.negative_class, img_num_channels=self.img_num_channels, normalize=self.normalize,
                    min_pixel=self.min_pixel, max_pixel=self.max_pixel, val_positive=self.val_positive, val_negative=self.val_negative, epochs=self.epochs, 
                    smote_sampling=self.smote_sampling, patience=self.patience, metric=self.metric, save_training_data=save_training, path=self.path)
            elif self.clf == 'resnet18':
                print("Returning the base ResNet-18 model...")
                self.model, self.history = Resnet18(self.positive_class, self.negative_class, img_num_channels=self.img_num_channels, normalize=self.normalize,
                    min_pixel=self.min_pixel, max_pixel=self.max_pixel, val_positive=self.val_positive, val_negative=self.val_negative, epochs=self.epochs, 
                    smote_sampling=self.smote_sampling, patience=self.patience, metric=self.metric, save_training_data=save_training, path=self.path)
            elif self.clf == 'custom_cnn':
                print("Returning the base custom model (1 convolutional layer + 1 dense layer)...")
                self.model, self.history = custom_model(self.positive_class, self.negative_class, img_num_channels=self.img_num_channels, normalize=self.normalize,
                    min_pixel=self.min_pixel, max_pixel=self.max_pixel, val_positive=self.val_positive, val_negative=self.val_negative, epochs=self.epochs, 
                    smote_sampling=self.smote_sampling, patience=self.patience, metric=self.metric, save_training_data=save_training, path=self.path)          
            
            print(); print('Complete! To save the final model and optimization results, call the save() method.') 
            if overwrite_training:
                print(); print("Can only use overwrite_training=True if optimizing the model!")

            return      

        if self.best_params is None:
            self.best_params, self.optimization_results = optimization.hyper_opt(self.positive_class, self.negative_class, val_X=self.val_positive, val_Y=self.val_negative, img_num_channels=self.img_num_channels, clf=self.clf,
                normalize=self.normalize, min_pixel=self.min_pixel, max_pixel=self.max_pixel, n_iter=self.n_iter, patience=self.patience, metric=self.metric, metric2=self.metric2, metric3=self.metric3, average=self.average,
                test_positive=self.test_positive, test_negative=self.test_negative, test_acc_threshold=self.test_acc_threshold, post_metric=self.post_metric, opt_model=self.opt_model, batch_size_min=self.batch_size_min, batch_size_max=self.batch_size_max, 
                train_epochs=self.train_epochs, opt_cv=self.opt_cv, opt_aug=self.opt_aug, batch_min=self.batch_min, batch_max=self.batch_max, batch_other=self.batch_other, balance=self.balance, image_size_min=self.image_size_min, image_size_max=self.image_size_max, 
                shift=self.shift, rotation=self.rotation, horizontal=self.horizontal, vertical=self.vertical, opt_max_min_pix=self.opt_max_min_pix, opt_max_max_pix=self.opt_max_max_pix, mask_size=self.mask_size, num_masks=self.num_masks, smote_sampling=self.smote_sampling, blend_max=self.blend_max, blend_other=self.blend_other, 
                num_images_to_blend=self.num_images_to_blend, blending_func=self.blending_func, zoom_range=self.zoom_range, skew_angle=self.skew_angle, limit_search=self.limit_search, monitor1=self.monitor1, monitor2=self.monitor2, monitor1_thresh=self.monitor1_thresh, 
                monitor2_thresh=self.monitor2_thresh, verbose=self.verbose, return_study=True)
            print("Fitting and returning final model...")
        else:
            if self.epochs != 0:
                print(); print('Fitting model using the best_params, if the class attributes were not loaded ensure they have been enabled...')
            else:
                print(); print('The epochs parameter is zero, returning nothing...')
                return 

        if self.epochs == 0:

            print(); print('The epochs parameter is zero, skipping final model training...')
            return

        else:

            clear_session()

            if self.opt_aug:
                if self.img_num_channels == 1:
                    channel1, channel2, channel3 = copy.deepcopy(self.positive_class), None, None 
                elif self.img_num_channels == 2:
                    channel1, channel2, channel3 = copy.deepcopy(self.positive_class[:,:,:,0]), copy.deepcopy(self.positive_class[:,:,:,1]), None 
                elif self.img_num_channels == 3:
                    channel1, channel2, channel3 = copy.deepcopy(self.positive_class[:,:,:,0]), copy.deepcopy(self.positive_class[:,:,:,1]), copy.deepcopy(self.positive_class[:,:,:,2])
                else:
                    raise ValueError('Only three filters are supported!')

                if self.opt_max_min_pix is not None:
                    self.normalize = True #In case it's mistakenly set to False by user
                    min_pix, max_pix = self.min_pixel, [] #Will append to a list because it's 1 max pix valuer per band!
                    max_pix.append(self.best_params['max_pixel_1']) if self.img_num_channels >= 1 else None
                    max_pix.append(self.best_params['max_pixel_2']) if self.img_num_channels >= 2 else None
                    max_pix.append(self.best_params['max_pixel_3']) if self.img_num_channels == 3 else None
                    self.max_pixel = max_pix; print('Setting max_pixel attribute to the tuned value(s)...')
                else:
                    min_pix, max_pix = self.min_pixel, self.max_pixel

                blend_multiplier = self.best_params['blend_multiplier'] if self.blend_max >= 1.1 else 0
                skew_angle = self.best_params['skew_angle'] if self.skew_angle > 0 else 0
                mask_size = self.best_params['mask_size'] if isinstance(self.mask_size, tuple) else self.mask_size
                num_masks = self.best_params['num_masks'] if isinstance(self.num_masks, tuple) else self.num_masks

                print(); print('======= Image Parameters ======'); print(); print('Num Augmentations :', self.best_params['num_aug']); print('Image Size : ', self.best_params['image_size']); print('Max Pixel(s) :', max_pix); print('Num Masks :', num_masks); print('Mask Size :', mask_size); print('Blend Multiplier :', blend_multiplier); print('Skew Angle :', skew_angle)

                augmented_images = augmentation(channel1=channel1, channel2=channel2, channel3=channel3, batch=self.best_params['num_aug'], 
                    width_shift=self.shift, height_shift=self.shift, horizontal=self.horizontal, vertical=self.vertical, rotation=self.rotation, 
                    image_size=self.best_params['image_size'], mask_size=mask_size, num_masks=num_masks, blend_multiplier=blend_multiplier, 
                    blending_func=self.blending_func, num_images_to_blend=self.num_images_to_blend, zoom_range=self.zoom_range, skew_angle=skew_angle)

                #The augmentation routine returns an output for each filter, e.g. 3 outputs for RGB
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

                #Perform same augmentation techniques on other data, batch_other=1 by default
                if self.img_num_channels == 1:
                    channel1, channel2, channel3 = copy.deepcopy(self.negative_class), None, None 
                elif self.img_num_channels == 2:
                    channel1, channel2, channel3 = copy.deepcopy(self.negative_class[:,:,:,0]), copy.deepcopy(self.negative_class[:,:,:,1]), None 
                elif self.img_num_channels == 3:
                    channel1, channel2, channel3 = copy.deepcopy(self.negative_class[:,:,:,0]), copy.deepcopy(self.negative_class[:,:,:,1]), copy.deepcopy(self.negative_class[:,:,:,2])
                
                augmented_images_negative = augmentation(channel1=channel1, channel2=channel2, channel3=channel3, batch=self.batch_other, 
                    width_shift=self.shift, height_shift=self.shift, horizontal=self.horizontal, vertical=self.vertical, rotation=self.rotation, 
                    image_size=self.best_params['image_size'], mask_size=mask_size, num_masks=num_masks, blend_multiplier=self.blend_other, 
                    blending_func=self.blending_func, num_images_to_blend=self.num_images_to_blend, zoom_range=self.zoom_range, skew_angle=skew_angle)
                
                #The augmentation routine returns an output for each filter, e.g. 3 outputs for RGB
                if self.img_num_channels > 1:
                    class_2=[]
                    if self.img_num_channels == 2:
                        for i in range(len(augmented_images_negative[0])):
                            class_2.append(concat_channels(augmented_images_negative[0][i], augmented_images_negative[1][i]))
                    else:
                        for i in range(len(augmented_images_negative[0])):
                            class_2.append(concat_channels(augmented_images_negative[0][i], augmented_images_negative[1][i], augmented_images_negative[2][i]))
                    class_2 = np.array(class_2)
                else:
                    class_2 = augmented_images_negative

                #Balance the class sizes if necessary
                if self.balance:
                    if self.batch_other > 1: #Must shuffle!!!
                        ix = np.random.permutation(len(class_2))
                        class_2 = class_2[ix]
                    class_2 = class_2[:len(class_1)]     

                if self.img_num_channels == 1:
                    class_2 = resize(class_2, size=self.best_params['image_size'])
                else:
                    channel1 = resize(class_2[:,:,:,0], size=self.best_params['image_size'])
                    channel2 = resize(class_2[:,:,:,1], size=self.best_params['image_size'])
                    if self.img_num_channels == 2:
                        class_2 = concat_channels(channel1, channel2)
                    else:
                        channel3 = resize(class_2[:,:,:,2], size=self.best_params['image_size'])
                        class_2 = concat_channels(channel1, channel2, channel3)

                if self.val_positive is not None:
                    if self.img_num_channels == 1:
                        val_class_1 = resize(self.val_positive, size=self.best_params['image_size'])
                    else:
                        val_channel1 = resize(self.val_positive[:,:,:,0], size=self.best_params['image_size'])
                        val_channel2 = resize(self.val_positive[:,:,:,1], size=self.best_params['image_size'])
                        if self.img_num_channels == 2:
                            val_class_1 = concat_channels(val_channel1, val_channel2)
                        else:
                            val_channel3 = resize(self.val_positive[:,:,:,2], size=self.best_params['image_size'])
                            val_class_1 = concat_channels(val_channel1, val_channel2, val_channel3)
                else:
                    val_class_1 = None

                if self.val_negative is not None:
                    if self.img_num_channels == 1:
                        val_class_2 = resize(self.val_negative, size=self.best_params['image_size'])
                    elif self.img_num_channels > 1:
                        val_channel1 = resize(self.val_negative[:,:,:,0], size=self.best_params['image_size'])
                        val_channel2 = resize(self.val_negative[:,:,:,1], size=self.best_params['image_size'])
                        if self.img_num_channels == 2:
                            val_class_2 = concat_channels(val_channel1, val_channel2)
                        else:
                            val_channel3 = resize(self.val_negative[:,:,:,2], size=self.best_params['image_size'])
                            val_class_2 = concat_channels(val_channel1, val_channel2, val_channel3)
                else:
                    val_class_2 = None
            else:
                class_1, class_2 = self.positive_class, self.negative_class
                val_class_1, val_class_2 = self.val_positive, self.val_negative
                min_pix, max_pix = self.min_pixel, self.max_pixel

            #Set the batch_size and learning parameters
            if self.batch_size_min == self.batch_size_max:
                batch_size = self.batch_size_max 
            else:
                batch_size = self.best_params['batch_size']

            lr = self.best_params['lr']
            optimizer = self.best_params['optimizer']
            decay = 0

            #Inverse time decay is set to 0, optimizzing beta and rho parameters instead.
            if optimizer == 'sgd':
                momentum = self.best_params['momentum']
                nesterov = self.best_params['nesterov']
                beta_1 = beta_2 = 0; amsgrad = False
            elif optimizer == 'adam' or optimizer == 'adamax' or optimizer == 'nadam':
                beta_1 = self.best_params['beta_1']
                beta_2 = self.best_params['beta_2']
                amsgrad = self.best_params['amsgrad'] if optimizer == 'adam' else False
                momentum, nesterov = 0.0, False
            elif optimizer == 'adadelta' or optimizer == 'rmsprop':
                rho = self.best_params['rho']
                momentum = beta_1 = beta_2 = 0; nesterov = amsgrad = False

            if self.opt_cv is not None and self.verbose == 1:
                    print(); print('***********  CV - 1 ***********'); print()

            if self.opt_model:
                #If opt_model=True, the optimization routine will tune the model regularizer and the pooling types
                #If limit_search=False, the layer parameters (filters, pool sizes, etc) will be tuned as well (might crash machine due to memory allocation error!)
                if self.clf == 'alexnet':
                    if self.limit_search:
                        self.model, self.history = AlexNet(class_1, class_2, img_num_channels=self.img_num_channels, 
                            normalize=self.normalize, min_pixel=min_pix, max_pixel=max_pix, val_positive=val_class_1, val_negative=val_class_2, 
                            epochs=self.epochs, batch_size=batch_size, optimizer=optimizer, lr=lr, decay=decay, momentum=momentum, nesterov=nesterov, 
                            beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad, loss=self.best_params['loss'], activation_conv=self.best_params['activation_conv'], 
                            activation_dense=self.best_params['activation_dense'], conv_init=self.best_params['conv_init'], dense_init=self.best_params['dense_init'], 
                            model_reg=self.best_params['model_reg'], pooling_1=self.best_params['pooling_1'], pooling_2=self.best_params['pooling_2'], pooling_3=self.best_params['pooling_3'], 
                            smote_sampling=self.smote_sampling, patience=self.patience, metric=self.metric, checkpoint=False, verbose=self.verbose, save_training_data=save_training, path=self.path)
                    else:
                        self.model, self.history = AlexNet(class_1, class_2, img_num_channels=self.img_num_channels, 
                            normalize=self.normalize, min_pixel=min_pix, max_pixel=max_pix, val_positive=val_class_1, val_negative=val_class_2, 
                            epochs=self.epochs, batch_size=batch_size, optimizer=optimizer, lr=lr, decay=decay, momentum=momentum, nesterov=nesterov, 
                            beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad, loss=self.best_params['loss'], activation_conv=self.best_params['activation_conv'], 
                            activation_dense=self.best_params['activation_dense'], conv_init=self.best_params['conv_init'], dense_init=self.best_params['dense_init'], model_reg=self.best_params['model_reg'], 
                            filter_1=self.best_params['filter_1'], filter_size_1=self.best_params['filter_size_1'], strides_1=self.best_params['strides_1'], pooling_1=self.best_params['pooling_1'], pool_size_1=self.best_params['pool_size_1'], pool_stride_1=self.best_params['pool_stride_1'],
                            filter_2=self.best_params['filter_2'], filter_size_2=self.best_params['filter_size_2'], strides_2=self.best_params['strides_2'], pooling_2=self.best_params['pooling_2'], pool_size_2=self.best_params['pool_size_2'], pool_stride_2=self.best_params['pool_stride_2'],
                            filter_3=self.best_params['filter_3'], filter_size_3=self.best_params['filter_size_3'], strides_3=self.best_params['strides_3'], pooling_3=self.best_params['pooling_3'], pool_size_3=self.best_params['pool_size_3'], pool_stride_3=self.best_params['pool_stride_3'], 
                            filter_4=self.best_params['filter_4'], filter_size_4=self.best_params['filter_size_4'], strides_4=self.best_params['strides_4'], 
                            filter_5=self.best_params['filter_5'], filter_size_5=self.best_params['filter_size_5'], strides_5=self.best_params['strides_5'], 
                            dense_neurons_1=self.best_params['dense_neurons_1'], dense_neurons_2=self.best_params['dense_neurons_2'], dropout_1=self.best_params['dropout_1'], dropout_2=self.best_params['dropout_2'],
                            smote_sampling=self.smote_sampling, patience=self.patience, metric=self.metric, checkpoint=False, verbose=self.verbose, save_training_data=save_training, path=self.path)

                elif self.clf == 'resnet18':
                    if self.limit_search:
                        self.model, self.history = Resnet18(class_1, class_2, img_num_channels=self.img_num_channels, 
                            normalize=self.normalize, min_pixel=min_pix, max_pixel=max_pix, val_positive=val_class_1, val_negative=val_class_2, 
                            epochs=self.epochs, batch_size=batch_size, optimizer=optimizer, lr=lr, decay=decay, momentum=momentum, nesterov=nesterov, 
                            beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad, loss=self.best_params['loss'], activation_conv=self.best_params['activation_conv'], 
                            activation_dense=self.best_params['activation_dense'], conv_init=self.best_params['conv_init'], dense_init=self.best_params['dense_init'], 
                            model_reg=self.best_params['model_reg'], pooling=self.best_params['pooling'],
                            smote_sampling=self.smote_sampling, patience=self.patience, metric=self.metric, checkpoint=False, verbose=self.verbose, save_training_data=save_training, path=self.path)
                    else:
                        self.model, self.history = Resnet18(class_1, class_2, img_num_channels=self.img_num_channels, 
                            normalize=self.normalize, min_pixel=min_pix, max_pixel=max_pix, val_positive=val_class_1, val_negative=val_class_2, 
                            epochs=self.epochs, batch_size=batch_size, optimizer=optimizer, lr=lr, decay=decay, momentum=momentum, nesterov=nesterov, 
                            beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad, loss=self.best_params['loss'], activation_conv=self.best_params['activation_conv'], 
                            activation_dense=self.best_params['activation_dense'], conv_init=self.best_params['conv_init'], dense_init=self.best_params['dense_init'], 
                            model_reg=self.best_params['model_reg'], filters=self.best_params['filters'], filter_size=self.best_params['filter_size'], strides=self.best_params['strides'],  
                            pooling=self.best_params['pooling'], pool_size=self.best_params['pool_size'], pool_stride=self.best_params['pool_stride'], block_filters_1=self.best_params['block_filters_1'], 
                            block_filters_2=self.best_params['block_filters_2'], block_filters_3=self.best_params['block_filters_3'], block_filters_4=self.best_params['block_filters_4'], 
                            block_filters_size=self.best_params['block_filters_size'], smote_sampling=self.smote_sampling, patience=self.patience, metric=self.metric, checkpoint=False, verbose=self.verbose, save_training_data=save_training, path=self.path)
                
                elif self.clf == 'vgg16':
                    if self.limit_search:
                        self.model, self.history = VGG16(class_1, class_2, img_num_channels=self.img_num_channels, 
                            normalize=self.normalize, min_pixel=min_pix, max_pixel=max_pix, val_positive=val_class_1, val_negative=val_class_2, 
                            epochs=self.epochs, batch_size=batch_size, optimizer=optimizer, lr=lr, decay=decay, momentum=momentum, nesterov=nesterov, 
                            beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad, loss=self.best_params['loss'], activation_conv=self.best_params['activation_conv'], 
                            activation_dense=self.best_params['activation_dense'], conv_init=self.best_params['conv_init'], dense_init=self.best_params['dense_init'], 
                            model_reg=self.best_params['model_reg'], pooling_1=self.best_params['pooling_1'], pooling_2=self.best_params['pooling_2'], pooling_3=self.best_params['pooling_3'], 
                            pooling_4=self.best_params['pooling_4'], pooling_5=self.best_params['pooling_5'], smote_sampling=self.smote_sampling, patience=self.patience, metric=self.metric, checkpoint=False, verbose=self.verbose, save_training_data=save_training, path=self.path)
                    else:
                        self.model, self.history = VGG16(class_1, class_2, img_num_channels=self.img_num_channels, 
                            normalize=self.normalize, min_pixel=min_pix, max_pixel=max_pix, val_positive=val_class_1, val_negative=val_class_2, 
                            epochs=self.epochs, batch_size=batch_size, optimizer=optimizer, lr=lr, decay=decay, momentum=momentum, nesterov=nesterov, 
                            beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad, loss=self.best_params['loss'], activation_conv=self.best_params['activation_conv'], 
                            activation_dense=self.best_params['activation_dense'], conv_init=self.best_params['conv_init'], dense_init=self.best_params['dense_init'], model_reg=self.best_params['model_reg'],                 
                            filter_1=self.best_params['filter_1'], filter_size_1=self.best_params['filter_size_1'], strides_1=self.best_params['strides_1'], pooling_1=self.best_params['pooling_1'], pool_size_1=self.best_params['pool_size_1'], pool_stride_1=self.best_params['pool_stride_1'],
                            filter_2=self.best_params['filter_2'], filter_size_2=self.best_params['filter_size_2'], strides_2=self.best_params['strides_2'], pooling_2=self.best_params['pooling_2'], pool_size_2=self.best_params['pool_size_2'], pool_stride_2=self.best_params['pool_stride_2'],
                            filter_3=self.best_params['filter_3'], filter_size_3=self.best_params['filter_size_3'], strides_3=self.best_params['strides_3'], pooling_3=self.best_params['pooling_3'], pool_size_3=self.best_params['pool_size_3'], pool_stride_3=self.best_params['pool_stride_3'],
                            filter_4=self.best_params['filter_4'], filter_size_4=self.best_params['filter_size_4'], strides_4=self.best_params['strides_4'], pooling_4=self.best_params['pooling_4'], pool_size_4=self.best_params['pool_size_4'], pool_stride_4=self.best_params['pool_stride_4'],
                            filter_5=self.best_params['filter_5'], filter_size_5=self.best_params['filter_size_5'], strides_5=self.best_params['strides_5'], pooling_5=self.best_params['pooling_5'], pool_size_5=self.best_params['pool_size_5'], pool_stride_5=self.best_params['pool_stride_5'],
                            dense_neurons_1=self.best_params['dense_neurons_1'], dense_neurons_2=self.best_params['dense_neurons_2'], dropout_1=self.best_params['dropout_1'], dropout_2=self.best_params['dropout_2'],
                            smote_sampling=self.smote_sampling, patience=self.patience, metric=self.metric, checkpoint=False, verbose=self.verbose, save_training_data=save_training, path=self.path)
                
                elif self.clf == 'custom_cnn':
                    #Need to extract the second and third layers manually 
                    #Conv2D and Pooling Layers
                    strides_1 = pool_stride_1 = 1 
                    if self.best_params['num_conv_layers'] == 1:
                        filter_2 = filter_size_2 = strides_2 = pool_size_2 = pool_stride_2 = filter_3 = filter_size_3 = strides_3 = pool_size_3 = pool_stride_3 = 0; pooling_2 = pooling_3 = None
                    if self.best_params['num_conv_layers'] >= 2:
                        filter_2 = self.best_params['filter_2']
                        filter_size_2 = self.best_params['filter_size_2'] 
                        pooling_2 = self.best_params['pooling_2']
                        pool_size_2 = self.best_params['pool_size_2']
                        strides_2 = pool_stride_2 = 1; filter_3 = filter_size_3 = strides_3 = pool_size_3 = pool_stride_3 = 0; pooling_3 = None
                    if self.best_params['num_conv_layers'] == 3:
                        filter_3 = self.best_params['filter_3']
                        filter_size_3 = self.best_params['filter_size_3']
                        pooling_3 = self.best_params['pooling_3']
                        pool_size_3 = self.best_params['pool_size_3']
                        strides_3 = pool_stride_3 = 1
                    #Dense Layers
                    if self.best_params['num_dense_layers'] == 1:
                        dense_neurons_2 = dropout_2 = dense_neurons_3 = dropout_3 = 0
                    if self.best_params['num_dense_layers'] >= 2:
                        dense_neurons_2 = self.best_params['dense_neurons_2']
                        dropout_2 = self.best_params['dropout_2'] 
                        dense_neurons_3 = dropout_3 = 0
                    if self.best_params['num_dense_layers'] == 3:
                        dense_neurons_3 =  self.best_params['dense_neurons_3']                         
                        dropout_3 = self.best_params['dropout_3'] 

                    self.model, self.history = custom_model(class_1, class_2, img_num_channels=self.img_num_channels, 
                        normalize=self.normalize, min_pixel=min_pix, max_pixel=max_pix, val_positive=val_class_1, val_negative=val_class_2, 
                        epochs=self.epochs, batch_size=batch_size, optimizer=optimizer, lr=lr, decay=decay, momentum=momentum, nesterov=nesterov, 
                        beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad, loss=self.best_params['loss'], activation_conv=self.best_params['activation_conv'], 
                        activation_dense=self.best_params['activation_dense'], conv_init=self.best_params['conv_init'], dense_init=self.best_params['dense_init'], model_reg=self.best_params['model_reg'],
                        filter_1=self.best_params['filter_1'], filter_size_1=self.best_params['filter_size_1'], strides_1=strides_1, pooling_1=self.best_params['pooling_1'], pool_size_1=self.best_params['pool_size_1'], pool_stride_1=pool_stride_1, 
                        filter_2=filter_2, filter_size_2=filter_size_2, strides_2=strides_2, pooling_2=pooling_2, pool_size_2=pool_size_2, pool_stride_2=pool_stride_2, 
                        filter_3=filter_3, filter_size_3=filter_size_3, strides_3=strides_3, pooling_3=pooling_3, pool_size_3=pool_size_3, pool_stride_3=pool_stride_3, 
                        dense_neurons_1=self.best_params['dense_neurons_1'], dense_neurons_2=dense_neurons_2, dense_neurons_3=dense_neurons_3, 
                        dropout_1=self.best_params['dropout_1'], dropout_2=dropout_2, dropout_3=dropout_3, 
                        smote_sampling=self.smote_sampling, patience=self.patience, metric=self.metric, checkpoint=False, verbose=self.verbose, save_training_data=save_training, path=self.path)
            else: 
                if self.clf == 'alexnet':
                    self.model, self.history = AlexNet(class_1, class_2, img_num_channels=self.img_num_channels, normalize=self.normalize,
                        min_pixel=min_pix, max_pixel=max_pix, val_positive=val_class_1, val_negative=val_class_2, epochs=self.epochs,
                        batch_size=batch_size, optimizer=optimizer, lr=lr, momentum=momentum, decay=decay, nesterov=nesterov, 
                        smote_sampling=self.smote_sampling, patience=self.patience, metric=self.metric, checkpoint=False, verbose=self.verbose, save_training_data=save_training, path=self.path)
                elif self.clf == 'custom_cnn':
                    self.model, self.history = custom_model(class_1, class_2, img_num_channels=self.img_num_channels, normalize=self.normalize,
                        min_pixel=min_pix, max_pixel=max_pix, val_positive=val_class_1, val_negative=val_class_2, epochs=self.epochs,
                        batch_size=batch_size, optimizer=optimizer, lr=lr, momentum=momentum, decay=decay, nesterov=nesterov, 
                        smote_sampling=self.smote_sampling, patience=self.patience, metric=self.metric, checkpoint=False, verbose=self.verbose, save_training_data=save_training, path=self.path)
                elif self.clf == 'vgg16':
                    self.model, self.history = VGG16(class_1, class_2, img_num_channels=self.img_num_channels, normalize=self.normalize,
                        min_pixel=min_pix, max_pixel=max_pix, val_positive=val_class_1, val_negative=val_class_2, epochs=self.epochs,
                        batch_size=batch_size, optimizer=optimizer, lr=lr, momentum=momentum, decay=decay, nesterov=nesterov, 
                        smote_sampling=self.smote_sampling, patience=self.patience, metric=self.metric, checkpoint=False, verbose=self.verbose, save_training_data=save_training, path=self.path)
                elif self.clf == 'resnet18':
                    self.model, self.history = Resnet18(class_1, class_2, img_num_channels=self.img_num_channels, normalize=self.normalize,
                        min_pixel=min_pix, max_pixel=max_pix, val_positive=val_class_1, val_negative=val_class_2, epochs=self.epochs,
                        batch_size=batch_size, optimizer=optimizer, lr=lr, momentum=momentum, decay=decay, nesterov=nesterov, 
                        smote_sampling=self.smote_sampling, patience=self.patience, metric=self.metric, checkpoint=False, verbose=self.verbose, save_training_data=save_training, path=self.path)
        
            #################################

            ##### Cross-Validation Routine - implementation in which the validation data is inserted into the training data with the replacement serving as the new validation#####
            if self.opt_cv is not None:

                models, histories = [], [] #Will be used to append additional models, it opt_cv is enabled

                models.append(self.model); histories.append(self.history) #Appending the already created first model & history

                if self.val_positive is None and self.val_negative is None:
                    raise ValueError('CNN cross-validation is only supported if validation data is input.')
                if self.val_positive is not None:
                    if len(self.positive_class) / len(self.val_positive) < self.opt_cv-1:
                        raise ValueError('Cannot evenly partition the positive training/validation data, refer to the MicroLIA API documentation for instructions on how to use the opt_cv parameter.')
                if self.val_negative is not None:
                    if len(self.negative_class) / len(self.val_negative) < self.opt_cv-1:
                        raise ValueError('Cannot evenly partition the negative training/validation data, refer to the MicroLIA API documentation for instructions on how to use the opt_cv parameter.')
                
                #The first model (therefore the first "fold") already ran, therefore sutbract 1      
                for k in range(self.opt_cv-1):        

                    #Make deep copies to avoid overwriting arrays
                    class_1, class_2 = copy.deepcopy(self.positive_class), copy.deepcopy(self.negative_class)
                    val_class_1, val_class_2 = copy.deepcopy(self.val_positive), copy.deepcopy(self.val_negative)

                    #Sort the new data samples, no random shuffling, just a linear sequence
                    if val_class_1 is not None:
                        val_hold_1 = copy.deepcopy(class_1[k*len(val_class_1):len(val_class_1)*(k+1)]) #The new positive validation data
                        class_1[k*len(val_class_1):len(val_class_1)*(k+1)] = copy.deepcopy(val_class_1) #The new class_1, copying to avoid linkage between arrays
                        val_class_1 = val_hold_1 
                    if val_class_2 is not None:
                        val_hold_2 = copy.deepcopy(class_2[k*len(val_class_2):len(val_class_2)*(k+1)]) #The new validation data
                        class_2[k*len(val_class_2):len(val_class_2)*(k+1)] = copy.deepcopy(val_class_2) #The new class_2, copying to avoid linkage between arrays
                        val_class_2 = val_hold_2 

                    if self.opt_aug:
                        if self.img_num_channels == 1:
                            channel1, channel2, channel3 = copy.deepcopy(class_1), None, None 
                        elif self.img_num_channels == 2:
                            channel1, channel2, channel3 = copy.deepcopy(class_1[:,:,:,0]), copy.deepcopy(class_1[:,:,:,1]), None 
                        else:
                            channel1, channel2, channel3 = copy.deepcopy(class_1[:,:,:,0]), copy.deepcopy(class_1[:,:,:,1]), copy.deepcopy(class_1[:,:,:,2])

                        augmented_images = augmentation(channel1=channel1, channel2=channel2, channel3=channel3, batch=self.best_params['num_aug'], 
                            width_shift=self.shift, height_shift=self.shift, horizontal=self.horizontal, vertical=self.vertical, rotation=self.rotation, 
                            image_size=self.best_params['image_size'], mask_size=mask_size, num_masks=num_masks, blend_multiplier=blend_multiplier, 
                            blending_func=self.blending_func, num_images_to_blend=self.num_images_to_blend, zoom_range=self.zoom_range, skew_angle=skew_angle)

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

                        #Perform same augmentation techniques on negative class data, batch_other=1 by default
                        if self.img_num_channels == 1:
                            channel1, channel2, channel3 = copy.deepcopy(class_2), None, None 
                        elif self.img_num_channels == 2:
                            channel1, channel2, channel3 = copy.deepcopy(class_2[:,:,:,0]), copy.deepcopy(class_2[:,:,:,1]), None 
                        elif self.img_num_channels == 3:
                            channel1, channel2, channel3 = copy.deepcopy(class_2[:,:,:,0]), copy.deepcopy(class_2[:,:,:,1]), copy.deepcopy(class_2[:,:,:,2])
                        
                        augmented_images_negative = augmentation(channel1=channel1, channel2=channel2, channel3=channel3, batch=self.batch_other, 
                            width_shift=self.shift, height_shift=self.shift, horizontal=self.horizontal, vertical=self.vertical, rotation=self.rotation, 
                            image_size=self.best_params['image_size'], mask_size=mask_size, num_masks=num_masks, blend_multiplier=self.blend_other, 
                            blending_func=self.blending_func, num_images_to_blend=self.num_images_to_blend, zoom_range=self.zoom_range, skew_angle=skew_angle)

                        #The augmentation routine returns an output for each filter, e.g. 3 outputs for RGB
                        if self.img_num_channels > 1:
                            class_2=[]
                            if self.img_num_channels == 2:
                                for i in range(len(augmented_images_negative[0])):
                                    class_2.append(concat_channels(augmented_images_negative[0][i], augmented_images_negative[1][i]))
                            else:
                                for i in range(len(augmented_images_negative[0])):
                                    class_2.append(concat_channels(augmented_images_negative[0][i], augmented_images_negative[1][i], augmented_images_negative[2][i]))
                            class_2 = np.array(class_2)
                        else:
                            class_2 = augmented_images_negative

                        #Balance the class sizes if necessary
                        if self.balance:
                            if self.batch_other > 1: #Must shuffle!!!
                                ix = np.random.permutation(len(class_2))
                                class_2 = class_2[ix]
                            class_2 = class_2[:len(class_1)]   

                        if self.img_num_channels == 1:
                            class_2 = resize(class_2, size=self.best_params['image_size'])
                        else:
                            channel1 = resize(class_2[:,:,:,0], size=self.best_params['image_size'])
                            channel2 = resize(class_2[:,:,:,1], size=self.best_params['image_size'])
                            if self.img_num_channels == 2:
                                class_2 = concat_channels(channel1, channel2)
                            else:
                                channel3 = resize(class_2[:,:,:,2], size=self.best_params['image_size'])
                                class_2 = concat_channels(channel1, channel2, channel3)

                        if val_class_1 is not None:
                            if self.img_num_channels == 1:
                                val_class_1 = resize(val_class_1, size=self.best_params['image_size'])
                            else:
                                val_channel1 = resize(val_class_1[:,:,:,0], size=self.best_params['image_size'])
                                val_channel2 = resize(val_class_1[:,:,:,1], size=self.best_params['image_size'])
                                if self.img_num_channels == 2:
                                    val_class_1 = concat_channels(val_channel1, val_channel2)
                                else:
                                    val_channel3 = resize(val_class_1[:,:,:,2], size=self.best_params['image_size'])
                                    val_class_1 = concat_channels(val_channel1, val_channel2, val_channel3)

                        if val_class_2 is not None:
                            if self.img_num_channels == 1:
                                val_class_2 = resize(val_class_2, size=self.best_params['image_size'])
                            elif self.img_num_channels > 1:
                                val_channel1 = resize(val_class_2[:,:,:,0], size=self.best_params['image_size'])
                                val_channel2 = resize(val_class_2[:,:,:,1], size=self.best_params['image_size'])
                                if self.img_num_channels == 2:
                                    val_class_2 = concat_channels(val_channel1, val_channel2)
                                else:
                                    val_channel3 = resize(val_class_2[:,:,:,2], size=self.best_params['image_size'])
                                    val_class_2 = concat_channels(val_channel1, val_channel2, val_channel3)

                    if self.verbose == 1:
                        print(); print('***********  CV - {} ***********'.format(k+2)); print()

                    clear_session()

                    if self.opt_model is False:
                        if self.clf == 'alexnet':
                            model, history = AlexNet(class_1, class_2, img_num_channels=self.img_num_channels, 
                                normalize=self.normalize, min_pixel=min_pix, max_pixel=max_pix, val_positive=val_class_1, val_negative=val_class_2, 
                                epochs=self.epochs, batch_size=batch_size, optimizer=optimizer, lr=lr, decay=decay, momentum=momentum, nesterov=nesterov, 
                                beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad, smote_sampling=self.smote_sampling, patience=self.patience, metric=self.metric, 
                                checkpoint=False, verbose=self.verbose, save_training_data=save_training, path=self.path)
                        elif self.clf == 'custom_cnn':
                            model, history = custom_model(class_1, class_2, img_num_channels=self.img_num_channels, 
                                normalize=self.normalize, min_pixel=min_pix, max_pixel=max_pix, val_positive=val_class_1, val_negative=val_class_2, 
                                epochs=self.epochs, batch_size=batch_size, optimizer=optimizer, lr=lr, decay=decay, momentum=momentum, nesterov=nesterov, 
                                beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad, smote_sampling=self.smote_sampling, patience=self.patience, metric=self.metric, 
                                checkpoint=False, verbose=self.verbose, save_training_data=save_training, path=self.path)
                        elif self.clf == 'vgg16':
                            model, history = VGG16(class_1, class_2, img_num_channels=self.img_num_channels, 
                                normalize=self.normalize, min_pixel=min_pix, max_pixel=max_pix, val_positive=val_class_1, val_negative=val_class_2, 
                                epochs=self.epochs, batch_size=batch_size, optimizer=optimizer, lr=lr, decay=decay, momentum=momentum, nesterov=nesterov, 
                                beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad, smote_sampling=self.smote_sampling, patience=self.patience, metric=self.metric, 
                                checkpoint=False, verbose=self.verbose, save_training_data=save_training, path=self.path)
                        elif self.clf == 'resnet18':
                            model, history = Resnet18(class_1, class_2, img_num_channels=self.img_num_channels, 
                                normalize=self.normalize, min_pixel=min_pix, max_pixel=max_pix, val_positive=val_class_1, val_negative=val_class_2, 
                                epochs=self.epochs, batch_size=batch_size, optimizer=optimizer, lr=lr, decay=decay, momentum=momentum, nesterov=nesterov, 
                                beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad, smote_sampling=self.smote_sampling, patience=self.patience, metric=self.metric, 
                                checkpoint=False, verbose=self.verbose, save_training_data=save_training, path=self.path)
                    else:
                        if self.clf == 'alexnet':
                            if self.limit_search:
                                model, history = AlexNet(class_1, class_2, img_num_channels=self.img_num_channels, 
                                    normalize=self.normalize, min_pixel=min_pix, max_pixel=max_pix, val_positive=val_class_1, val_negative=val_class_2, 
                                    epochs=self.epochs, batch_size=batch_size, optimizer=optimizer, lr=lr, decay=decay, momentum=momentum, nesterov=nesterov, 
                                    beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad, loss=self.best_params['loss'], activation_conv=self.best_params['activation_conv'], 
                                    activation_dense=self.best_params['activation_dense'], conv_init=self.best_params['conv_init'], dense_init=self.best_params['dense_init'], 
                                    model_reg=self.best_params['model_reg'], pooling_1=self.best_params['pooling_1'], pooling_2=self.best_params['pooling_2'], pooling_3=self.best_params['pooling_3'], 
                                    smote_sampling=self.smote_sampling, patience=self.patience, metric=self.metric, checkpoint=False, verbose=self.verbose, save_training_data=save_training, path=self.path)
                            else:
                                model, history = AlexNet(class_1, class_2, img_num_channels=self.img_num_channels, 
                                    normalize=self.normalize, min_pixel=min_pix, max_pixel=max_pix, val_positive=val_class_1, val_negative=val_class_2, 
                                    epochs=self.epochs, batch_size=batch_size, optimizer=optimizer, lr=lr, decay=decay, momentum=momentum, nesterov=nesterov, 
                                    beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad, loss=self.best_params['loss'], activation_conv=self.best_params['activation_conv'], 
                                    activation_dense=self.best_params['activation_dense'], conv_init=self.best_params['conv_init'], dense_init=self.best_params['dense_init'], model_reg=self.best_params['model_reg'], 
                                    filter_1=self.best_params['filter_1'], filter_size_1=self.best_params['filter_size_1'], strides_1=self.best_params['strides_1'], pooling_1=self.best_params['pooling_1'], pool_size_1=self.best_params['pool_size_1'], pool_stride_1=self.best_params['pool_stride_1'],
                                    filter_2=self.best_params['filter_2'], filter_size_2=self.best_params['filter_size_2'], strides_2=self.best_params['strides_2'], pooling_2=self.best_params['pooling_2'], pool_size_2=self.best_params['pool_size_2'], pool_stride_2=self.best_params['pool_stride_2'],
                                    filter_3=self.best_params['filter_3'], filter_size_3=self.best_params['filter_size_3'], strides_3=self.best_params['strides_3'], pooling_3=self.best_params['pooling_3'], pool_size_3=self.best_params['pool_size_3'], pool_stride_3=self.best_params['pool_stride_3'], 
                                    filter_4=self.best_params['filter_4'], filter_size_4=self.best_params['filter_size_4'], strides_4=self.best_params['strides_4'], 
                                    filter_5=self.best_params['filter_5'], filter_size_5=self.best_params['filter_size_5'], strides_5=self.best_params['strides_5'], 
                                    dense_neurons_1=self.best_params['dense_neurons_1'], dense_neurons_2=self.best_params['dense_neurons_2'], dropout_1=self.best_params['dropout_1'], dropout_2=self.best_params['dropout_2'],
                                    smote_sampling=self.smote_sampling, patience=self.patience, metric=self.metric, checkpoint=False, verbose=self.verbose, save_training_data=save_training, path=self.path)

                        elif self.clf == 'resnet18':
                            if self.limit_search:
                                model, history = Resnet18(class_1, class_2, img_num_channels=self.img_num_channels, 
                                    normalize=self.normalize, min_pixel=min_pix, max_pixel=max_pix, val_positive=val_class_1, val_negative=val_class_2, 
                                    epochs=self.epochs, batch_size=batch_size, optimizer=optimizer, lr=lr, decay=decay, momentum=momentum, nesterov=nesterov, 
                                    beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad, loss=self.best_params['loss'], activation_conv=self.best_params['activation_conv'], 
                                    activation_dense=self.best_params['activation_dense'], conv_init=self.best_params['conv_init'], dense_init=self.best_params['dense_init'], 
                                    model_reg=self.best_params['model_reg'], pooling=self.best_params['pooling'],
                                    smote_sampling=self.smote_sampling, patience=self.patience, metric=self.metric, checkpoint=False, verbose=self.verbose, save_training_data=save_training, path=self.path)
                            else:
                                model, history = Resnet18(class_1, class_2, img_num_channels=self.img_num_channels, 
                                    normalize=self.normalize, min_pixel=min_pix, max_pixel=max_pix, val_positive=val_class_1, val_negative=val_class_2, 
                                    epochs=self.epochs, batch_size=batch_size, optimizer=optimizer, lr=lr, decay=decay, momentum=momentum, nesterov=nesterov, 
                                    beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad, loss=self.best_params['loss'], activation_conv=self.best_params['activation_conv'], 
                                    activation_dense=self.best_params['activation_dense'], conv_init=self.best_params['conv_init'], dense_init=self.best_params['dense_init'], 
                                    model_reg=self.best_params['model_reg'], filters=self.best_params['filters'], filter_size=self.best_params['filter_size'], strides=self.best_params['strides'],  
                                    pooling=self.best_params['pooling'], pool_size=self.best_params['pool_size'], pool_stride=self.best_params['pool_stride'], block_filters_1=self.best_params['block_filters_1'], 
                                    block_filters_2=self.best_params['block_filters_2'], block_filters_3=self.best_params['block_filters_3'], block_filters_4=self.best_params['block_filters_4'], 
                                    block_filters_size=self.best_params['block_filters_size'], smote_sampling=self.smote_sampling, patience=self.patience, metric=self.metric, checkpoint=False, verbose=self.verbose, save_training_data=save_training, path=self.path)
                        
                        elif self.clf == 'vgg16':
                            if self.limit_search:
                                model, history = VGG16(class_1, class_2, img_num_channels=self.img_num_channels, 
                                    normalize=self.normalize, min_pixel=min_pix, max_pixel=max_pix, val_positive=val_class_1, val_negative=val_class_2, 
                                    epochs=self.epochs, batch_size=batch_size, optimizer=optimizer, lr=lr, decay=decay, momentum=momentum, nesterov=nesterov, 
                                    beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad, loss=self.best_params['loss'], activation_conv=self.best_params['activation_conv'], 
                                    activation_dense=self.best_params['activation_dense'], conv_init=self.best_params['conv_init'], dense_init=self.best_params['dense_init'], 
                                    model_reg=self.best_params['model_reg'], pooling_1=self.best_params['pooling_1'], pooling_2=self.best_params['pooling_2'], pooling_3=self.best_params['pooling_3'], 
                                    pooling_4=self.best_params['pooling_4'], pooling_5=self.best_params['pooling_5'], smote_sampling=self.smote_sampling, patience=self.patience, metric=self.metric, checkpoint=False, verbose=self.verbose, save_training_data=save_training, path=self.path)
                            else:
                                model, history = VGG16(class_1, class_2, img_num_channels=self.img_num_channels, 
                                    normalize=self.normalize, min_pixel=min_pix, max_pixel=max_pix, val_positive=val_class_1, val_negative=val_class_2, 
                                    epochs=self.epochs, batch_size=batch_size, optimizer=optimizer, lr=lr, decay=decay, momentum=momentum, nesterov=nesterov, 
                                    beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad, loss=self.best_params['loss'], activation_conv=self.best_params['activation_conv'], 
                                    activation_dense=self.best_params['activation_dense'], conv_init=self.best_params['conv_init'], dense_init=self.best_params['dense_init'], model_reg=self.best_params['model_reg'],                 
                                    filter_1=self.best_params['filter_1'], filter_size_1=self.best_params['filter_size_1'], strides_1=self.best_params['strides_1'], pooling_1=self.best_params['pooling_1'], pool_size_1=self.best_params['pool_size_1'], pool_stride_1=self.best_params['pool_stride_1'],
                                    filter_2=self.best_params['filter_2'], filter_size_2=self.best_params['filter_size_2'], strides_2=self.best_params['strides_2'], pooling_2=self.best_params['pooling_2'], pool_size_2=self.best_params['pool_size_2'], pool_stride_2=self.best_params['pool_stride_2'],
                                    filter_3=self.best_params['filter_3'], filter_size_3=self.best_params['filter_size_3'], strides_3=self.best_params['strides_3'], pooling_3=self.best_params['pooling_3'], pool_size_3=self.best_params['pool_size_3'], pool_stride_3=self.best_params['pool_stride_3'],
                                    filter_4=self.best_params['filter_4'], filter_size_4=self.best_params['filter_size_4'], strides_4=self.best_params['strides_4'], pooling_4=self.best_params['pooling_4'], pool_size_4=self.best_params['pool_size_4'], pool_stride_4=self.best_params['pool_stride_4'],
                                    filter_5=self.best_params['filter_5'], filter_size_5=self.best_params['filter_size_5'], strides_5=self.best_params['strides_5'], pooling_5=self.best_params['pooling_5'], pool_size_5=self.best_params['pool_size_5'], pool_stride_5=self.best_params['pool_stride_5'],
                                    dense_neurons_1=self.best_params['dense_neurons_1'], dense_neurons_2=self.best_params['dense_neurons_2'], dropout_1=self.best_params['dropout_1'], dropout_2=self.best_params['dropout_2'],
                                    smote_sampling=self.smote_sampling, patience=self.patience, metric=self.metric, checkpoint=False, verbose=self.verbose, save_training_data=save_training, path=self.path)
                        
                        elif self.clf == 'custom_cnn':
                            #Need to extract the second and third layers manually 
                            #Conv2D and Pooling Layers
                            strides_1 = pool_stride_1 = 1 
                            if self.best_params['num_conv_layers'] == 1:
                                filter_2 = filter_size_2 = strides_2 = pool_size_2 = pool_stride_2 = filter_3 = filter_size_3 = strides_3 = pool_size_3 = pool_stride_3 = 0; pooling_2 = pooling_3 = None
                            if self.best_params['num_conv_layers'] >= 2:
                                filter_2 = self.best_params['filter_2']
                                filter_size_2 = self.best_params['filter_size_2'] 
                                pooling_2 = self.best_params['pooling_2']
                                pool_size_2 = self.best_params['pool_size_2']
                                strides_2 = pool_stride_2 = 1; filter_3 = filter_size_3 = strides_3 = pool_size_3 = pool_stride_3 = 0; pooling_3 = None
                            if self.best_params['num_conv_layers'] == 3:
                                filter_3 = self.best_params['filter_3']
                                filter_size_3 = self.best_params['filter_size_3']
                                pooling_3 = self.best_params['pooling_3']
                                pool_size_3 = self.best_params['pool_size_3']
                                strides_3 = pool_stride_3 = 1
                            #Dense Layers
                            if self.best_params['num_dense_layers'] == 1:
                                dense_neurons_2 = dropout_2 = dense_neurons_3 = dropout_3 = 0
                            if self.best_params['num_dense_layers'] >= 2:
                                dense_neurons_2 = self.best_params['dense_neurons_2']
                                dropout_2 = self.best_params['dropout_2'] 
                                dense_neurons_3 = dropout_3 = 0
                            if self.best_params['num_dense_layers'] == 3:
                                dense_neurons_3 =  self.best_params['dense_neurons_3']                         
                                dropout_3 = self.best_params['dropout_3'] 

                            model, history = custom_model(class_1, class_2, img_num_channels=self.img_num_channels, 
                                normalize=self.normalize, min_pixel=min_pix, max_pixel=max_pix, val_positive=val_class_1, val_negative=val_class_2, 
                                epochs=self.epochs, batch_size=batch_size, optimizer=optimizer, lr=lr, decay=decay, momentum=momentum, nesterov=nesterov, 
                                beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad, loss=self.best_params['loss'], activation_conv=self.best_params['activation_conv'], 
                                activation_dense=self.best_params['activation_dense'], conv_init=self.best_params['conv_init'], dense_init=self.best_params['dense_init'], model_reg=self.best_params['model_reg'],
                                filter_1=self.best_params['filter_1'], filter_size_1=self.best_params['filter_size_1'], strides_1=strides_1, pooling_1=self.best_params['pooling_1'], pool_size_1=self.best_params['pool_size_1'], pool_stride_1=pool_stride_1, 
                                filter_2=filter_2, filter_size_2=filter_size_2, strides_2=strides_2, pooling_2=pooling_2, pool_size_2=pool_size_2, pool_stride_2=pool_stride_2, 
                                filter_3=filter_3, filter_size_3=filter_size_3, strides_3=strides_3, pooling_3=pooling_3, pool_size_3=pool_size_3, pool_stride_3=pool_stride_3, 
                                dense_neurons_1=self.best_params['dense_neurons_1'], dense_neurons_2=dense_neurons_2, dense_neurons_3=dense_neurons_3, 
                                dropout_1=self.best_params['dropout_1'], dropout_2=dropout_2, dropout_3=dropout_3, 
                                smote_sampling=self.smote_sampling, patience=self.patience, metric=self.metric, checkpoint=False, verbose=self.verbose, save_training_data=save_training, path=self.path)

                    models.append(model), histories.append(history)

                    try:
                        if np.isfinite(history.history['loss'][-1]) is False:
                            print(); print(f"NOTE: Training failed during fold {k} due to numerical instability!")
                    except Exception as e:    
                        print(); print(f"ERROR: Training failed during fold {k} due to error: {e}!")
                        return

            #################################

            if self.opt_cv is None:
                self.model_train_metrics = np.c_[self.history.history['binary_accuracy'], self.history.history['loss'], self.history.history['f1_score']]
                if self.val_positive is not None:
                    self.model_val_metrics = np.c_[self.history.history['val_binary_accuracy'], self.history.history['val_loss'], self.history.history['val_f1_score']]
                print('Complete! To save the final model and optimization results, call the save() method.') 
            else:
                self.model, self.history = models, histories
                self.model_train_metrics = [] 
                for i in range(100): #If more than 100 CVs then this will break 
                    try:
                        model_train_metrics = np.c_[self.history[i].history['binary_accuracy'], self.history[i].history['loss'], self.history[i].history['f1_score']]
                        self.model_train_metrics.append(model_train_metrics)
                    except:
                        break

                if self.val_positive is not None:
                    self.model_val_metrics = []
                    for i in range(100): #If more than 100 CVs then this will break 
                        try:
                            model_val_metrics = np.c_[self.history[i].history['val_binary_accuracy'], self.history[i].history['val_loss'], self.history[i].history['val_f1_score']]
                            self.model_val_metrics.append(model_val_metrics)
                        except:
                            break

                print('Complete!'); print('NOTE: Cross-validation was enabled, therefore the model and history class attribute are lists containing all. To save, call the save() method.') 

            if overwrite_training:
                self.positive_class, self.negative_class, self.val_positive, self.val_negative = class_1, class_2, val_class_1, val_class_2

            return

    def save(self, dirname=None, overwrite=False):
        """
        Saves the trained classifier in a new directory named 'MicroLIA_models', 
        as well as the imputer and the features to use, if applicable.
        
        Args:
            dirname (str): The name of the directory where the model folder will be saved.
                This directory will be created, and therefore if it already exists
                in the system an error will appear.
            overwrite (bool, optional): If True the 'MicroLIA_cnn_model' folder this
                function creates in the specified path will be deleted if it exists
                and created anew to avoid duplicate files. 
        """

        if self.model is None:
            print('The model has not been created!')

        path = str(Path.home()) if self.path is None else self.path
        path += '/' if path[-1] != '/' else ''

        if dirname is not None:
            if dirname[-1] != '/':
                dirname+='/'
            path = path+dirname
            try:
                os.makedirs(path)
            except FileExistsError:
                raise ValueError('The dirname folder already exists!')

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
            if isinstance(self.model, list) is False:
                save_model(self.model, path+'Keras_Model.h5')#,  custom_objects={'f1_score': f1_score})
                np.savetxt(path+'model_train_metrics', np.c_[self.history.history['binary_accuracy'], self.history.history['loss'], self.history.history['f1_score']], header='binary_accuracy\tloss\tf1_score')
                if self.val_positive is not None:
                    np.savetxt(path+'model_val_metrics', np.c_[self.history.history['val_binary_accuracy'], self.history.history['val_loss'], self.history.history['val_f1_score']], header='val_binary_accuracy\tval_loss\tval_f1_score')
            else:
                for counter in range(len(self.model)):
                    save_model(self.model[counter], path+'Keras_Model_CV_'+str(counter+1)+'.h5')#,  custom_objects={'f1_score': f1_score})
                    np.savetxt(path+'model_train_metrics_CV_'+str(counter+1), np.c_[self.history[counter].history['binary_accuracy'], self.history[counter].history['loss'], self.history[counter].history['f1_score']], header='binary_accuracy\tloss\tf1_score')
                    if self.val_positive is not None:
                        np.savetxt(path+'model_val_metrics_CV_'+str(counter+1), np.c_[self.history[counter].history['val_binary_accuracy'], self.history[counter].history['val_loss'], self.history[counter].history['val_f1_score']], header='val_binary_accuracy\tval_loss\tval_f1_score')

        if self.best_params is not None:
            joblib.dump(self.best_params, path+'Best_Params')
        if self.optimization_results is not None:
            joblib.dump(self.optimization_results, path+'HyperOpt_Results')

        try:
            #Save all class attributes except the ones that are generated during the routine, as these are saved above
            exclude_attrs = ['positive_class', 'negative_class', 'val_positive', 
                             'val_negative', 'model', 'history', 'best_params', 
                             'optimization_results']
            attrs_dict = {attr: getattr(self, attr) for attr in dir(self) 
                          if not callable(getattr(self, attr)) and 
                          not attr.startswith("__") and 
                          attr not in exclude_attrs}
            joblib.dump(attrs_dict, path + 'class_attributes.pkl')
            print('Succesfully saved all class attributes!')
        except Exception as e:
            print(f"Could not save all class attributes to {path} due to error: {e}")

        print('Files saved in: {}'.format(path))
        self.path = path

        return 

    def load(self, path=None, load_training_data=False):
        """ 
        Loads the model, imputer, and feats to use, if created and saved.
        This function will look for a folder named 'MicroLIA_models' in the
        local home directory, unless a path argument is set. 

        Args:
            load_training_data (bool): If True the training data and validation data
                will be loaded if found the directory.
        """

        path = str(Path.home()) if path is None else path
        path += '/' if path[-1] != '/' else ''
        path += 'MicroLIA_cnn_model/'

        try:
            attrs_dict = joblib.load(path + 'class_attributes.pkl')
            for attr, value in attrs_dict.items():
                setattr(self, attr, value)
            class_attributes = ', class_attributes'
        except:
            class_attributes = ''

        try:
            self.model = load_model(path+'Keras_Model.h5', compile=False) #custom_objects={'f1_score': f1_score, 'loss': loss})
            model = 'model'
        except:
            try:
                self.model = []
                for i in range(1, 101): #If more than 100 CVs then this will break 
                    try:
                        model = load_model(path+'Keras_Model_CV_'+str(i)+'.h5', compile=False) #custom_objects={'f1_score': f1_score, 'loss': loss})
                        self.model.append(model)
                    except:
                        break

                if len(self.model) >= 1:
                    model = 'models'
                else:
                    print('Could not load models!')
                    model = ''
            except:
                print('Could not load model!')
                model = ''

        try:
            self.model_train_metrics = np.loadtxt(path+'model_train_metrics')
            train_metrics = ', training_history'
        except:
            try:
                self.model_train_metrics = [] 
                for i in range(1, 101): #If more than 100 CVs then this will break 
                    try:
                        model_train_metrics = np.loadtxt(path+'model_train_metrics_CV_'+str(i)) 
                        self.model_train_metrics.append(model_train_metrics)
                    except:
                        continue

                if len(self.model_train_metrics) >= 1:
                    train_metrics = ', training_histories'
                else:
                    print('Could not load training histories!')
                    train_metrics = ''
            except:
                print('Could not load training history!')
                train_metrics = ''

        try:
            self.model_val_metrics = np.loadtxt(path+'model_val_metrics')
            val_metrics = ', val_training_history'
        except:
            try:
                self.model_val_metrics = []
                for i in range(1, 101): #If more than 100 CVs then this will break 
                    try:
                        model_val_metrics = np.loadtxt(path+'model_val_metrics_CV_'+str(i)) 
                        self.model_val_metrics.append(model_val_metrics)
                    except:
                        continue

                if len(self.model_val_metrics) >= 1:
                    val_metrics = ', val_training_histories'
                else:
                    print('Could not load validation training histories!')
                    val_metrics = ''
            except:
                print('Could not load training history!')
                val_metrics = ''

        try:
            self.optimization_results = joblib.load(path+'HyperOpt_Results')
            optimization_results = ', optimization_results'
        except:
            optimization_results = '' 

        try:
            self.best_params = joblib.load(path+'Best_Params')
            best_params = ', best_params'
        except:
            best_params = '' 

        if load_training_data:
            if self.opt_cv is None:
                print('IMPORTANT: If re-creating the model with loaded data, set opt_aug=False and normalize=False to avoid re-augmenting and re-normalizing the loaded data!')
            else:
                print('IMPORTANT: If re-creating the model with loaded data, set opt_aug=False and normalize=False to avoid re-augmenting and re-normalizing the loaded data! Also, set opt_cv=None if the training data has been augmented!')
            
            try:
                self.positive_class = np.load(path+'class_1.npy')
                positive_class = ', positive_class'
            except:
                positive_class = ''

            try:
                self.negative_class = np.load(path+'class_2.npy')
                negative_class = ', negative_class'
            except:
                negative_class = ''

            try:
                self.val_positive = np.load(path+'val_class_1.npy')
                val_positive = ', val_positive'
            except:
                val_positive = ''

            try:
                self.val_negative = np.load(path+'val_class_2.npy')
                val_negative = ', val_negative'
            except:
                val_negative = ''

            print('Successfully loaded the following: {}{}{}{}{}{}{}{}{}{}'.format(model, train_metrics, val_metrics, optimization_results, best_params, class_attributes, positive_class, negative_class, val_positive, val_negative))
        else:
            print('Successfully loaded the following: {}{}{}{}{}{}'.format(model, train_metrics, val_metrics, optimization_results, best_params, class_attributes))

        self.path = path

        return

    def predict(self, data, target='ML', return_proba=False, cv_model=0):
        """
        Returns the class prediction. The input can either be a single 2D array 
        or a 3D array if there are multiple samples.

        Args:
            data: 2D array for single image, 3D array for multiple images.
            target (str): The name of the target class, assuming binary classification in 
                which there is an 'OTHER' class. Defaults to 'ML'. 
            return_proba (bool): If True the output will return the probability prediction.
                Defaults to False. 
            cv_model (int): Index of the model to use. Only applicable if the model class
                attribute is a list containing multiple models due to cross-validation.
                Defaults to 0, the first model in the list. Can be set to 'all', in which case
                all models will be used and an averaged prediction will be output.

        Returns:
            The class prediction(s).
        """

        data = process_class(data, normalize=self.normalize, min_pixel=self.min_pixel, max_pixel=self.max_pixel, img_num_channels=self.img_num_channels)
        if self.normalize:
            data[data > 1] = 1; data[data < 0] = 0

        model = self.model[0] if isinstance(self.model, list) else self.model 
        image_size = model.layers[0].input_shape[1:][0]

        if data.shape[1] != image_size:
            if data.shape[1] < image_size:
                raise ValueError('Model requires images of size {}, but the input images are size {}!'.format(image_size, data.shape[1]))
            print('Incorrect image size, the model requires size {}, resizing...'.format(image_size))
            data = resize(data, image_size)
    

        if isinstance(self.model, list) is False or isinstance(cv_model, int):

            model = self.model[cv_model] if isinstance(self.model, list) else self.model
            predictions = model.predict(data)

            output, probas = [], [] 
            for i in range(len(predictions)):
                if np.argmax(predictions[i]) == 1:
                    prediction = target
                    probas.append(predictions[i][1])
                else:
                    prediction = 'OTHER'
                    probas.append(predictions[i][0])
                output.append(prediction)

            output = np.c_[output, probas] if return_proba else np.array(output)
            
        else: #cv_model='all' 

            model_outputs, model_probas = [], []
            for __model__ in self.model:

                predictions = __model__.predict(data)

                output, probas = [], []                 
                for i in range(len(predictions)):
                    if np.argmax(predictions[i]) == 1:
                        prediction = target
                        probas.append(predictions[i][1])
                    else:
                        prediction = 'OTHER'
                        probas.append(predictions[i][0])
                    output.append(prediction)

                model_outputs.append(output); model_probas.append(probas)

            average_output, average_proba = [], [] 
            for j in range(len(model_outputs[0])):
                column = [model_outputs[i][j] for i in range(len(model_outputs))]
                avg_output = target if column.count(target) >= column.count('OTHER') else 'OTHER'
                avg_proba = np.mean([model_probas[i][j] for i in range(len(model_probas))])

                average_output.append(avg_output); average_proba.append(avg_proba)

            output = np.c_[average_output, average_proba] if return_proba else np.array(average_output)
            
        return output

    def augment_positive(self, batch=1, width_shift=0, height_shift=0, horizontal=False, vertical=False, 
        rotation=False, fill='nearest', image_size=None, zoom_range=None, mask_size=None, num_masks=None, 
        blend_multiplier=0, blending_func='mean', num_images_to_blend=2, skew_angle=0):
        """
        Method to augment the positive class, requires all manual inputs!

        Args:
            batch (int): How many augmented images to create and save. Defaults to 1.
            width_shift (int): The max pixel shift allowed in either horizontal direction.
                If set to zero no horizontal shifts will be performed. Defaults to 0 pixels.
            height_shift (int): The max pixel shift allowed in either vertical direction.
                If set to zero no vertical shifts will be performed. Defaults to 0 pixels.
            horizontal (bool): If False no horizontal flips are allowed. Defaults to False.
            vertical (bool): If False no vertical reflections are allowed. Defaults to False.
            rotation (int): If True full 360 rotation is allowed, if False no rotation is performed.
                Defaults to False.
            fill (str): This is the treatment for data outside the boundaries after roration
                and shifts. Default is set to 'nearest' which repeats the closest pixel values.
                Can be set to: {"constant", "nearest", "reflect", "wrap"}.
            image_size (int, bool): The length/width of the cropped image. This can be used to remove
                anomalies caused by the fill (defaults to 50). This can also be set to None in which case 
                the image in its original size is returned.
            mask_size (int): The size of the cutout mask. Defaults to None to disable random cutouts.
            num_masks (int): Number of masks to apply to each image. Defaults to None, must be an integer
                if mask_size is used as this designates how many masks of that size to randomly place in the image.
            blend_multiplier (float): Sets the amount of synthetic images to make via image blending.
                Must be a ratio greater than or equal to 1. If set to 1, the data will be replaced with
                randomly blended images, if set to 1.5, it will increase the training set by 50% with blended images,
                and so forth. Deafults to 0 which disables this feature.
            blending_func (str): The blending function to use. Options are 'mean', 'max', 'min', and 'random'. 
                Only used when blend_multiplier >= 1. Defaults to 'mean'.
            num_images_to_blend (int): The number of images to randomly select for blending. Only used when 
                blend_multiplier >= 1. Defaults to 2.
            zoom_range (tuple): Tuple of floats (min_zoom, max_zoom) specifying the range of zoom in/out values.
                If set to (0.9, 1.1), for example, the zoom will be randomly chosen between 90% to 110% the original 
                image size, note that the image size thus increases if the randomly selected zoom is greater than 1,
                therefore it is recommended to also input an appropriate image_size. Defaults to None, which disables this procedure.
            skew_angle (float): The maximum absolute value of the skew angle, in degrees. This is the maximum because 
                the actual angle to skew by will be chosen from a uniform distribution between the negative and positive 
                skew_angle values. Defaults to 0, which disables this feature.
        """

        #The augmentation function takes in each channel as individual inputs
        if self.img_num_channels == 1:
            channel1, channel2, channel3 = self.positive_class, None, None 
        elif self.img_num_channels == 2:
            channel1, channel2, channel3 = self.positive_class[:,:,:,0], self.positive_class[:,:,:,1], None 
        elif self.img_num_channels == 3:
            channel1, channel2, channel3 = self.positive_class[:,:,:,0], self.positive_class[:,:,:,1], self.positive_class[:,:,:,2]
        
        self.positive_class = augmentation(channel1, channel2, channel3, batch=batch, width_shift=width_shift, height_shift=height_shift, 
            horizontal=horizontal, vertical=vertical, rotation=rotation, fill=fill, image_size=image_size, zoom_range=zoom_range, 
            mask_size=mask_size, num_masks=num_masks, blend_multiplier=blend_multiplier, blending_func=blending_func, num_images_to_blend=num_images_to_blend, 
            skew_angle=skew_angle, return_stacked=True)

        return 

    def augment_negative(self, batch=1, width_shift=0, height_shift=0, horizontal=False, vertical=False, 
        rotation=False, fill='nearest', image_size=None, zoom_range=None, mask_size=None, num_masks=None, 
        blend_multiplier=0, blending_func='mean', num_images_to_blend=2, skew_angle=0):
        """
        Method to augment the positive class, requires all manual inputs!

        Args:
            batch (int): How many augmented images to create and save. Defaults to 1.
            width_shift (int): The max pixel shift allowed in either horizontal direction.
                If set to zero no horizontal shifts will be performed. Defaults to 0 pixels.
            height_shift (int): The max pixel shift allowed in either vertical direction.
                If set to zero no vertical shifts will be performed. Defaults to 0 pixels.
            horizontal (bool): If False no horizontal flips are allowed. Defaults to False.
            vertical (bool): If False no vertical reflections are allowed. Defaults to False.
            rotation (int): If True full 360 rotation is allowed, if False no rotation is performed.
                Defaults to False.
            fill (str): This is the treatment for data outside the boundaries after roration
                and shifts. Default is set to 'nearest' which repeats the closest pixel values.
                Can be set to: {"constant", "nearest", "reflect", "wrap"}.
            image_size (int, bool): The length/width of the cropped image. This can be used to remove
                anomalies caused by the fill (defaults to 50). This can also be set to None in which case 
                the image in its original size is returned.
            mask_size (int): The size of the cutout mask. Defaults to None to disable random cutouts.
            num_masks (int): Number of masks to apply to each image. Defaults to None, must be an integer
                if mask_size is used as this designates how many masks of that size to randomly place in the image.
            blend_multiplier (float): Sets the amount of synthetic images to make via image blending.
                Must be a ratio greater than or equal to 1. If set to 1, the data will be replaced with
                randomly blended images, if set to 1.5, it will increase the training set by 50% with blended images,
                and so forth. Deafults to 0 which disables this feature.
            blending_func (str): The blending function to use. Options are 'mean', 'max', 'min', and 'random'. 
                Only used when blend_multiplier >= 1. Defaults to 'mean'.
            num_images_to_blend (int): The number of images to randomly select for blending. Only used when 
                blend_multiplier >= 1. Defaults to 2.
            zoom_range (tuple): Tuple of floats (min_zoom, max_zoom) specifying the range of zoom in/out values.
                If set to (0.9, 1.1), for example, the zoom will be randomly chosen between 90% to 110% the original 
                image size, note that the image size thus increases if the randomly selected zoom is greater than 1,
                therefore it is recommended to also input an appropriate image_size. Defaults to None, which disables this procedure.
            skew_angle (float): The maximum absolute value of the skew angle, in degrees. This is the maximum because 
                the actual angle to skew by will be chosen from a uniform distribution between the negative and positive 
                skew_angle values. Defaults to 0, which disables this feature.
        """

        #The augmentation function takes in each channel as individual inputs
        if self.img_num_channels == 1:
            channel1, channel2, channel3 = self.negative_class, None, None 
        elif self.img_num_channels == 2:
            channel1, channel2, channel3 = self.negative_class[:,:,:,0], self.negative_class[:,:,:,1], None 
        elif self.img_num_channels == 3:
            channel1, channel2, channel3 = self.negative_class[:,:,:,0], self.negative_class[:,:,:,1], self.negative_class[:,:,:,2]
        
        self.negative_class = augmentation(channel1, channel2, channel3, batch=batch, width_shift=width_shift, height_shift=height_shift, 
            horizontal=horizontal, vertical=vertical, rotation=rotation, fill=fill, image_size=image_size, zoom_range=zoom_range, 
            mask_size=mask_size, num_masks=num_masks, blend_multiplier=blend_multiplier, blending_func=blending_func, num_images_to_blend=num_images_to_blend, 
            skew_angle=skew_angle, return_stacked=True)

        return 

    def plot_tsne(self, legend_loc='upper center', title='Feature Parameter Space', savefig=False):
        """
        Plots a t-SNE projection using the sklearn.manifold.TSNE() method.

        Note:
            Data must be normalized (0 to 1 or -1 to 1) otherwise the scaling will be off!
            If you wish to save the normalized data your models train with, set
            save_training_data=True when running the create() method as this will save
            the training data right before it is input into the model, as the data will be
            normalized at that point.
    
        Args:
            legend_loc (str): Location of legend, using matplotlib style.
            title (str): Title of the figure.
            savefig (bool): If True the figure will not disply but will be saved instead.
                Defaults to False. 

        Returns:
            AxesImage. 
        """

        if not (hasattr(self, 'positive_class') and hasattr(self, 'negative_class')):
            raise ValueError('The training data is missing! Make sure the positive_class and negative_class are input.')

        #Reshape if 3D array (single-band) -- need 4D array first.
        if len(self.positive_class.shape) == 3:
            positive_class = np.reshape(self.positive_class, (self.positive_class.shape[0], self.positive_class.shape[1], self.positive_class.shape[2], 1))
            negative_class = np.reshape(self.negative_class, (self.negative_class.shape[0], self.negative_class.shape[1], self.negative_class.shape[2], 1))
            data = np.r_[positive_class, negative_class]
            data_y = np.r_[['ML Train']*len(positive_class),['OTHER Train']*len(negative_class)]
            if self.val_positive is not None:
                val_positive = np.reshape(self.val_positive, (self.val_positive.shape[0], self.val_positive.shape[1], self.val_positive.shape[2], 1))
            if self.val_negative is not None:
                val_negative = np.reshape(self.val_negative, (self.val_negative.shape[0], self.val_negative.shape[1], self.val_negative.shape[2], 1))
            if self.val_positive is not None and self.val_negative is not None:
                val_data = np.r_[val_positive, val_negative]
                val_data_y = np.r_[['ML Val']*len(val_positive),['OTHER Val']*len(val_negative)]
            elif self.val_positive is not None and self.val_negative is None:
                val_data = val_positive
                val_data_y = np.r_[['ML Val']*len(val_data)]
            elif self.val_positive is None and self.val_negative is not None:
                val_data = val_negative
                val_data_y = np.r_[['OTHER Val']*len(val_data)]
            else:
                val_data = val_data_y = None 
        else:
            data = np.r_[self.positive_class, self.negative_class]
            data_y = np.r_[['ML Train']*len(self.positive_class),['OTHER Train']*len(self.negative_class)]
            if self.val_positive is not None and self.val_negative is not None:
                val_data = np.r_[self.val_positive, self.val_negative]
                val_data_y = np.r_[['ML Val']*len(self.val_positive),['OTHER Val']*len(self.val_negative)]
            elif self.val_positive is not None and self.val_negative is None:
                val_data = self.val_positive
                val_data_y = np.r_[['ML Val']*len(val_data)]
            elif self.val_positive is None and self.val_negative is not None:
                val_data = self.val_negative
                val_data_y = np.r_[['OTHER Val']*len(val_data)]
            else:
                val_data = val_data_y = None 

        if val_data is not None:
            data = np.r_[data, val_data]
            data_y = np.r_[data_y, val_data_y]

        #Assuming img_array is a 4D array of shape, which is the standard input for CNN models
        num_images, image_size = data.shape[0], data.shape[1]*data.shape[2]*data.shape[3]
        #Flatten each image to a 1D array
        data_x = np.reshape(data, (num_images, image_size))
        #print(flattened_images.shape)
        #Concatenate the flattened images along the first axis, this can now be input into ensemble algorithms like XGBoost
        #data_x = np.concatenate(flattened_images, axis=0)

        if len(data_x) > 5e3:
            method = 'barnes_hut' #Scales with O(N)
        else:
            method = 'exact' #Scales with O(N^2)
        print(data_x.shape)
        feats = TSNE(n_components=2, method=method, learning_rate=1000, 
            perplexity=35, init='random').fit_transform(data_x)
        x, y = feats[:,0], feats[:,1]

        markers = ['o', 's', '+', 'v', '.', 'x', 'h', 'p', '<', '>', '*']
        color = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']

        feats = np.unique(data_y)

        for count, feat in enumerate(feats):
            marker = markers[count % len(markers)]  # Wrap around the markers list
            color_val = color[count % len(color)]  # Wrap around the color list
            mask = np.where(data_y == feat)[0]
            plt.scatter(x[mask], y[mask], marker=marker, c=color_val, label=str(feat), alpha=0.44)
        """
        markers = ['o', 's', '+', 'v', '.', 'x', 'h', 'p', '<', '>', '*']
        #color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c']
        color = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']

        feats = np.unique(data_y) 

        for count, feat in enumerate(feats):
            if count+1 > len(markers):
                count = -1
            mask = np.where(data_y == feat)[0]
            plt.scatter(x[mask], y[mask], marker=markers[count], c=color[count], label=str(feat), alpha=0.44)
        """
        plt.legend(loc=legend_loc, ncol=len(np.unique(data_y)), frameon=False, handlelength=2)#prop={'size': 14}
        plt.title(title)#, size=18)
        plt.xticks()#fontsize=14)
        plt.yticks()#fontsize=14)
        plt.ylabel('t-SNE Dimension 1')
        plt.xlabel('t-SNE Dimension 2')

        if savefig:
            _set_style_()
            plt.savefig('Images_tSNE_Projection.png', bbox_inches='tight', dpi=300)
            plt.clf(); plt.style.use('default')
        else:
            plt.show()

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

        if self.metric == 'val_accuracy' or self.metric == 'val_binary_accuracy':
            ylabel = 'Validation Accuracy'
        elif self.metric == 'accuracy' or self.metric == 'acc':
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
            _set_style_()
            plt.savefig('CNN_Hyperparameter_Optimization.png', bbox_inches='tight', dpi=300)
            plt.clf(); plt.style.use('default')
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
                    raise ValueError('Could not find the importance file in the '+self.path+' directory')

                try:
                    duration_importances = joblib.load(self.path+'Duration_Importance')
                except FileNotFoundError:
                    raise ValueError('Could not find the importance file in the '+self.path+' directory')
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
            _set_style_()
            if plot_time:
                plt.savefig('CNN_Hyperparameter_Importance.png', bbox_inches='tight', dpi=300)
            else:
                plt.savefig('CNN_Hyperparameter_Duration_Importance.png', bbox_inches='tight', dpi=300)
            plt.clf(); plt.style.use('default')
        else:
            plt.show()

    def save_hyper_importance(self):
        """
        Calculates and saves binary files containing dictionaries with importance information, one
        for the importance and one for the duration importance

        Note:
            This procedure can be time-consuming but must be run once before the importances can be displayed. 
            This function will save two files in the model folder for future use. 

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

    def plot_performance(self, metric='acc', combine=False, cv_model=0, ylabel=None, title=None,
        xlim=None, ylim=None, xlog=False, ylog=False, legend_loc=9, savefig=False):
        """
        Plots the training/performance histories.
    
        Args:
            metric (str): Metric to plot, options are: 'acc', 'f1_score', 'loss'. Defaults to 'acc'
            combine (bool): If True the validation history will also be included, if applicable.
            ylabel (str, optional): The y-label of the plot.
            title (str, optional): The title of the plot.
            xlim (tuple, optional): The xlim range, matplotlib style.
            ylim (tuple, optional): The ylim range, matplotlib style.
            xlog (bool): Whether to log-scale the x-axis. Defaults to False.
            ylog (bool): Whether to log-scale the y-axis. Defaults to False.
            savefig (bool): If True the figure will not disply but will be saved instead. Defaults to False. 
            cv_model (int): Index of the model to use. Only applicable if the model_train_metrics class
                attribute is a list containing multiple models due to cross-validation.
                Defaults to 0, the first history object in the list. Can be set to 'all', in which case
                all histories will be used and plotted.
            legend_loc (int, str, optional): The location of the legend, using the matplotlib.pyplot conventino.
                Defaults to 0 aka 'upper center'.

        Returns:
            AxesImage
        """

        if not hasattr(self, 'model_train_metrics'):
            raise ValueError('Training history not found! Run the load() method first!')

        if combine and not hasattr(self, 'model_val_metrics'):
            raise ValueError('combine=True but no validation metrics found!')

        if metric == 'acc':
            index = 0 
        elif metric == 'loss':
            index = 1 
        elif metric == 'f1':
            index = 2
        else:
            raise ValueError('Invalid metric input! Valid options include: "acc", "loss" and "f1"')

        if isinstance(self.model_train_metrics, list) is False or isinstance(cv_model, int):
            metric1 = self.model_train_metrics[cv_model] if isinstance(self.model_train_metrics, list) else self.model_train_metrics
            metric1 = metric1[:,index]
        
            if combine:
                metric2 = self.model_val_metrics[cv_model] if isinstance(self.model_val_metrics, list) else self.model_val_metrics
                metric2 = metric2[:,index]
                label1, label2 = 'Training', 'Validation'
            else:
                label1 = 'Training'
        else: #cv_model='all'
            metric1 = []
            for _metric_ in self.model_train_metrics:
                metric1.append(_metric_[:,index])

            if combine:
                metric2 = []
                for _metric_ in self.model_val_metrics:
                    metric2.append(_metric_[:,index])
                label1, label2 = 'Training', 'Validation'
            else:
                label1 = 'Training'
        
        if cv_model == 'all':
            markers = ['o', 's', '+', 'v', '.', 'x', 'h', 'p', '<', '>', '*']
            color = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']

            for i in range(len(metric1)):
                marker = markers[i % len(markers)]  # Wrap around the markers list
                plt.plot(range(1, len(metric1[i])+1), metric1[i], color=color[i % len(color)], alpha=0.83, linestyle='-', label=label1+' CV '+str(i+1))

            if combine:
                for i in range(len(metric2)):
                    marker = markers[i % len(markers)]  # Wrap around the markers list
                    plt.plot(range(1, len(metric2[i])+1), metric2[i], color=color[i % len(color)], alpha=0.83, linestyle='--', label=label2+' CV '+str(i+1))
                plt.legend(loc=legend_loc, frameon=False, ncol=2)
            else:
                plt.legend(loc=legend_loc, frameon=False)

        else:
            plt.plot(range(1, len(metric1)+1), metric1, color='r', alpha=0.83, linestyle='-', label=label1)
            if combine:
                plt.plot(range(1, len(metric2)+1), metric2, color='b', alpha=0.83, linestyle='--', label=label2)
                plt.legend(loc=legend_loc, frameon=False, ncol=2)
            else:
                plt.legend(loc=legend_loc, frameon=False)

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
            if cv_model == 'all':
                len_ = []
                for _metric_ in self.model_train_metrics:
                    len_.append(len(_metric_))
                plt.xlim(1, np.max(len_))
            else:
                plt.xlim((1, len(metric1)))
        if ylim is not None:
            plt.ylim(ylim)
        if xlog:
            plt.xscale('log')
        if ylog:
            plt.yscale('log')

        plt.rcParams['axes.facecolor']='white'
        if savefig:
            _set_style_()
            plt.savefig('CNN_Training_History_'+metric+'.png', bbox_inches='tight', dpi=300)
            plt.clf(); plt.style.use('default')
        else:
            plt.show()

    def _plot_positive(self, index=0, channel=0, default_scale=True, vmin=None, vmax=None, cmap='gray', title=''):
        """
        Plots the sample in the ``positive`` class, located an the specified index.

        The channel parameter determines what filter to display, must be less than
        or equal to the ``img_num_channels`` - 1, or 'all', to plot a colorized image.

        The plotting procedure employs the matplotlib imshow display, with a robust
        vmin and vmax, unless these are set as arguments.
    
        Args:
            index (int): The index of the sample to be displayed. Defaults to 0.
            channel (int): The channel to plot, can be 0, 1, 2, or 'all'. Defaults to 0.  
            default_scale (bool): If True the figure will be generated using the matplotlib 
                imshow display, using the default scaling. If False, the vmin and vmax arguments must
                be input, otherwise a robust vmin and vmax will be calculated. Defaults to True. 
            vmin (float): The vmin to control the colorbar scaling.
            vmax (float): The vmax to control the colorbar scaling. 
            cmap (str): Colormap to use when generating the image.
            title (str, optional): Title displayed above the image. 

        Returns:
            AxesImage.
        """

        if len(self.positive_class.shape) == 3:
            data = self.positive_class.reshape(self.positive_class.shape[0], self.positive_class.shape[1], self.positive_class.shape[2], 1)
            data = data[index]
        else:
            data = self.positive_class[index]

        if channel == 'all':
            if vmin is None and default_scale is False:
                plot(data)
            else:
                if default_scale is False:
                    plt.imshow(data[:,:,channel], vmin=vmin, vmax=vmax, cmap=cmap); plt.title(title); plt.show()
                else:
                    plt.imshow(data[:,:,channel], cmap=cmap); plt.title(title); plt.show()
  
            return 

        if vmin is None and default_scale is False:
            plot(data[:,:,channel])
        else:   
            if default_scale is False:
                plt.imshow(data[:,:,channel], vmin=vmin, vmax=vmax, cmap=cmap); plt.title(title); plt.show()
            else:
                plt.imshow(data[:,:,channel], cmap=cmap); plt.title(title); plt.show()
          
        return

    def _plot_negative(self, index=0, channel=0, default_scale=True, vmin=None, vmax=None, cmap='gray', title=''):
        """
        Plots the sample in the ``negative`` class, located an the specified index.

        The channel parameter determines what filter to display, must be less than
        or equal to the ``img_num_channels`` - 1, or 'all', to plot a colorized image.

        The plotting procedure employs the matplotlib imshow display, with a robust
        vmin and vmax, unless these are set as arguments.
    
        Args:
            index (int): The index of the sample to be displayed. Defaults to 0.
            channel (int): The channel to plot, can be 0, 1, 2, or 'all'. Defaults to 0.  
            default_scale (bool): If True the figure will be generated using the matplotlib 
                imshow display, using the default scaling. If False, the vmin and vmax arguments must
                be input, otherwise a robust vmin and vmax will be calculated. Defaults to True. 
            cmap (str): Colormap to use when generating the image.
            vmin (float): The vmin to control the colorbar scaling.
            vmax (float): The vmax to control the colorbar scaling. 
            title (str, optional): Title displayed above the image. 

        Returns:
            AxesImage.
        """

        if len(self.negative_class.shape) == 3:
            data = self.negative_class.reshape(self.negative_class.shape[0], self.negative_class.shape[1], self.negative_class.shape[2], 1)
            data = data[index]
        else:
            data = self.negative_class[index]

        if channel == 'all':
            if vmin is None and default_scale is False:
                plot(data)
            else:
                if default_scale is False:
                    plt.imshow(data[:,:,channel], vmin=vmin, vmax=vmax, cmap=cmap); plt.title(title); plt.show()
                else:
                    plt.imshow(data[:,:,channel], cmap=cmap); plt.title(title); plt.show()

            return 

        if vmin is None and default_scale is False:
            plot(data[:,:,channel])
        else:   
            if default_scale is False:
                plt.imshow(data[:,:,channel], vmin=vmin, vmax=vmax, cmap=cmap); plt.title(title); plt.show()
            else:
                plt.imshow(data[:,:,channel], vmin=vmin, vmax=vmax, cmap=cmap); plt.title(title); plt.show()

        return

#Custom CNN model configured to genereate shallower CNNs than AlexNet

def custom_model(positive_class, negative_class, img_num_channels=1, normalize=True, 
    min_pixel=0, max_pixel=100, val_positive=None, val_negative=None, epochs=100, batch_size=32, 
    optimizer='sgd', lr=0.0001, momentum=0.9, decay=0.0, nesterov=False, rho=0.9, beta_1=0.9, beta_2=0.999, amsgrad=False,
    loss='binary_crossentropy', conv_init='uniform_scaling', dense_init='TruncatedNormal',
    activation_conv='relu', activation_dense='relu', conv_reg=0, dense_reg=0, padding='same', model_reg='batch_norm',
    filter_1=256, filter_size_1=7, strides_1=1, pooling_1='average', pool_size_1=3, pool_stride_1=3, 
    filter_2=0, filter_size_2=0, strides_2=0, pooling_2=None, pool_size_2=0, pool_stride_2=0, 
    filter_3=0, filter_size_3=0, strides_3=0, pooling_3=None, pool_size_3=0, pool_stride_3=0, 
    dense_neurons_1=4096, dropout_1=0.5, dense_neurons_2=0, dropout_2=0, dense_neurons_3=0, dropout_3=0,
    smote_sampling=0, patience=0, metric='binary_accuracy', early_stop_callback=None, checkpoint=False, 
    weight=None, verbose=1, save_training_data=False, path=None):
    """
    CNN Model that allows between 1 and 3 convolutional layers (with pooling) followed by dense layers,
    also up to three layers. This is a simpler model than AlexNet that can be used to limit overfitting behavior.
    Batch normalization is hard-coded after every Conv2D layer.        

    Args:
        positive_class (ndarray): 3D array containing more than one image of the positive objects.
        negative_class (ndarray): 3D array containing more than one image of negative objects.
        img_num_channels (int): The number of filters used. Defaults to 1, as MicroLIA version 1
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
        checkpoint (bool, optional): If False no checkpoint will be saved. Defaults to True.
        verbose (int): Controls the amount of output printed during the training process. A value of 0 is for silent mode, 
            a value of 1 is used for progress bar mode, and 2 for one line per epoch mode. Defaults to 1.
        smote_sampling (float): The smote_sampling parameter is used in the SMOTE algorithm to specify the desired 
            ratio of the minority class to the majority class. Defaults to 0 which disables the procedure.
        patience (int): Number of epochs without improvement before the training is terminated. Defaults to 0, which
            disables this feature.
        metric (str): The metric to monitor according to the input patience. Defaults to 'binary_accuracy'.
        early_stop_callback (list, optional): Callbacks for early stopping and pruning with Optuna, defaults
            to None. Should only be used with the optimization routine, refer to MicroLIA.optimization.objective_cnn().
        weight (int): Weight to apply if using the weighted loss function. Defaults to None. 
        save_training_data (bool): Whether to save the training data, useful for visualizing the images as they were
            input for training. Defaults to False.
        paht (str): Path where the training data should be saved, only used if save_training_data is True.
            Defaults to None, which saves the data in the local home directory.

    Returns:
        The trained CNN model and accompanying history.
    """
    
    if batch_size < 16:
        print("Batch Normalization can be unstable with low batch sizes, if loss returns nan try a larger batch size and/or smaller learning rate.")
    
    if 'all' in metric: #This is an option for optimization purposes but not a valid argument
        if 'val' in metric:
            print("Cannot combine combined metrics for these callbacks, setting metric='val_loss'"); metric = 'val_loss'
        else:
            print("Cannot combine combined metrics for these callbacks, setting metric='loss'"); metric = 'loss'

    if val_positive is not None:
        val_X1, val_Y1 = process_class(val_positive, label=1, img_num_channels=img_num_channels, 
            min_pixel=min_pixel, max_pixel=max_pixel, normalize=normalize)
        if val_negative is None:
            val_X, val_Y = val_X1, val_Y1
        else:
            val_X2, val_Y2 = process_class(val_negative, label=0, img_num_channels=img_num_channels, min_pixel=min_pixel, max_pixel=max_pixel, normalize=normalize)
            val_X, val_Y = np.r_[val_X1, val_X2], np.r_[val_Y1, val_Y2]
        if normalize:
            val_X[val_X > 1] = 1; val_X[val_X < 0] = 0
    else:
        if val_negative is None:
            val_X, val_Y = None, None
        else:
            val_X2, val_Y2 = process_class(val_negative, label=0, img_num_channels=img_num_channels, min_pixel=min_pixel, max_pixel=max_pixel, normalize=normalize)
            val_X, val_Y = val_X2, val_Y2
        if normalize:
            val_X[val_X > 1] = 1; val_X[val_X < 0] = 0

    img_width, img_height = positive_class[0].shape[0], positive_class[0].shape[1]
     
    ix = np.random.permutation(len(positive_class))
    positive_class = positive_class[ix]

    ix = np.random.permutation(len(negative_class))
    negative_class = negative_class[ix]

    X_train, Y_train = create_training_set(positive_class, negative_class, normalize=normalize, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)
    
    if normalize:
        X_train[X_train > 1] = 1; X_train[X_train < 0] = 0
        
    #Apply SMOTE to oversample the minority class
    if smote_sampling > 0:
        X_train[np.isfinite(X_train)==False] = 0
        if len(np.where(Y_train[:,0]==1)[0]) == len(np.where(Y_train[:,1]==1)[0]):
            X_train_res, Y_train_res = X_train, Y_train
            print('Classes are already balanced, skipping SMOTE sampling.')
        else:
            X_train_res, Y_train_res = smote_oversampling(X_train, Y_train, smote_sampling=smote_sampling)
    elif smote_sampling == 0:
        X_train_res, Y_train_res = X_train, Y_train
    else:
        raise ValueError('smote_sampling must be a float between 0.0 and 1.0!')

    num_classes, input_shape = 2, (img_width, img_height, img_num_channels)

    if verbose == 1:
        filter_size_4 = filter_size_5 = filter_4 = filter_5 = 0; pooling_4 = pool_size_4 = pooling_5 = pool_size_5 ='None' 
        print_params(batch_size, lr, decay, momentum, nesterov, loss, optimizer, model_reg, conv_init, activation_conv, 
            dense_init, activation_dense, filter_1, filter_2, filter_3, filter_4, filter_5, filter_size_1, filter_size_2, 
            filter_size_3, filter_size_4, filter_size_5, pooling_1, pooling_2, pooling_3, pooling_4, pooling_5, pool_size_1, 
            pool_size_2, pool_size_3, pool_size_4, pool_size_5, conv_reg, dense_reg, dense_neurons_1, dense_neurons_2, 
            dense_neurons_3, dropout_1, dropout_2, dropout_3, beta_1, beta_2, amsgrad, rho)

    #Uniform scaling initializer
    conv_init = VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform', seed=None) if conv_init == 'uniform_scaling' else conv_init
    dense_init = VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform', seed=None) if dense_init == 'uniform_scaling' else dense_init

    #Call the appropriate tf.keras.losses.Loss function
    loss = get_loss_function(loss, weight=weight)

    #Model configuration
    model = Sequential()
    
    #Convolutional layers
    model.add(Conv2D(filter_1, filter_size_1, strides=strides_1, activation=activation_conv, input_shape=input_shape, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))
    #Regularizer: batch_norm
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None
    #The Pooling Layer
    model.add(tf.keras.layers.Lambda(lambda x: -tf.nn.max_pool2d(-x, ksize=(pool_size_1, pool_size_1), strides=(pool_stride_1, pool_stride_1), padding='SAME'))) if pooling_1 == 'min' else None
    model.add(MaxPool2D(pool_size=pool_size_1, strides=pool_stride_1, padding=padding)) if pooling_1 == 'max' else None
    model.add(AveragePooling2D(pool_size=pool_size_1, strides=pool_stride_1, padding=padding)) if pooling_1 == 'average' else None
    #Regularizer: local_response, following the AlexNet convention of placing after the pooling layer
    model.add(Lambda(lambda x: tf.nn.local_response_normalization(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75))) if model_reg == 'local_response' else None
    
    if filter_2 > 0:
        if filter_size_2 is None or strides_2 is None:
            raise ValueError('Filter 2 parameters are missing, input the missing arguments.')
        model.add(Conv2D(filter_2, filter_size_2, strides=strides_2, activation=activation_conv, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))
        #Regularizer: batch_norm
        model.add(BatchNormalization()) if model_reg == 'batch_norm' else None
        #The Pooling Layer
        model.add(tf.keras.layers.Lambda(lambda x: -tf.nn.max_pool2d(-x, ksize=(pool_size_2, pool_size_2), strides=(pool_stride_2, pool_stride_2), padding='SAME'))) if pooling_2 == 'min' else None
        model.add(MaxPool2D(pool_size=pool_size_2, strides=pool_stride_2, padding=padding)) if pooling_2 == 'max' else None
        model.add(AveragePooling2D(pool_size=pool_size_2, strides=pool_stride_2, padding=padding)) if pooling_2 == 'average' else None
        #Regularizer: local_response, following the AlexNet convention of placing after the pooling layer
        model.add(Lambda(lambda x: tf.nn.local_response_normalization(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75))) if model_reg == 'local_response' else None
    
    if filter_3 > 0:
        if filter_size_3 is None or strides_3 is None:
            raise ValueError('Filter 3 parameters are missing, input the missing arguments.')
        model.add(Conv2D(filter_3, filter_size_3, strides=strides_3, activation=activation_conv, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))
        #Regularizer: batch_norm
        model.add(BatchNormalization()) if model_reg == 'batch_norm' else None
        #The Pooling Layer
        model.add(tf.keras.layers.Lambda(lambda x: -tf.nn.max_pool2d(-x, ksize=(pool_size_3, pool_size_3), strides=(pool_stride_3, pool_stride_3), padding='SAME'))) if pooling_3 == 'min' else None
        model.add(MaxPool2D(pool_size=pool_size_3, strides=pool_stride_3, padding=padding)) if pooling_3 == 'max' else None
        model.add(AveragePooling2D(pool_size=pool_size_3, strides=pool_stride_3, padding=padding)) if pooling_3 == 'average' else None
        #Regularizer: local_response, following the AlexNet convention of placing after the pooling layer
        model.add(Lambda(lambda x: tf.nn.local_response_normalization(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75))) if model_reg == 'local_response' else None
    
    #Dense layers
    model.add(Flatten())

    #FCC 1
    model.add(Dense(dense_neurons_1, activation=activation_dense, kernel_initializer=dense_init, kernel_regularizer=l2(dense_reg)))
    model.add(Dropout(dropout_1))
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None
    
    #FCC 2
    if dense_neurons_2 > 0:
        model.add(Dense(dense_neurons_2, activation=activation_dense, kernel_initializer=dense_init, kernel_regularizer=l2(dense_reg)))
        model.add(Dropout(dropout_2))
        model.add(BatchNormalization()) if model_reg == 'batch_norm' else None

    #FCC 3
    if dense_neurons_3 > 0:
        model.add(Dense(dense_neurons_3, activation=activation_dense, kernel_initializer=dense_init, kernel_regularizer=l2(dense_reg)))
        model.add(Dropout(dropout_3))
        model.add(BatchNormalization()) if model_reg == 'batch_norm' else None

    #Output layer
    model.add(Dense(num_classes, activation='sigmoid', kernel_initializer=dense_init)) 
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None

    #Call the appropriate tf.keras.optimizers function
    optimizer = get_optimizer(optimizer, lr, momentum=momentum, decay=decay, rho=rho, nesterov=nesterov, beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad)

    #Compile the Model
    model.compile(loss=loss, optimizer=optimizer, metrics=[tf.keras.metrics.BinaryAccuracy(), f1_score])
    
    #Wheter to maximize or minimize the metric
    mode = 'min' if 'loss' in metric else 'max'

    #Optional checkpoint callback, with the monitor being the input metric.
    callbacks_list = []
    callbacks_list.append(ModelCheckpoint(str(Path.home())+'/'+'checkpoint.hdf5', monitor=metric, verbose=2, save_best_only=True, mode=mode)) if checkpoint else None

    #Early stopping callback
    callbacks_list.append(EarlyStopping(monitor=metric, mode=mode, patience=patience)) if patience > 0 else None

    #Early stop callback for use during the optimization routine
    callbacks_list.append(early_stop_callback) if early_stop_callback is not None else None

    #Fit the Model
    if val_X is None:
        history = model.fit(X_train_res, Y_train_res, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, verbose=verbose)
    else:
        history = model.fit(X_train_res, Y_train_res, batch_size=batch_size, validation_data=(val_X, val_Y), epochs=epochs, callbacks=callbacks_list, verbose=verbose)

    if save_training_data:
        path = str(Path.home()) if path is None else path
        path += '/' if path[-1] != '/' else ''
        try:
            flattened_labels = np.argmax(Y_train_res, axis=1)
            ix1 = np.where(flattened_labels == 1)[0]; ix2 = np.where(flattened_labels == 0)[0]; 
            np.save(path+'class_1.npy', X_train_res[ix1]); np.save(path+'class_2.npy', X_train_res[ix2]); 
            if val_positive is not None:
                np.save(path+'val_class_1.npy', val_X1)
            if val_negative is not None:
                np.save(path+'val_class_2.npy', val_X2)     
            print('Files saved in: {}'.format(path)); print('NOTE: The training data files may have to be manually moved to the MicroLIA_cnn_model folder in order for them to be loaded when running load()!') 
        except Exception as e:
            print(f"Could not save training data due to error: {e}")

    return model, history

def AlexNet(positive_class, negative_class, img_num_channels=1, normalize=True, 
    min_pixel=0, max_pixel=100, val_positive=None, val_negative=None, epochs=100, batch_size=32, 
    optimizer='sgd', lr=0.0001, momentum=0.9, decay=0.0, nesterov=False, rho=0.9, beta_1=0.9, beta_2=0.999, amsgrad=False,
    loss='binary_crossentropy', conv_init='uniform_scaling', dense_init='TruncatedNormal',
    activation_conv='relu', activation_dense='relu', conv_reg=0, dense_reg=0, padding='same',model_reg='local_response',
    filter_1=96, filter_size_1=11, strides_1=4, pooling_1='max', pool_size_1=3, pool_stride_1=2, 
    filter_2=256, filter_size_2=5, strides_2=1, pooling_2='max', pool_size_2=3, pool_stride_2=2,
    filter_3=384, filter_size_3=3, strides_3=1, pooling_3='max', pool_size_3=3, pool_stride_3=2, 
    filter_4=384, filter_size_4=3, strides_4=1, filter_5=256, filter_size_5=3, strides_5=1, 
    dense_neurons_1=4096, dense_neurons_2=4096, dropout_1=0.5, dropout_2=0.5,  
    smote_sampling=0, patience=0, metric='binary_accuracy', early_stop_callback=None, checkpoint=False, 
    weight=None, verbose=1, save_training_data=False, path=None):
    """
    The CNN model infrastructure presented by the 2012 ImageNet Large Scale 
    Visual Recognition Challenge, AlexNet. 

    To avoid exploding gradients we need to normalize our pixels to be 
    between 0 and 1. By default normalize=True, which will perform
    min-max normalization using the min_pixel and max_pixel arguments, 
    which should be set carefully.

    The original AlexNet architecture employed the use of Local Response Normalization (LRN)
    which normalizes the responses across neighboring channels for each spatial location in a feature map, 
    which can help in improving the model's accuracy. LRN is applied after the pooling layer so as to enhance 
    the model's ability to detect small-scale features, by normalizing the outputs of the neurons that respond 
    maximally to said features. The idea was that normalization would reduce the responses of the nearby neurons 
    that are also sensitive to the same feature, thus improving the model's ability to distinguish different image
    features. On the other hand, batch normalization is applied after the convolutional layer and before the 
    activation function. This is because batch normalization normalizes the output of a convolutional layer by 
    adjusting and scaling the activations before the activation function, which helps in reducing the internal 
    covariate shift and improves the generalization performance of the model.

    Note:
        SMOTE expects a 2D input array that represents the feature space of the minority class. 
        In the case of image classification, the feature space is usually flattened to a 1D vector 
        for each image. This means that each row of the 2D input array represents a single image 
        from the minority class, and each column represents a feature (pixel) of that image.

        SMOTE expects a 2D input array because it works by computing the k-nearest neighbors of each minority 
        class sample in the feature space and generating synthetic samples by interpolating between those neighbors.

        By working in the feature space, SMOTE is able to generate synthetic samples that are similar 
        to the existing minority class samples, and therefore more representative of the true distribution 
        of the minority class. The resulting synthetic samples can then be added to the training set to 
        balance the class distribution.

        Once SMOTE has generated the synthetic samples, the 2D array can be reshaped back into its 
        original image format to be used as input to a CNN model.

    Args:
        positive_class (ndarray): 3D array containing more than one image of the positive objects.
        negative_class (ndarray): 3D array containing more than one image of negative objects.
        img_num_channels (int): The number of filters used. Defaults to 1, as MicroLIA version 1
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
        checkpoint (bool, optional): If False no checkpoint will be saved. Defaults to True.
        verbose (int): Controls the amount of output printed during the training process. A value of 0 is for silent mode, 
            a value of 1 is used for progress bar mode, and 2 for one line per epoch mode. Defaults to 1.
        smote_sampling (float): The smote_sampling parameter is used in the SMOTE algorithm to specify the desired 
            ratio of the minority class to the majority class. Defaults to 0 which disables the procedure.
        patience (int): Number of epochs without improvement before the training is terminated. Defaults to 0, which
            disables this feature.
        metric (str): The metric to monitor according to the input patience. Defaults to 'binary_accuracy'.
        early_stop_callback (list, optional): Callbacks for early stopping and pruning with Optuna, defaults
            to None. Should only be used with the optimization routine, refer to MicroLIA.optimization.objective_cnn().
        weight (int): Weight to apply if using the weighted loss function. Defaults to None. 
        save_training_data (bool): Whether to save the training data, useful for visualizing the images as they were
            input for training. Defaults to False.
        paht (str): Path where the training data should be saved, only used if save_training_data is True.
            Defaults to None, which saves the data in the local home directory.

    Returns:
        The trained CNN model and accompanying history.
    """
    
    if batch_size < 16 and model_reg == 'batch_norm':
        print("Batch Normalization can be unstable with low batch sizes, if loss returns nan try a larger batch size and/or smaller learning rate.")
    
    if 'all' in metric: #This is an option for optimization purposes but not a valid argument
        if 'val' in metric:
            print("Cannot combine combined metrics for these callbacks, setting metric='val_loss'"); metric = 'val_loss'
        else:
            print("Cannot combine combined metrics for these callbacks, setting metric='loss'"); metric = 'loss'

    if val_positive is not None:
        val_X1, val_Y1 = process_class(val_positive, label=1, img_num_channels=img_num_channels, min_pixel=min_pixel, max_pixel=max_pixel, normalize=normalize)
        if val_negative is None:
            val_X, val_Y = val_X1, val_Y1
        else:
            val_X2, val_Y2 = process_class(val_negative, label=0, img_num_channels=img_num_channels, min_pixel=min_pixel, max_pixel=max_pixel, normalize=normalize)
            val_X, val_Y = np.r_[val_X1, val_X2], np.r_[val_Y1, val_Y2]
    else:
        if val_negative is None:
            val_X, val_Y = None, None
        else:
            val_X2, val_Y2 = process_class(val_negative, label=0, img_num_channels=img_num_channels, min_pixel=min_pixel, max_pixel=max_pixel, normalize=normalize)
            val_X, val_Y = val_X2, val_Y2

    img_width, img_height = positive_class[0].shape[0], positive_class[0].shape[1]
    
    ix = np.random.permutation(len(positive_class))
    positive_class = positive_class[ix]

    ix = np.random.permutation(len(negative_class))
    negative_class = negative_class[ix]

    X_train, Y_train = create_training_set(positive_class, negative_class, normalize=normalize, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)

    if normalize:
        X_train[X_train > 1] = 1; X_train[X_train < 0] = 0

    #Apply SMOTE to oversample the minority class
    if smote_sampling > 0:
        X_train[np.isfinite(X_train)==False] = 0
        if len(np.where(Y_train[:,0]==1)[0]) == len(np.where(Y_train[:,1]==1)[0]):
            X_train_res, Y_train_res = X_train, Y_train
            print('Classes are already balanced, skipping SMOTE sampling.')
        else:
            X_train_res, Y_train_res = smote_oversampling(X_train, Y_train, smote_sampling=smote_sampling)
    elif smote_sampling == 0:
        X_train_res, Y_train_res = X_train, Y_train
    else:
        raise ValueError('smote_sampling must be a float between 0.0 and 1.0!')

    num_classes, input_shape = 2, (img_width, img_height, img_num_channels)
    
    if verbose == 1:
        dense_neurons_3 = dropout_3 = 0; pooling_4 = pool_size_4 = pooling_5 = pool_size_5 = 'None' 
        print_params(batch_size, lr, decay, momentum, nesterov, loss, optimizer, model_reg, conv_init, activation_conv, 
            dense_init, activation_dense, filter_1, filter_2, filter_3, filter_4, filter_5, filter_size_1, filter_size_2, 
            filter_size_3, filter_size_4, filter_size_5, pooling_1, pooling_2, pooling_3, pooling_4, pooling_5, pool_size_1, 
            pool_size_2, pool_size_3, pool_size_4, pool_size_5, conv_reg, dense_reg, dense_neurons_1, dense_neurons_2, 
            dense_neurons_3, dropout_1, dropout_2, dropout_3, beta_1, beta_2, amsgrad, rho)

    #Uniform scaling initializer
    conv_init = VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform', seed=None) if conv_init == 'uniform_scaling' else conv_init
    dense_init = VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform', seed=None) if dense_init == 'uniform_scaling' else dense_init

    #Call the appropriate tf.keras.losses.Loss function
    loss = get_loss_function(loss, weight=weight)

    #Model configuration
    model = Sequential()
    
    #Convolutional layers
    model.add(Conv2D(filter_1, filter_size_1, strides=strides_1, activation=activation_conv, input_shape=input_shape, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))
    #Regularizer: batch_norm, local_response, or None
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None
    #The Pooling Layer
    model.add(tf.keras.layers.Lambda(lambda x: -tf.nn.max_pool2d(-x, ksize=(pool_size_1, pool_size_1), strides=(pool_stride_1, pool_stride_1), padding='SAME'))) if pooling_1 == 'min' else None
    model.add(MaxPool2D(pool_size=pool_size_1, strides=pool_stride_1, padding=padding)) if pooling_1 == 'max' else None
    model.add(AveragePooling2D(pool_size=pool_size_1, strides=pool_stride_1, padding=padding)) if pooling_1 == 'average' else None
    #Regularizer: local_response, placed here in accordance with the original AlexNet architecture, in practice batch_norm is placed after conv2d
    model.add(Lambda(lambda x: tf.nn.local_response_normalization(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75))) if model_reg == 'local_response' else None

    model.add(Conv2D(filter_2, filter_size_2, strides=strides_2, activation=activation_conv, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))
    #Regularizer: batch_norm, local_response, or None
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None
    #The Pooling Layer
    model.add(tf.keras.layers.Lambda(lambda x: -tf.nn.max_pool2d(-x, ksize=(pool_size_2, pool_size_2), strides=(pool_stride_2, pool_stride_2), padding='SAME'))) if pooling_2 == 'min' else None
    model.add(MaxPool2D(pool_size=pool_size_2, strides=pool_stride_2, padding=padding)) if pooling_2 == 'max' else None
    model.add(AveragePooling2D(pool_size=pool_size_2, strides=pool_stride_2, padding=padding)) if pooling_2 == 'average' else None
    #Regularizer: local_response, placed here in accordance with the original AlexNet architecture, in practice batch_norm is placed after conv2d
    model.add(Lambda(lambda x: tf.nn.local_response_normalization(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75))) if model_reg == 'local_response' else None

    model.add(Conv2D(filter_3, filter_size_3, strides=strides_3, activation=activation_conv, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None

    model.add(Conv2D(filter_4, filter_size_4, strides=strides_4, activation=activation_conv, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None

    model.add(Conv2D(filter_5, filter_size_5, strides=strides_5, activation=activation_conv, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None
    #The Pooling Layer
    model.add(tf.keras.layers.Lambda(lambda x: -tf.nn.max_pool2d(-x, ksize=(pool_size_3, pool_size_3), strides=(pool_stride_3, pool_stride_3), padding='SAME'))) if pooling_3 == 'min' else None
    model.add(MaxPool2D(pool_size=pool_size_3, strides=pool_stride_3, padding=padding)) if pooling_3 == 'max' else None
    model.add(AveragePooling2D(pool_size=pool_size_3, strides=pool_stride_3, padding=padding)) if pooling_3 == 'average' else None
    
    #Dense layers
    model.add(Flatten())

    #FCC 1
    model.add(Dense(dense_neurons_1, activation=activation_dense, kernel_initializer=dense_init, kernel_regularizer=l2(dense_reg)))
    model.add(Dropout(dropout_1))
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None

    #FCC 2
    model.add(Dense(dense_neurons_2, activation=activation_dense, kernel_initializer=dense_init, kernel_regularizer=l2(dense_reg)))
    model.add(Dropout(dropout_2))
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None

    #Output layer
    model.add(Dense(num_classes, activation='sigmoid', kernel_initializer=dense_init)) #adding a kernel model_reg has no effect if sigmoid is being used, but works for layers with trainable weights like softmax
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None

    #Call the appropriate tf.keras.optimizers function
    optimizer = get_optimizer(optimizer, lr, momentum=momentum, decay=decay, rho=rho, nesterov=nesterov, beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad)

    #Compile the Model
    model.compile(loss=loss, optimizer=optimizer, metrics=[tf.keras.metrics.BinaryAccuracy(), f1_score])
    
    #Wheter to maximize or minimize the metric
    mode = 'min' if 'loss' in metric else 'max'

    #Optional checkpoint callback, with the monitor being the input metric.
    callbacks_list = []
    callbacks_list.append(ModelCheckpoint(str(Path.home())+'/'+'checkpoint.hdf5', monitor=metric, verbose=2, save_best_only=True, mode=mode)) if checkpoint else None

    #Early stopping callback
    callbacks_list.append(EarlyStopping(monitor=metric, mode=mode, patience=patience)) if patience > 0 else None

    #Early stop callback for use during the optimization routine
    callbacks_list.append(early_stop_callback) if early_stop_callback is not None else None

    #Fit the Model
    if val_X is None:
        history = model.fit(X_train_res, Y_train_res, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, verbose=verbose)
    else:
        history = model.fit(X_train_res, Y_train_res, batch_size=batch_size, validation_data=(val_X, val_Y), epochs=epochs, callbacks=callbacks_list, verbose=verbose)

    if save_training_data:
        path = str(Path.home()) if path is None else path
        path += '/' if path[-1] != '/' else ''
        try:
            flattened_labels = np.argmax(Y_train_res, axis=1)
            ix1 = np.where(flattened_labels == 1)[0]; ix2 = np.where(flattened_labels == 0)[0]; 
            np.save(path+'class_1.npy', X_train_res[ix1]); np.save(path+'class_2.npy', X_train_res[ix2]); 
            if val_positive is not None:
                np.save(path+'val_class_1.npy', val_X1)
            if val_negative is not None:
                np.save(path+'val_class_2.npy', val_X2)     
            print('Files saved in: {}'.format(path)); print('NOTE: The training data files may have to be manually moved to the MicroLIA_cnn_model folder in order for them to be loaded when running load()!') 
        except Exception as e:
            print(f"Could not save training data due to error: {e}")

    return model, history

def VGG16(positive_class, negative_class, img_num_channels=1, normalize=True, 
    min_pixel=0, max_pixel=100, val_positive=None, val_negative=None, epochs=100, batch_size=32, 
    optimizer='sgd', lr=0.0001, momentum=0.9, decay=0.0, nesterov=False, rho=0.9, beta_1=0.9, beta_2=0.999, amsgrad=False,
    loss='binary_crossentropy', conv_init='uniform_scaling', dense_init='TruncatedNormal',
    activation_conv='relu', activation_dense='relu', conv_reg=0, dense_reg=0, padding='same',model_reg=None,
    filter_1=64, filter_size_1=3, strides_1=1, pooling_1='max', pool_size_1=2, pool_stride_1=2,
    filter_2=128, filter_size_2=3, strides_2=1, pooling_2='max',  pool_size_2=2, pool_stride_2=2,
    filter_3=256, filter_size_3=3, strides_3=1, pooling_3='max', pool_size_3=2, pool_stride_3=2,
    filter_4=512, filter_size_4=3, strides_4=1, pooling_4='max', pool_size_4=2, pool_stride_4=2,
    filter_5=512, filter_size_5=3, strides_5=1, pooling_5='max', pool_size_5=2, pool_stride_5=2,
    dense_neurons_1=4096, dense_neurons_2=4096, dropout_1=0.5, dropout_2=0.5,
    smote_sampling=0, patience=0, metric='binary_accuracy', early_stop_callback=None, checkpoint=False, 
    weight=None, verbose=1, save_training_data=False, path=None):
    """
    Trains a VGG16 model, which is a convolutional neural network (CNN) architecture developed by the Visual Geometry Group (VGG) 
    at the University of Oxford. It was presented in the 2014 ImageNet Large Scale Visual Recognition Challenge (ILSVRC) and 
    achieved state-of-the-art results on the classification task.

    The VGG16 network consists of 16 layers, including 13 convolutional layers and 3 fully connected layers. 
    The input to the network is a RGB image of size 224x224 pixels. The first layers of the network are convolutional 
    layers with small 3x3 filters, followed by a max pooling layer with a 2x2 filter (repeated 5 times).

    Args:
        positive_class (ndarray): 3D array containing more than one image of the positive objects.
        negative_class (ndarray): 3D array containing more than one image of the negative objects.
        img_num_channels (int): The number of filters used. Defaults to 1, as MicroLIA version 1
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
        checkpoint (bool, optional): If False no checkpoint will be saved. Defaults to True.
        verbose (int): Controls the amount of output printed during the training process. A value of 0 is for silent mode, 
            a value of 1 is used for progress bar mode, and 2 for one line per epoch mode. Defaults to 1.
        smote_sampling (float): The smote_sampling parameter is used in the SMOTE algorithm to specify the desired 
            ratio of the minority class to the majority class. Defaults to 0 which disables the procedure.
        patience (int): Number of epochs without improvement before the training is terminated. Defaults to 0, which
            disables this feature.
        metric (str): The metric to monitor according to the input patience. Defaults to 'binary_accuracy'.
        early_stop_callback (list, optional): Callbacks for early stopping and pruning with Optuna, defaults
            to None. Should only be used with the optimization routine, refer to MicroLIA.optimization.objective_cnn().
        weight (int): Weight to apply if using the weighted loss function. Defaults to None. 
        save_training_data (bool): Whether to save the training data, useful for visualizing the images as they were
            input for training. Defaults to False.
        paht (str): Path where the training data should be saved, only used if save_training_data is True.
            Defaults to None, which saves the data in the local home directory.

    Returns:
        The trained CNN model and accompanying history.
    """

    if batch_size < 16 and model_reg == 'batch_norm':
        print("Batch Normalization can be unstable with low batch sizes, if loss returns nan try a larger batch size and/or smaller learning rate.")
    
    if 'all' in metric: #This is an option for optimization purposes but not a valid argument
        if 'val' in metric:
            print("Cannot combine combined metrics for these callbacks, setting metric='val_loss'"); metric = 'val_loss'
        else:
            print("Cannot combine combined metrics for these callbacks, setting metric='loss'"); metric = 'loss'

    if val_positive is not None:
        val_X1, val_Y1 = process_class(val_positive, label=1, img_num_channels=img_num_channels, min_pixel=min_pixel, max_pixel=max_pixel, normalize=normalize)
        if val_negative is None:
            val_X, val_Y = val_X1, val_Y1
        else:
            val_X2, val_Y2 = process_class(val_negative, label=0, img_num_channels=img_num_channels, min_pixel=min_pixel, max_pixel=max_pixel, normalize=normalize)
            val_X, val_Y = np.r_[val_X1, val_X2], np.r_[val_Y1, val_Y2]
    else:
        if val_negative is None:
            val_X, val_Y = None, None
        else:
            val_X2, val_Y2 = process_class(val_negative, label=0, img_num_channels=img_num_channels, min_pixel=min_pixel, max_pixel=max_pixel, normalize=normalize)
            val_X, val_Y = val_X2, val_Y2

    img_width, img_height = positive_class[0].shape[0], positive_class[0].shape[1]
    
    ix = np.random.permutation(len(positive_class))
    positive_class = positive_class[ix]

    ix = np.random.permutation(len(negative_class))
    negative_class = negative_class[ix]

    X_train, Y_train = create_training_set(positive_class, negative_class, normalize=normalize, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)
    
    if normalize:
        X_train[X_train > 1] = 1; X_train[X_train < 0] = 0
        
    #Apply SMOTE to oversample the minority class
    if smote_sampling > 0:
        X_train[np.isfinite(X_train)==False] = 0
        if len(np.where(Y_train[:,0]==1)[0]) == len(np.where(Y_train[:,1]==1)[0]):
            X_train_res, Y_train_res = X_train, Y_train
            print('Classes are already balanced, skipping SMOTE sampling.')
        else:
            X_train_res, Y_train_res = smote_oversampling(X_train, Y_train, smote_sampling=smote_sampling)
    elif smote_sampling == 0:
        X_train_res, Y_train_res = X_train, Y_train
    else:
        raise ValueError('smote_sampling must be a float between 0.0 and 1.0!')

    num_classes, input_shape = 2, (img_width, img_height, img_num_channels)
   
    if verbose == 1:
        dense_neurons_3 = dropout_3 = 'N/A'
        print_params(batch_size, lr, decay, momentum, nesterov, loss, optimizer, model_reg, conv_init, activation_conv, 
            dense_init, activation_dense, filter_1, filter_2, filter_3, filter_4, filter_5, filter_size_1, filter_size_2, 
            filter_size_3, filter_size_4, filter_size_5, pooling_1, pooling_2, pooling_3, pooling_4, pooling_5, pool_size_1, 
            pool_size_2, pool_size_3, pool_size_4, pool_size_5, conv_reg, dense_reg, dense_neurons_1, dense_neurons_2, 
            dense_neurons_3, dropout_1, dropout_2, dropout_3, beta_1, beta_2, amsgrad, rho)

    #Uniform scaling initializer
    conv_init = VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform', seed=None) if conv_init == 'uniform_scaling' else conv_init
    dense_init = VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform', seed=None) if dense_init == 'uniform_scaling' else dense_init

    #Call the appropriate tf.keras.losses.Loss function
    loss = get_loss_function(loss, weight=weight)

    #Model configuration
    model = Sequential()

    #Block 1
    #Conv2D
    model.add(Conv2D(filter_1, filter_size_1, strides=strides_1, activation=activation_conv, input_shape=input_shape, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))     
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None  
    #Conv2D
    model.add(Conv2D(filter_1, filter_size_1, strides=strides_1, activation=activation_conv, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))     
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None        
    #Pooling Layer
    model.add(tf.keras.layers.Lambda(lambda x: -tf.nn.max_pool2d(-x, ksize=(pool_size_1, pool_size_1), strides=(pool_stride_1, pool_stride_1), padding='SAME'))) if pooling_1 == 'min' else None
    model.add(MaxPool2D(pool_size=pool_size_1, strides=pool_stride_1, padding=padding)) if pooling_1 == 'max' else None 
    model.add(AveragePooling2D(pool_size=pool_size_1, strides=pool_stride_1, padding=padding)) if pooling_1 == 'average' else None 
    #Only applying LRN after pooling layers, as per AlexNet convention. 
    model.add(Lambda(lambda x: tf.nn.local_response_normalization(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75))) if model_reg == 'local_response' else None

    #Block 2
    #Conv2D
    model.add(Conv2D(filter_2, filter_size_2, strides=strides_2, activation=activation_conv, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))     
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None  
    #Conv2D
    model.add(Conv2D(filter_2, filter_size_2, strides=strides_2, activation=activation_conv, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))     
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None  
    #Pooling Layer
    model.add(tf.keras.layers.Lambda(lambda x: -tf.nn.max_pool2d(-x, ksize=(pool_size_2, pool_size_2), strides=(pool_stride_2, pool_stride_2), padding='SAME'))) if pooling_2 == 'min' else None
    model.add(MaxPool2D(pool_size=pool_size_2, strides=pool_stride_2, padding=padding)) if pooling_2 == 'max' else None
    model.add(AveragePooling2D(pool_size=pool_size_2, strides=pool_stride_2, padding=padding)) if pooling_2 == 'average' else None 
    #Only applying LRN after pooling layers, as per AlexNet convention. 
    model.add(Lambda(lambda x: tf.nn.local_response_normalization(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75))) if model_reg == 'local_response' else None

    #Block 3
    #Conv2D
    model.add(Conv2D(filter_3, filter_size_3, strides=strides_3, activation=activation_conv, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))     
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None  
    #Conv2D
    model.add(Conv2D(filter_3, filter_size_3, strides=strides_3, activation=activation_conv, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))     
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None  
    #Conv2D
    model.add(Conv2D(filter_3, filter_size_3, strides=strides_3, activation=activation_conv, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))     
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None  
    #Pooling Layer
    model.add(tf.keras.layers.Lambda(lambda x: -tf.nn.max_pool2d(-x, ksize=(pool_size_3, pool_size_3), strides=(pool_stride_3, pool_stride_3), padding='SAME'))) if pooling_3 == 'min' else None
    model.add(MaxPool2D(pool_size=pool_size_3, strides=pool_stride_3, padding=padding)) if pooling_3 == 'max' else None
    model.add(AveragePooling2D(pool_size=pool_size_3, strides=pool_stride_3, padding=padding)) if pooling_3 == 'average' else None
    #Only applying LRN after pooling layers, as per AlexNet convention. 
    model.add(Lambda(lambda x: tf.nn.local_response_normalization(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75))) if model_reg == 'local_response' else None

    #Block 4
    #Conv2D
    model.add(Conv2D(filter_4, filter_size_4, strides=strides_4, activation=activation_conv, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))     
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None  
    #Conv2D
    model.add(Conv2D(filter_4, filter_size_4, strides=strides_4, activation=activation_conv, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))     
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None  
    #Conv2D
    model.add(Conv2D(filter_4, filter_size_4, strides=strides_4, activation=activation_conv, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))     
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None  
    #Pooling Layer
    model.add(tf.keras.layers.Lambda(lambda x: -tf.nn.max_pool2d(-x, ksize=(pool_size_4, pool_size_4), strides=(pool_stride_4, pool_stride_4), padding='SAME'))) if pooling_4 == 'min' else None
    model.add(MaxPool2D(pool_size=pool_size_4, strides=pool_stride_4, padding=padding)) if pooling_4 == 'max' else None
    model.add(AveragePooling2D(pool_size=pool_size_4, strides=pool_stride_4, padding=padding)) if pooling_4 == 'average' else None
    #Only applying LRN after pooling layers, as per AlexNet convention. 
    model.add(Lambda(lambda x: tf.nn.local_response_normalization(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75))) if model_reg == 'local_response' else None

    #Block 5
    #Conv2D
    model.add(Conv2D(filter_5, filter_size_5, strides=strides_5, activation=activation_conv, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))     
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None  
    #Conv2D
    model.add(Conv2D(filter_5, filter_size_5, strides=strides_5, activation=activation_conv, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))     
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None  
    #Conv2D
    model.add(Conv2D(filter_5, filter_size_5, strides=strides_5, activation=activation_conv, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))     
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None  
    #Pooling Layer
    model.add(tf.keras.layers.Lambda(lambda x: -tf.nn.max_pool2d(-x, ksize=(pool_size_5, pool_size_5), strides=(pool_stride_5, pool_stride_5), padding='SAME'))) if pooling_5 == 'min' else None
    model.add(MaxPool2D(pool_size=pool_size_5, strides=pool_stride_5, padding=padding)) if pooling_5 == 'max' else None
    model.add(AveragePooling2D(pool_size=pool_size_5, strides=pool_stride_5, padding=padding)) if pooling_5 == 'average' else None
    #Only applying LRN after pooling layers, as per AlexNet convention. 
    model.add(Lambda(lambda x: tf.nn.local_response_normalization(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75))) if model_reg == 'local_response' else None

    #Dense layers
    model.add(Flatten())

    #FCC 1
    model.add(Dense(dense_neurons_1, activation=activation_dense, kernel_initializer=dense_init, kernel_regularizer=l2(dense_reg)))
    model.add(Dropout(dropout_1))
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None  
    
    #FCC 2
    model.add(Dense(dense_neurons_2, activation=activation_dense, kernel_initializer=dense_init, kernel_regularizer=l2(dense_reg)))
    model.add(Dropout(dropout_2))
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None  

    #Output layer
    model.add(Dense(num_classes, activation='sigmoid', kernel_initializer=dense_init)) #adding a kernel model_reg has no effect if sigmoid is being used, but works for layers with trainable weights like softmax
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None  

    #Call the appropriate tf.keras.optimizers function
    optimizer = get_optimizer(optimizer, lr, momentum=momentum, decay=decay, rho=rho, nesterov=nesterov, beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad)

    #Compile the Model
    model.compile(loss=loss, optimizer=optimizer, metrics=[tf.keras.metrics.BinaryAccuracy(), f1_score])

    #Wheter to maximize or minimize the metric
    mode = 'min' if 'loss' in metric else 'max'

    #Optional checkpoint callback, with the monitor being the input metric.
    callbacks_list = []
    callbacks_list.append(ModelCheckpoint(str(Path.home())+'/'+'checkpoint.hdf5', monitor=metric, verbose=2, save_best_only=True, mode=mode)) if checkpoint else None

    #Early stopping callback
    callbacks_list.append(EarlyStopping(monitor=metric, mode=mode, patience=patience)) if patience > 0 else None

    #Early stop callback for use during the optimization routine
    callbacks_list.append(early_stop_callback) if early_stop_callback is not None else None

    #Fit the Model
    if val_X is None:
        history = model.fit(X_train_res, Y_train_res, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, verbose=verbose)
    else:
        history = model.fit(X_train_res, Y_train_res, batch_size=batch_size, validation_data=(val_X, val_Y), epochs=epochs, callbacks=callbacks_list, verbose=verbose)

    if save_training_data:
        path = str(Path.home()) if path is None else path
        path += '/' if path[-1] != '/' else ''
        try:
            flattened_labels = np.argmax(Y_train_res, axis=1)
            ix1 = np.where(flattened_labels == 1)[0]; ix2 = np.where(flattened_labels == 0)[0]; 
            np.save(path+'class_1.npy', X_train_res[ix1]); np.save(path+'class_2.npy', X_train_res[ix2]); 
            if val_positive is not None:
                np.save(path+'val_class_1.npy', val_X1)
            if val_negative is not None:
                np.save(path+'val_class_2.npy', val_X2)     
            print('Files saved in: {}'.format(path)); print('NOTE: The training data files may have to be manually moved to the MicroLIA_cnn_model folder in order for them to be loaded when running load()!') 
        except Exception as e:
            print(f"Could not save training data due to error: {e}")

    return model, history

def Resnet18(positive_class, negative_class, img_num_channels=1, normalize=True, 
    min_pixel=0, max_pixel=100, val_positive=None, val_negative=None, epochs=100, batch_size=32, 
    optimizer='sgd', lr=0.0001, momentum=0.9, decay=0.0, nesterov=False, rho=0.9, beta_1=0.9, beta_2=0.999, amsgrad=False,
    loss='binary_crossentropy', conv_init='uniform_scaling', dense_init='TruncatedNormal',
    activation_conv='relu', activation_dense='relu', conv_reg=0, dense_reg=0, padding='same',model_reg=None,
    filters=64, filter_size=7, strides=1, pooling='max', pool_size=3, pool_stride=2,
    block_filters_1=64, block_filters_2=128, block_filters_3=256, block_filters_4=512, block_filters_size=3, 
    smote_sampling=0, patience=0, metric='binary_accuracy', early_stop_callback=None, checkpoint=False, 
    weight=None, verbose=1, save_training_data=False, path=None):#use_zero_padding=True, zero_padding=3, final_avg_pool_size=7
    """
    Builds a ResNet-18 model with the default parameters from the original He et al. 2015 paper.

    ResNet-18 uses residual blocks to enable the training of very deep neural networks. A residual block 
    consists of two or more convolutional layers, batch normalization, and shortcut connections that bypass 
    the convolutional layers (see the identity_block and conv_block functions). By using residual blocks and 
    shortcut connections, the gradients can flow directly from the output to the input of the block, which makes 
    it easier for the network to learn the identity mapping and deeper representations.

    This model is configured with the layer parameters presented in the paper, including
    a zero-padding layer with default padding of zero_padding=(3, 3) which is applied to the input of the first convolutional layer. 
    This is done to ensure that the spatial dimensions of the feature maps are preserved after the convolutional 
    layers with stride 2, and to facilitate the downsampling operation in the subsequent max pooling layers.

    In practice this layer can be omitted as it is not strictly necessary for the architecture to work properly. 
    In practice, the absence of the zero-padding layer may result in a slightly smaller output size and a different 
    receptive field size for the first convolutional layer, but this should not have a major impact on the overall 
    performance of the network.

    Note:
        Resnet-18 does not apply any fully-connected (aka Dense) layers at the end of the network. Instead, Resnet-18
        uses a global average pooling layer at the end because it was found to be effective in reducing overfitting and 
        improving generalization performance. GAP is a pooling operation that computes the average value of each feature 
        map in the last convolutional block to generate a single value for each feature map. This produces a fixed-length 
        feature vector that can be fed directly to the final fully connected layer for classification.

    Returns:
        The trained CNN model and accompanying history.
    """

    raise ValueError('ResNet-18 Model is not currently stable, please select another model!')

    if batch_size < 16 and model_reg == 'batch_norm':
        print("Batch Normalization can be unstable with low batch sizes, if loss returns nan try a larger batch size and/or smaller learning rate.")
    
    if 'all' in metric: #This is an option for optimization purposes but not a valid argument
        if 'val' in metric:
            print("Cannot combine combined metrics for these callbacks, setting metric='val_loss'"); metric = 'val_loss'
        else:
            print("Cannot combine combined metrics for these callbacks, setting metric='loss'"); metric = 'loss'

    if val_positive is not None:
        val_X1, val_Y1 = process_class(val_positive, label=1, img_num_channels=img_num_channels, min_pixel=min_pixel, max_pixel=max_pixel, normalize=normalize)
        if val_negative is None:
            val_X, val_Y = val_X1, val_Y1
        else:
            val_X2, val_Y2 = process_class(val_negative, label=0, img_num_channels=img_num_channels, min_pixel=min_pixel, max_pixel=max_pixel, normalize=normalize)
            val_X, val_Y = np.r_[val_X1, val_X2], np.r_[val_Y1, val_Y2]
    else:
        if val_negative is None:
            val_X, val_Y = None, None
        else:
            val_X2, val_Y2 = process_class(val_negative, label=0, img_num_channels=img_num_channels, min_pixel=min_pixel, max_pixel=max_pixel, normalize=normalize)
            val_X, val_Y = val_X2, val_Y2

    img_width = positive_class[0].shape[0]
    img_height = positive_class[0].shape[1]
    
    ix = np.random.permutation(len(positive_class))
    positive_class = positive_class[ix]

    ix = np.random.permutation(len(negative_class))
    negative_class = negative_class[ix]

    X_train, Y_train = create_training_set(positive_class, negative_class, normalize=normalize, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)
    
    if normalize:
        X_train[X_train > 1] = 1; X_train[X_train < 0] = 0
        
    #Apply SMOTE to oversample the minority class
    if smote_sampling > 0:
        X_train[np.isfinite(X_train)==False] = 0
        if len(np.where(Y_train[:,0]==1)[0]) == len(np.where(Y_train[:,1]==1)[0]):
            X_train_res, Y_train_res = X_train, Y_train
            print('Classes are already balanced, skipping SMOTE sampling.')
        else:
            X_train_res, Y_train_res = smote_oversampling(X_train, Y_train, smote_sampling=smote_sampling)
    elif smote_sampling == 0:
        X_train_res, Y_train_res = X_train, Y_train
    else:
        raise ValueError('smote_sampling must be a float between 0.0 and 1.0!')

    num_classes, input_shape = 2, (img_width, img_height, img_num_channels)
   
    if verbose == 1:
        activation_dense = dropout_3 = 'N/A'; dense_reg = 0
        filter_5 = filter_size_5 = pool_size_5 = dropout_1 = dropout_2 = dense_neurons_1 = dense_neurons_2 = dense_neurons_3 = pooling_2 = pooling_3 = pooling_4 = pooling_5 = 'None'
        print_params(batch_size, lr, decay, momentum, nesterov, loss, optimizer, model_reg, conv_init, activation_conv, 
            dense_init, activation_dense, filters, block_filters_1, block_filters_2, block_filters_3, block_filters_4, 
            filter_size, block_filters_size, block_filters_size, block_filters_size, block_filters_size, pooling, pooling_2, 
            pooling_3, pooling_4, pooling_5, pool_size, pool_size, pool_size, pool_size, pool_size, conv_reg, dense_reg, 
            dense_neurons_1, dense_neurons_2, dense_neurons_3, dropout_1, dropout_2, dropout_3)

    #Uniform scaling initializer
    conv_init = VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform', seed=None) if conv_init == 'uniform_scaling' else conv_init
    dense_init = VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform', seed=None) if dense_init == 'uniform_scaling' else dense_init

    #Call the appropriate tf.keras.losses.Loss function
    loss = get_loss_function(loss, weight=weight)

    #Model configuration
    input_tensor = Input(shape=input_shape)
    x = input_tensor

    #zero-padding layer that VGG16 traditionally uses
    if img_num_channels == 1:
        x = ZeroPadding2D(padding=((3,3),(3,3)))(x)
    else:
        x = ZeroPadding2D(padding=((3,3),(3,3)), data_format='channels_last')(x)

    #Only Conv2D Layer before the Residual Blocks
    x = Conv2D(filters, kernel_size=filter_size, strides=strides, activation=activation_conv, kernel_initializer=conv_init)(x)
    x = BatchNormalization()(x) if model_reg == 'batch_norm' else x
    #Pooling layer
    x = tf.keras.layers.Lambda(lambda x: -tf.nn.max_pool2d(-x, ksize=(pool_size, pool_size), strides=(pool_stride, pool_stride), padding='SAME') if pooling == 'min' else tf.nn.max_pool2d(x, ksize=(pool_size, pool_size), strides=(pool_stride, pool_stride), padding='SAME'))(x)
    x = MaxPool2D(pool_size=pool_size, strides=pool_stride, padding=padding)(x) if pooling == 'max' else x
    x = AveragePooling2D(pool_size=pool_size, strides=pool_stride, padding=padding)(x) if pooling == 'average' else x
    #Only applying LRN after pooling layers, as per AlexNet convention. Note that if this is set, then the blocks won't have ANY regularization since the only option is 'batch_norm' for the blocks 
    x = Lambda(lambda x: tf.nn.local_response_normalization(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75))(x) if model_reg == 'local_response' else x

    #Block 1
    x = resnet_block(x, block_filters_1, block_filters_1, block_filters_size, activation=activation_conv, stride=1, padding=padding, kernel_initializer=conv_init, model_reg=model_reg)
    x = resnet_block(x, block_filters_1, block_filters_1, block_filters_size, activation=activation_conv, stride=1, padding=padding, kernel_initializer=conv_init, model_reg=model_reg)
    x = resnet_block(x, block_filters_1, block_filters_1, block_filters_size, activation=activation_conv, stride=1, padding=padding, kernel_initializer=conv_init, model_reg=model_reg)

    #Block 2
    x = resnet_block(x, block_filters_1, block_filters_2, block_filters_size, activation=activation_conv, stride=2, padding=padding, kernel_initializer=conv_init, model_reg=model_reg)
    x = resnet_block(x, block_filters_2, block_filters_2, block_filters_size, activation=activation_conv, stride=1, padding=padding, kernel_initializer=conv_init, model_reg=model_reg)
    x = resnet_block(x, block_filters_2, block_filters_2, block_filters_size, activation=activation_conv, stride=1, padding=padding, kernel_initializer=conv_init, model_reg=model_reg)
    x = resnet_block(x, block_filters_2, block_filters_2, block_filters_size, activation=activation_conv, stride=1, padding=padding, kernel_initializer=conv_init, model_reg=model_reg)

    #Block 3
    x = resnet_block(x, block_filters_2, block_filters_3, block_filters_size, activation=activation_conv, stride=2, padding=padding, kernel_initializer=conv_init, model_reg=model_reg)
    x = resnet_block(x, block_filters_3, block_filters_3, block_filters_size, activation=activation_conv, stride=1, padding=padding, kernel_initializer=conv_init, model_reg=model_reg)
    x = resnet_block(x, block_filters_3, block_filters_3, block_filters_size, activation=activation_conv, stride=1, padding=padding, kernel_initializer=conv_init, model_reg=model_reg)
    x = resnet_block(x, block_filters_3, block_filters_3, block_filters_size, activation=activation_conv, stride=1, padding=padding, kernel_initializer=conv_init, model_reg=model_reg)

    #Block 4
    x = resnet_block(x, block_filters_3, block_filters_4, block_filters_size, activation=activation_conv, stride=2, padding=padding, kernel_initializer=conv_init, model_reg=model_reg)
    x = resnet_block(x, block_filters_4, block_filters_4, block_filters_size, activation=activation_conv, stride=1, padding=padding, kernel_initializer=conv_init, model_reg=model_reg)
    x = resnet_block(x, block_filters_4, block_filters_4, block_filters_size, activation=activation_conv, stride=1, padding=padding, kernel_initializer=conv_init, model_reg=model_reg)

    x = GlobalAveragePooling2D()(x) #Global pooling operation that reduces the spatial dimensions (height and width) of the input tensor to a single value per channel using the mean of all values in a given channel
    x = Dense(num_classes, kernel_initializer=dense_init)(x)
    x = Activation('sigmoid')(x)

    model = Model(inputs=input_tensor, outputs=x)

    #Call the appropriate tf.keras.optimizers function
    optimizer = get_optimizer(optimizer, lr, momentum=momentum, decay=decay, rho=rho, nesterov=nesterov, beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad)

    #Compile the Model
    model.compile(loss=loss, optimizer=optimizer, metrics=[tf.keras.metrics.BinaryAccuracy(), f1_score])

    #Wheter to maximize or minimize the metric
    mode = 'min' if 'loss' in metric else 'max'

    #Optional checkpoint callback, with the monitor being the input metric.
    callbacks_list = []
    callbacks_list.append(ModelCheckpoint(str(Path.home())+'/'+'checkpoint.hdf5', monitor=metric, verbose=2, save_best_only=True, mode=mode)) if checkpoint else None

    #Early stopping callback
    callbacks_list.append(EarlyStopping(monitor=metric, mode=mode, patience=patience)) if patience > 0 else None

    #Early stop callback for use during the optimization routine
    callbacks_list.append(early_stop_callback) if early_stop_callback is not None else None

    #Fit the Model
    if val_X is None:
        history = model.fit(X_train_res, Y_train_res, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, verbose=verbose)
    else:
        history = model.fit(X_train_res, Y_train_res, batch_size=batch_size, validation_data=(val_X, val_Y), epochs=epochs, callbacks=callbacks_list, verbose=verbose)

    if save_training_data:
        path = str(Path.home()) if path is None else path
        path += '/' if path[-1] != '/' else ''
        try:
            flattened_labels = np.argmax(Y_train_res, axis=1)
            ix1 = np.where(flattened_labels == 1)[0]; ix2 = np.where(flattened_labels == 0)[0]; 
            np.save(path+'class_1.npy', X_train_res[ix1]); np.save(path+'class_2.npy', X_train_res[ix2]); 
            if val_positive is not None:
                np.save(path+'val_class_1.npy', val_X1)
            if val_negative is not None:
                np.save(path+'val_class_2.npy', val_X2)     
            print('Files saved in: {}'.format(path)); print('NOTE: The training data files may have to be manually moved to the MicroLIA_cnn_model folder in order for them to be loaded when running load()!') 
        except Exception as e:
            print(f"Could not save training data due to error: {e}")

    return model, history

def resnet_block(x, filters_in, filters_out, filter_size=3, activation='relu', stride=1, padding='same', kernel_initializer='he_normal', model_reg='batch_norm'):
    """
    ResNet block implementation for convolutional neural networks.
    
    The block includes two convolutional layers with batch normalization and ReLU activation, 
    followed by a skip connection that adds the original input tensor to the output of the second 
    convolutional layer. The input x is first assigned to the variable residual, since x will be modified 
    by the convolutional and normalization layers in the function, and the original 
    input x is needed to add back to the output of the final convolutional layer.

    The purpose of the residual connection is to skip over some convolutional layers and 
    directly connect the input and output of a block in a neural network, allowing for easier 
    training and deeper architectures. In this implementation, the residual connection is added 
    after the first convolutional layer, but before the second convolutional layer, so the 
    original input can be added back to the output of the final convolutional layer.
    
    Args:
        x (tf.Tensor): The input tensor to the block.
        filters_in (int): Number of input filters.
        filters_out (int): Number of output filters.
        filter_size (int): Size of the filter kernel.
        stride (int, optional): Stride for the first convolutional layer. Defaults to 1.
    
    Returns:
        tf.Tensor: The output tensor of the block.
    """

    #Save the input tensor as the residual
    residual = x
    
    #Conv2D
    x = Conv2D(filters_out, kernel_size=filter_size, activation=activation, strides=stride, padding=padding, kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization()(x) if model_reg == 'batch_norm' else x
    #Conv2D
    x = Conv2D(filters_out, kernel_size=filter_size, activation=activation, strides=1, padding=padding, kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization()(x) if model_reg == 'batch_norm' else x

    #This is the ResNet shortcut connection
    if stride > 1 or filters_in != filters_out:
        residual = Conv2D(filters_out, kernel_size=1, strides=stride, padding='valid', kernel_initializer=kernel_initializer)(residual)
        residual = BatchNormalization()(residual) if model_reg == 'batch_norm' else residual

    x = Add()([x, residual])

    return x
    
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

    f1 = 2.0 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

    return f1

def calculate_tp_fp(model, sample, y_true):
    """
    Computes the true positives (tp) and false positives (fp) for a single sample using a given model.

    Args:
        model: The trained model.
        sample: The input sample (image) for prediction.
        y_true (array): The ground truth labels for the sample(s). Must be the same length as the input sample argument.

    Returns:
        tp: The number of true positives.
        fp: The number of false positives.
    """
    # Make a prediction using the model
    y_pred = model.predict(sample)

    # Convert y_true and y_pred to binary values
    y_true_binary = tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1))
    y_pred_binary = tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1))

    # Calculate true positives (tp)
    tp = tf.keras.backend.sum(y_true_binary * y_pred_binary)

    # Calculate false positives (fp)
    fp = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred_binary - y_true_binary, 0, 1)))

    return tp, fp


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

def weighted_binary_crossentropy(weight):
    """
    Return a binary cross-entropy loss function with weighted terms.

    This function returns a callable loss function that can be used as a loss argument
    in Keras models. The loss function calculates the binary cross-entropy between the
    true binary labels and the predicted probabilities, but applies a weight factor to
    the positive class to address class imbalance. The weight factor is given by `weight`,
    which can be any non-negative scalar.

    When `weight` is greater than 1, the loss function will assign more importance to the
    positive class, while when `weight` is less than 1, the loss function will assign less
    importance to the positive class. The weight factor can be used to balance the contribution
    of the positive and negative classes to the loss function.

    The implementation is a nested function, which allows for easy customization of the `weight`
    parameter.

    Args:
        weight (float): A scalar weight factor for the positive class. This parameter controls
            the relative weight of the positive class in the loss function. When `weight` is 1,
            the loss function is equivalent to the standard binary cross-entropy loss function.

    Returns:
        A callable binary cross-entropy loss function that can be used as a loss argument in Keras models.
    """

    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        bce = -(y_true * K.log(y_pred) * weight + (1 - y_true) * K.log(1 - y_pred))
        weighted_bce = K.mean(bce, axis=-1)
        return weighted_bce

    return loss


### AlexNet Helper Functions ###

def get_optimizer(optimizer, lr, momentum=None, decay=None, rho=0.9, nesterov=False, beta_1=0.9, beta_2=0.999, amsgrad=False):
    """
    Returns an optimizer object based on the specified parameters.
    
    Note:
     Adam is essentially RMSprop with momentum, and Nadam is Adam with Nesterov momentum.
     Adam with beta1=1 is equivalent to RMSProp with momentum=0. The argument beta2 of Adam 
     and the argument decay of RMSProp are the same. AdaGrad uses the second moment with no decay 
     while RMSProp uses the second moment -- Adam uses both first and second moments, and is generally 
     the better option. AdaDelta is used in lieue of AdaGrad as it resolves the monotonically decreasing 
     learning rate problem which arises when using AdaGrad.

    Args:
        optimizer (str): Name of the optimizer.
        lr (float): Learning rate.
        momentum (float): Momentum for SGD optimizer, defaults to None.
        rho (float): Discounting factor for the history/coming gradient. Used by the AdaDelta and the RMSProp
            optimizers. Defaults to 0.9.
        decay (float): Time inverse decay of learning rate, used by all optimizers, keras only adds to the optimizer
            class for convenience. Different than rho! Defaults to None.
        nesterov (bool): Whether to use Nesterov momentum for SGD optimizer, defaults to False.
        beta_1 (float): Decay rate when calculating the first moment gradient, multiplied by beta_2 at the end of each epoch.
            Used by the Adam optimizer. Decreasing this value will slow down the learning. Defaults to 0.9. 
        beta_2 (float): Decay rate when calculating the second moment gradient, multiplied by beta_2 at the end of each epoch.
            Used by the Adam optimizer. Decreasing this value will slow down the learning. Defaults to 0.999. 
        amsgrad (bool): Whether to apply AMSGrad variant of this algorithm from the Reddi et al (2018). Defaults to False.

    Returns:
        optimizer: Optimizer object.

    Raises:
        ValueError: If an invalid optimizer name is provided.
    """

    if optimizer == 'sgd':
        optimizer = SGD(learning_rate=lr,  momentum=momentum, nesterov=nesterov)
    elif optimizer == 'adam':
        optimizer = Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad)
    elif optimizer == 'adamax':
        optimizer = Adamax(learning_rate=lr, beta_1=beta_1, beta_2=beta_2)
    elif optimizer == 'nadam':
        optimizer = Nadam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2)
    elif optimizer == 'adadelta':
        optimizer = Adadelta(learning_rate=lr, rho=rho)
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate=lr, rho=rho)
    else:
        raise ValueError("Invalid optimizer name. Available options are 'sgd', 'adam', 'adamax', 'nadam', 'adadelta', or 'rmsprop'.")

    return optimizer

def get_loss_function(loss, weight=None):
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
    elif loss == 'weighted_binary_crossentropy':
        if weight is None:
            raise ValueError('If using weighted loss function, the weight parameter must be input!')
        return weighted_binary_crossentropy(weight)
    else:
        raise ValueError("Invalid loss function name")
   
def print_params(batch_size, lr, decay, momentum, nesterov, loss, optimizer, 
    model_reg, conv_init, activation_conv, dense_init, activation_dense,
    filter1, filter2, filter3, filter4, filter5, filter_size_1, filter_size_2, 
    filter_size_3, filter_size_4, filter_size_5, pooling_1, pooling_2, pooling_3,
    pooling_4, pooling_5, pool_size_1, pool_size_2, pool_size_3, pool_size_4, pool_size_5,
    conv_reg, dense_reg, dense_neurons_1, dense_neurons_2, dense_neurons_3, dropout_1, dropout_2, 
    dropout_3, beta_1, beta_2, amsgrad, rho):
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

    print(); print('===== Training Parameters ====='); print()
    print('|| Batch Size : '+str(batch_size), '|| Loss Function : '+loss, '||')

    if optimizer == 'sgd':
        print('|| Optimizer : '+optimizer, '|| lr : '+str(np.round(lr, 7)), '|| Decay : '+str(np.round(decay, 5)), '|| Momentum : '+str(momentum), '|| Nesterov : '+str(nesterov)+' ||')
    elif optimizer == 'adadelta' or optimizer == 'rmsprop':
        print('|| Optimizer : '+optimizer, '|| lr : '+str(np.round(lr, 7)), '|| rho : '+str(np.round(rho, 5)), '|| Decay : '+str(np.round(decay, 5))+' ||')
    elif optimizer == 'adam' or optimizer == 'adamax' or optimizer == 'nadam':
        print('|| Optimizer : '+optimizer, '|| lr : '+str(np.round(lr, 7)), '|| Beta 1 : '+str(np.round(beta_1, 5)), '|| Beta 2 : '+str(np.round(beta_2, 5)), '||  amsgrad : '+str(amsgrad)+' ||')
    
    print(); print('=== Architecture Parameters ==='); print()
    print('Model Regularizer : '+ str(model_reg))
    print('Convolutional L2 Regularizer : '+ str(conv_reg))
    print('Convolutional Initializer : '+ conv_init)
    print('Convolutional Activation Fn : '+ activation_conv)
    print('Dense L2 Regularizer : '+ str(dense_reg))
    print('Dense Initializer : '+ dense_init)
    print('Dense Activation Fn : '+ activation_dense); print()

    if dropout_3 != 'N/A': #This is AlexNet and custom_model since droput_3 = N/A is set for VGG16 and Resnet-18 only
        print('======= Conv2D Layer Parameters ======'); print()
        print('Filter 1 || Num: {}, Size : {}, Pooling : {}, Pooling Size : {}'.format(filter1, filter_size_1, pooling_1, pool_size_1))
        if filter_size_2 > 0:
            print('Filter 2 || Num: {}, Size : {}, Pooling : {}, Pooling Size : {}'.format(filter2, filter_size_2, pooling_2, pool_size_2))
        if filter_size_3 > 0:
            print('Filter 3 || Num: {}, Size : {}, Pooling : {}, Pooling Size : {}'.format(filter3, filter_size_3, pooling_3, pool_size_3))
        if filter_size_4 > 0:
            print('Filter 4 || Num: {}, Size : {}, Pooling : {}, Pooling Size : {}'.format(filter4, filter_size_4, pooling_4, pool_size_4))
        if filter_size_5 > 0:
            print('Filter 5 || Num: {}, Size : {}, Pooling : {}, Pooling Size : {}'.format(filter5, filter_size_5, pooling_5, pool_size_5))
        
        print(); print('======= Dense Layer Parameters ======'); print()
        print('Neurons 1 || Num : {}, Dropout : {}'.format(dense_neurons_1, dropout_1))
        if dense_neurons_2 > 0:
            print('Neurons 2 || Num : {}, Dropout : {}'.format(dense_neurons_2, dropout_2))
        if dense_neurons_3 > 0:
            print('Neurons 3 || Num : {}, Dropout : {}'.format(dense_neurons_3, dropout_3))
        print(); print('==============================='); print()
    else:
        if activation_dense == 'N/A': #For Resnet-18 
            print('======= Conv2D Layer Parameters ======'); print()
            print('Filter 1 || Num: {}, Size : {}, Pooling : {}'.format(filter1, filter_size_1, pooling_1, pool_size_1))
            if filter_size_2 > 0:
                print('Residual Block 1 || Num: {}, Size : {}'.format(filter2, filter_size_2))
            if filter_size_3 > 0:
                print('Residual Block 2 || Num: {}, Size : {}'.format(filter3, filter_size_3))
            if filter_size_4 > 0:
                print('Residual Block 3 || Num: {}, Size : {}'.format(filter4, filter_size_4))
            if filter_size_5 > 0:
                print('Residual Block 4 || Num: {}, Size : {},'.format(filter5, filter_size_5))
            print(); print('==============================='); print()
        else: #For VGG16
            print('======= Conv2D Layer Parameters ======'); print()
            print('Block 1 || Num: {}, Size : {}, Pooling : {}'.format(filter1, filter_size_1, pooling_1, pool_size_1))
            print('Block 2 || Num: {}, Size : {}'.format(filter2, filter_size_2))
            print('Block 3 || Num: {}, Size : {}'.format(filter3, filter_size_3))
            print('Block 4 || Num: {}, Size : {}'.format(filter4, filter_size_4))
            print('Block 5 || Num: {}, Size : {},'.format(filter5, filter_size_5))
            print(); print('======= Dense Layer Parameters ======'); print()
            print('Neurons 1 || Num : {}, Dropout : {}'.format(dense_neurons_1, dropout_1))
            print('Neurons 2 || Num : {}, Dropout : {}'.format(dense_neurons_2, dropout_2))    
            print(); print('==============================='); print()

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

def _set_style_():
    """
    Function to configure the matplotlib.pyplot style. This function is called before any images are saved,
    after which the style is reset to the default.
    """

    plt.rcParams["xtick.color"] = "323034"
    plt.rcParams["ytick.color"] = "323034"
    plt.rcParams["text.color"] = "323034"
    plt.rcParams["lines.markeredgecolor"] = "black"
    plt.rcParams["patch.facecolor"] = "#bc80bd"  # Replace with a valid color code
    plt.rcParams["patch.force_edgecolor"] = True
    plt.rcParams["patch.linewidth"] = 0.8
    plt.rcParams["scatter.edgecolors"] = "black"
    plt.rcParams["grid.color"] = "#b1afb5"  # Replace with a valid color code
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["legend.title_fontsize"] = 12
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["font.size"] = 15
    plt.rcParams["axes.prop_cycle"] = (cycler('color', ['#bc80bd', '#fb8072', '#b3de69', '#fdb462', '#fccde5', '#8dd3c7', '#ffed6f', '#bebada', '#80b1d3', '#ccebc5', '#d9d9d9']))  # Replace with valid color codes
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["lines.markersize"] = 6
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.framealpha"] = 0.8
    plt.rcParams["legend.fontsize"] = 13
    plt.rcParams["legend.edgecolor"] = "black"
    plt.rcParams["legend.borderpad"] = 0.2
    plt.rcParams["legend.columnspacing"] = 1.5
    plt.rcParams["legend.labelspacing"] = 0.4
    plt.rcParams["text.usetex"] = False
    plt.rcParams["axes.labelsize"] = 17
    plt.rcParams["axes.titlelocation"] = "center"
    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams["axes.autolimit_mode"] = "round_numbers"
    plt.rcParams["axes.labelpad"] = 3
    plt.rcParams["axes.formatter.limits"] = (-4, 4)
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1
    plt.rcParams["axes.grid"] = False
    plt.rcParams["axes.spines.right"] = True
    plt.rcParams["axes.spines.left"] = True
    plt.rcParams["axes.spines.top"] = True
    plt.rcParams["figure.titlesize"] = 18
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.dpi"] = 300

    return

