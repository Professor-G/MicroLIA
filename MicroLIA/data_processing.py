#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 10:38:16 2023

@author: daniel
"""
import copy
import numpy as np
from tensorflow.keras.utils import to_categorical

def find_duplicate_features(features, tolerance=1e-9):
    """
    This function will check if there are any duplicate columns 
    within the array by comparing each column with the next columns. 

    Args:
        features (ndarray): 2D array of features, column-wise.
        tolerance (float): tolerance level for comparison of values.
    Returns:
        A set of duplicate indices.
    """
    # Initialize a set to store the unique feature indices
    unique_indices = set()
    # Initialize a set to store the duplicate feature indices
    duplicate_indices = set()
    # Get the transpose of the features array
    features_T = features.T
    # Get the number of columns
    num_cols = features_T.shape[0]
    for i in range(num_cols):
        column1 = features_T[i]
        for j in range(i+1, num_cols):
            column2 = features_T[j]
            if np.all(np.isclose(column1, column2, atol=tolerance)):
                if i not in unique_indices:
                    unique_indices.add(i)
                    duplicate_indices.add(i)
                if j not in unique_indices:
                    unique_indices.add(j)
                    duplicate_indices.add(j)
    return duplicate_indices

def crop_image(data, x, y, size=50, invert=False):
    """
    This function takes a 2D array and returns a sub-array
    centered around x and y. The sub array will be a square of length = size.

    Note:
        When applying data augmentation techniques it is best to start with a larger
        image and then crop it to the appropriate size afterward, so as to avoid the 
        rotational shear visible on the edges.

        IMPORTANT: When loading data from a .fits file the pixel convention
        is switched. The (x, y) = (0, 0) position is on the top left corner of the .fits
        image. The standard convention is for the (x, y) = (0, 0) to be at the bottom left
        corner of the data. We strongly recommend you double-check your data coordinate
        convention. We made use of .fits data with the (x, y) = (0, 0) position at the top
        left of the image, for this reason we switched x and y when cropping out individual
        objects. The parameter invert=True performs the coordinate switch for us. This is only
        required because MicroLIA's cropping function assumes standard convention.

    Args:
        data (array): 2D array.
        x (int): Central x-position of the sub-array to be cropped out, relative
            to the entire data.
        y (int): Central y-position of the sub-array to be cropped out, relative
            to the entire data.
        size (int): length/width of the output array. Defaults to 50.
        invert (bool): If True the x & y coordinates will be switched
            when cropping out the object, see Note above. Defaults to False.
    Returns:
        The cropped array.

    Example:
        If we have a 100x100 image, we can crop this by setting x,y = (50,50), which
        would be the center of of the image. Since MicroLIA standard is 50x50, we will 
        set the size of the reshaped array to 50.

        >>> from MicroLIA import data_processing
        >>> resize = data_processing.crop_image(data, x=50, y=50, size=50)

        If your image is 200x200, then x, y = (100,100), and so on.
    """

    if invert:
        x, y = y, x
        
    data_copy = copy.deepcopy(data)

    o, r = np.divmod(size, 2)
    l = (int(x)-(o+r-1)).clip(0)
    u = (int(y)-(o+r-1)).clip(0)
    array = data_copy[l: int(x)+o+1, u:int(y)+o+1]
    
    out = np.full((size, size), np.nan, dtype=data_copy.dtype)
    out[:array.shape[0], :array.shape[1]] = array

    return out

def concat_channels(channel1, channel2, channel3=None):
    """
    This function concatenates multiple 2D arrays, useful for image classification when using multiple filters.

    Can combine SDSS g,r,i for example, to make one 3D image. Order at which
    they are stacked must be conistent when data is input for classification.
    
    Args:
        Channel1 (array): 2D array of the first channel.
        Channel2 (array): 2D array of the second channel.
        Channel3 (array, optional): 2D array of the third channel.

    Returns:
        3D array with each channel stacked.

    """
    
    if channel3 is None:
        colorized = (channel1[..., np.newaxis], channel2[..., np.newaxis])
    else:
        colorized = (channel1[..., np.newaxis], channel2[..., np.newaxis], channel3[..., np.newaxis])

    return np.concatenate(colorized, axis=-1)

def normalize_pixels(channels, min_pixel, max_pixel, img_num_channels):
    """
    This function will apply min-max normalization. It returns a 4-d array.
    
    Args:
        channel (array): 2D array for one image, 3D array for multiple images.
        min_pixel (int, optional): The minimum pixel count, defaults to 0. 
            Pixels with counts below this threshold will be set to this limit.
        max_pixel (int, optional): The maximum pixel count, defaults to 100. 
            Pixels with counts above this threshold will be set to this limit.

    Returns:      
        Reshaped data and label arrays.
    """

    if isinstance(max_pixel, int) and img_num_channels != 1:
        raise ValueError('The max_pixel parameter should be a list containing the value for each band!')
    if isinstance(max_pixel, int) is False and img_num_channels == 1:
        if isinstance(max_pixel, list):
            max_pixel = max_pixel[0]
        else:
            raise ValueError('If img_num_channels is 1 the max_pixel input must be an integer/float or list.')

    images = copy.deepcopy(channels)

    #The min pixel replaces NaN and below threshold values.
    images[np.isfinite(images) == False] = min_pixel 
    images[images < min_pixel] = min_pixel

    if img_num_channels == 1:
        images[images > max_pixel] = max_pixel
        return (images - min_pixel) /  (max_pixel - min_pixel)

    #Setting array dimensions for consistency#
    if len(images.shape) == 4:
        axis = images.shape[0]
        if images.shape[-1] != img_num_channels:
            raise ValueError('img_num_channels parameter must match the number of filters! Number of filters detected: '+str(channel.shape[-1]))
        img_width, img_height = images[0].shape[1], images[0].shape[0]
    elif len(images.shape) == 3:
        if img_num_channels == 1:
            axis, img_width, img_height = images.shape[0], images.shape[1],images.shape[2]
        else:
            axis, img_width, img_height = 1, images.shape[0], images.shape[1]
    elif len(images.shape) == 2:
        axis, img_width, img_height = 1, images.shape[1], images.shape[0]
    else:
        raise ValueError("Channel must either be 2D for a single sample, 3D for multiple samples or single sample with multiple filters, or 4D for multifilter images.")

    images = images.reshape(axis, img_width, img_height, img_num_channels)

    for i in range(img_num_channels):
        images[:,:,:,i][images[:,:,:,i] > max_pixel[i]] = max_pixel[i]
        images[:,:,:,i] = (images[:,:,:,i] - min_pixel) /  (max_pixel[i] - min_pixel)

    return images 

def process_class(channel, label=None, img_num_channels=1, normalize=True, min_pixel=638, max_pixel=3000):
    """
    Takes image data and returns the reshaped data array, which is required when 
    entering data into the CNN classifier. Note that if using multiple bands, the filters
    must be processed individually, and concatenated afterwards.
    
    If label is set to either 0 or 1, then the reshaped data is
    returned along with an array containing the label array. 
    This is used for generating training or validations sets of appropriate shape.
    
    Note:
        Image anomalies can be removed by setting normalize=True, as the 
        values below/above the thresholds are set to the min/max limits. We
        strongly recommend normalizing your data.

    Args:
        channel (array): 2D array for one image, 3D array for multiple images.
        img_num_channels (int): The number of filters used. Defaults to 1.
        label (int, optional): Class label. Defaults to None.
        normalize (bool, optional): True will apply min-max normalization.
        min_pixel (int, optional): The minimum pixel count, defaults to 638. 
            Pixels with counts below this threshold will be set to this limit.
        max_pixel (int, optional): The maximum pixel count, defaults to 3000. 
            Pixels with counts above this threshold will be set to this limit.

    Returns:      
        Reshaped data and label arrays.
    """

    if normalize:
        if len(channel) >= 1000:
            print('Normalizing images...') #For when predictions are being made
        data = normalize_pixels(channel, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)
    else:
        images = copy.deepcopy(channel)
        if len(images.shape) == 4:
            axis = images.shape[0]
            if images.shape[-1] != img_num_channels:
                raise ValueError('img_num_channels parameter must match the number of filters! Number of filters detected: '+str(channel.shape[-1]))
            img_width = images[0].shape[1]
            img_height = images[0].shape[0]
        elif len(images.shape) == 3:
            if img_num_channels == 1 :
                axis = images.shape[0]
                img_width = images.shape[1]
                img_height = images.shape[2]
            else:
                axis = 1
                img_width = images.shape[0]
                img_height = images.shape[1]
        elif len(images.shape) == 2:
            img_width = images.shape[1]
            img_height = images.shape[0]
            axis = 1
        else:
            raise ValueError("Channel must either be 2D for a single sample, 3D for multiple samples or single sample with multiple filters, or 4D for multifilter images.")
        data = images.reshape(axis, img_width, img_height, img_num_channels)
    
    if label is None:
        return data

    #reshape
    label = np.expand_dims(np.array([label]*len(data)), axis=1)
    label = to_categorical(label, 2)
    
    return data, label

def create_training_set(positive_data, negative_data, img_num_channels=1, normalize=True, min_pixel=638, max_pixel=3000):
    """
    Combines image data of known class to create a training set.
    This is used for training the machine learning models. The 
    max_pixel parameter can be a single value corresponding to a single channel
    or to a list containing the value for each individual band.

    Note: 
        This function is for binary classification only, the manual procedure for multiclass
        training set creation looks as follows:

        >>> from MicroLIA.data_processing import process_class
        >>> import numpy as np 

        >>> class1_data, class1_label = process_class(data1, label=0)
        >>> class2_data, class2_label = process_class(data2, label=1)
        >>> class3_data, class3_label = process_class(data3, label=2)

        >>> training_data = np.r_[class1_data, class2_data, class3_data]
        >>> training_labels = np.r_[class1_label class2_label, class3_label]

    Args:
        positive_data (array): 3D array containing more than one image of positive objects.
        negative_data (array): 3D array containing more than one image of negative objects.
        img_num_channels (int): The number of filters used. Defaults to 1.
        normalize (bool, optional): True will normalize the data using the input min and max pixels
        min_pixel (int, optional): The minimum pixel count, defaults to 638. 
            Pixels with counts below this threshold will be set to this limit.
        max_pixel (int, optional): The maximum pixel count, defaults to 3000. 
            Pixels with counts above this threshold will be set to this limit.
    
    Returns:      
        Reshaped data and label arrays.
    """

    class1_data, class1_label = process_class(positive_data, label=1, normalize=normalize, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)
    class2_data, class2_label = process_class(negative_data, label=0, normalize=normalize, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)
    
    training_data = np.r_[class1_data, class2_data]
    training_labels = np.r_[class1_label, class2_label]

    return training_data, training_labels