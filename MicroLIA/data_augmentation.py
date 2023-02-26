#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 10:38:20 2023

@author: daniel
"""
from MicroLIA.data_processing import crop_image, concat_channels
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from warnings import warn
import numpy as np
import random

def augmentation(channel1, channel2=None, channel3=None, batch=10, width_shift=5, height_shift=5, 
    horizontal=True, vertical=True, rotation=True, fill='nearest', image_size=50, mask_size=None, num_masks=None):
    """
    This function takes in one image and applies data augmentation techniques.
    Shifts and rotations occur at random, for example, if width_shift is set
    to 10, then an image shift between -10 and 10 pixels will be chosen from a 
    random uniform distribution. Rotation angle is also chosen from a random uniform 
    distribution, between zero and 360. 

    Note:
        This function is used for offline data augmentation! In practice,
        online augmentation may be preferred as that exposes the CNN
        to significantly more samples. If multiple channels are input,
        this method will save the seeds from the augmentation of the first 
        channel, after which the seeds will be applied to the remaining channels,
        thus ensuring the same augmentation procedure is applied across all filters.

    Args:
        channel1 (ndarray): 2D array of containing a single image, or a 3D array containing
            multiple images. 
        channel2 (ndarray, optional): 2D array of containing a single image, or a 3D array containing
            multiple images. Must correspond with channel1. Default is None.
        channel3 (ndarray, optional): 2D array of containing a single image, or a 3D array containing
            multiple images. Must correspond with channel2. Default is None.
        batch (int): How many augmented images to create and save.
        width_shift (int): The max pixel shift allowed in either horizontal direction.
            If set to zero no horizontal shifts will be performed. Defaults to 5 pixels.
        height_shift (int): The max pixel shift allowed in either vertical direction.
            If set to zero no vertical shifts will be performed. Defaults to 5 pixels.
        horizontal (bool): If False no horizontal flips are allowed. Defaults to True.
        vertical (bool): If False no vertical reflections are allowed. Defaults to True.
        rotation (int): If True full 360 rotation is allowed, if False no rotation is performed.
            Defaults to True.
        fill (str): This is the treatment for data outside the boundaries after roration
            and shifts. Default is set to 'nearest' which repeats the closest pixel values.
            Can set to: {"constant", "nearest", "reflect", "wrap"}.
        image_size (int, bool): The length/width of the cropped image. This can be used to remove
            anomalies caused by the fill (defaults to 50). This can also
            be set to None in which case the image in its original size is returned.

    Returns:
        Array containing the augmented images. When input, channel2 and channel3 yield 
        additionl outputs, respectively.
    """

    if isinstance(width_shift, int) == False or isinstance(height_shift, int) == False:
        raise ValueError("Shift parameters must be integers indicating +- pixel range")
    if mask_size is not None:
        if num_masks is None:
            raise ValueError('Need to input num_masks parameter.')
    if num_masks is not None:
        if mask_size is None:
            raise ValueError('Need to input mask_size parameter.')

    if rotation:
        rotation = 360
    else:
        rotation = 0

    def image_rotation(data):
        """
        Function for the image data genereation which hardcodes the rotation parameter of the parent function.
        The order parameter to 0 to ensure that the rotation is performed using nearest-neighbor interpolation, 
        which minimizes the amount of distortion introduced into the image. Additionally, the reshape parameter is False, 
        which will ensure that the rotated image has the same shape as the original image. The prefilter parameter is also 
        set to False to prevent any additional processing from being performed on the image prior to rotation.
        """
        return rotate(data, np.random.choice(range(rotation+1), 1)[0], reshape=False, order=0, prefilter=False)
        
    #Tensorflow Image Data Generator with shifts and flips. While fill is also an option, in practice it is best to crop an oversized image 
    datagen = ImageDataGenerator(
        width_shift_range=width_shift,
        height_shift_range=height_shift,
        horizontal_flip=horizontal,
        vertical_flip=vertical,
        fill_mode=fill)

    #The rotation child function is only added to the Image Data Generator if rotation parameter is input
    if rotation != 0:
        datagen.preprocessing_function = image_rotation

    if len(channel1.shape) != 4:
        if len(channel1.shape) == 3: 
            data = np.array(np.expand_dims(channel1, axis=-1))
        elif len(channel1.shape) == 2:
            data = np.array(np.expand_dims(channel1, axis=-1))
            data = data.reshape((1,) + data.shape)
        else:
            raise ValueError("Input data must be 2D for single sample or 3D for multiple samples")

    augmented_data, seeds = [], []
    for i in np.arange(0, len(data)):
        original_data = data[i].reshape((1,) + data[-i].shape)
        for j in range(batch):
            seed = int(random.sample(range(int(1e6)), 1)[0])
            seeds.append(seed)
            augment = datagen.flow(original_data, batch_size=1, seed=seed)
            if mask_size is not None:
                augmented_data.append(random_cutout(augment[0][0], mask_size=mask_size, num_masks=num_masks, seed=seed, augmenting=True))
            else:
                augmented_data.append(augment[0][0])

    augmented_data = resize(np.array(augmented_data), size=image_size)

    if channel2 is None:
        return augmented_data
    else:
        seeds = np.array(seeds)
        if len(channel2.shape) != 4:
            if len(channel2.shape) == 3: 
                data = np.array(np.expand_dims(channel2, axis=-1))
            elif len(channel2.shape) == 2:
                data = np.array(np.expand_dims(channel2, axis=-1))
                data = data.reshape((1,) + data.shape)

        k=0
        augmented_data2 = []
        for i in np.arange(0, len(data)):
            original_data = data[i].reshape((1,) + data[-i].shape)
            for j in range(batch):
                augment = datagen.flow(original_data, batch_size=1, seed=seeds[k])
                if mask_size is not None:
                    augmented_data2.append(random_cutout(augment[0][0], mask_size=mask_size, num_masks=num_masks, seed=seeds[k], augmenting=True))
                else:
                    augmented_data2.append(augment[0][0])
                k+=1

        augmented_data2 = resize(np.array(augmented_data2), size=image_size)

    if channel3 is None:
        return augmented_data, augmented_data2
    else:
        if len(channel3.shape) != 4:
            if len(channel3.shape) == 3: 
                data = np.array(np.expand_dims(channel3, axis=-1))
            elif len(channel3.shape) == 2:
                data = np.array(np.expand_dims(channel3, axis=-1))
                data = data.reshape((1,) + data.shape)

        k=0
        augmented_data3 = []
        for i in np.arange(0, len(data)):
            original_data = data[i].reshape((1,) + data[-i].shape)
            for j in range(batch):
                augment = datagen.flow(original_data, batch_size=1, seed=seeds[k])
                if mask_size is not None:
                    augmented_data3.append(random_cutout(augment[0][0], mask_size=mask_size, num_masks=num_masks, seed=seeds[k], augmenting=True))
                else:
                    augmented_data3.append(augment[0][0])
                k+=1

    augmented_data3 = resize(np.array(augmented_data3), size=image_size)

    return augmented_data, augmented_data2, augmented_data3

def resize(data, size=50):
    """
    Resizes the data by cropping out the outer boundaries outside the size x size limit.
    Can be either a 2D array containing one sample, or a 3D array for multiple samples.

    Args:
        data (array): 2D array
        size (int): length/width of the output array. Defaults to 50 pixels.

    Returns:
        The cropped out array
    """

    if len(data.shape) == 3 or len(data.shape) == 4:
        width = data[0].shape[0]
        height = data[0].shape[1]
    elif len(data.shape) == 2:
        width = data.shape[0]
        height = data.shape[1]
    else:
        raise ValueError("Channel cannot be one dimensional")

    if width != height:
        raise ValueError("Can only resize square images")
    if width == size:
        #print("No resizing necessary, image shape is already in desired size, returning original data...")
        return data 

    if len(data.shape) == 2:
        resized_data = crop_image(np.array(np.expand_dims(data, axis=-1))[:, :, 0], int(width/2.), int(height/2.), size)
        return resized_data
    else:
        resized_images = [] 
        filter1, filter2, filter3 = [], [], []
        for i in np.arange(0, len(data)):
            if len(data[i].shape) == 2:
                resized_images.append(crop_image(np.array(np.expand_dims(data[i], axis=-1))[:, :, 0], int(width/2.), int(height/2.), size))
            elif len(data[i].shape) == 3:
                if data[i].shape[-1] >= 1:
                    filter1.append(crop_image(data[i][:, :, 0], int(width/2.), int(height/2.), size))
                if data[i].shape[-1] >= 2:
                    filter2.append(crop_image(data[i][:, :, 1], int(width/2.), int(height/2.), size))
                if data[i].shape[-1] == 3:
                    filter3.append(crop_image(data[i][:, :, 2], int(width/2.), int(height/2.), size))    
                if data[i].shape[-1] > 3:
                    raise ValueError('A maximum of 3 filters is currently supported!')            
            else:
                raise ValueError('Invalid data input size, the images must be shaped as follows (# of samples, width, height, filters)')

        if len(filter1) != 0:
            for j in range(len(filter1)):
                if data[i].shape[-1] == 1:
                    resized_images.append(filter1[j])
                elif data[i].shape[-1] == 2:
                    resized_images.append(concat_channels(filter1[j], filter2[j]))
                elif data[i].shape[-1] == 3:
                    resized_images.append(concat_channels(filter1[j], filter2[j], filter3[j]))
                
    resized_data = np.array(resized_images)

    return resized_data

def random_cutout(images, mask_size=16, num_masks=1, seed=None, augmenting=False):
    """
    Applies the cutout data augmentation technique to a sample of 2D images.
    This method applies `num_masks` random positioned (mask_size x mask_size) black squares to each image.

    Args:
        data (numpy array): A 3D array of shape (num_images, height, width).
        mask_size (int): The size of the cutout mask. Defaults to 16.
        num_masks (int): Number of masks to apply to each image. Defaults to 1.
        seed (int): Seed for the random number generator. Defaults to None.
        augmenting (bool): If False the image shape is assumed to be (num_img, height, width), if True
            it is assumed to be reversed. Defaults to False.

    Returns:
        A 3D array of the same shape as data, with cutout applied.
    """

    #images = copy.deepcopy(data)

    if augmenting: #The data augmentation procedure requires the input to be inversed.
        width, height, num_images = images.shape
    else:
        num_images, height, width = images.shape

    if seed is not None:
        np.random.seed(seed)

    for i in range(num_images):
        for j in range(num_masks):
            if height - mask_size > 0:
                h = np.random.randint(height - mask_size)
                w = np.random.randint(width - mask_size)
                images[i, h:h+mask_size, w:w+mask_size] = 0
            else:
                raise ValueError('WARNING: Mask size is too large for the image input!')

    np.random.seed(1909) #Set back to MicroLIA default

    return images

def plot(data, cmap='gray', title=''):
    """
    Plots 2D array using a robust colorbar range to
    ensure proper visibility.
    
    Args:
        data (array): 2D array for single image, or 3D array with stacked channels.
        cmap (str): Colormap to use when generating the image.
        title (str, optional): Title displayed above the image. 

    Returns:
        AxesImage.
    """
    
    index = np.where(np.isfinite(data))
    std = np.median(np.abs(data[index]-np.median(data[index])))
    vmin = np.median(data[index]) - 3*std
    vmax = np.median(data[index]) + 10*std
    
    plt.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.title(title)
    plt.show()

