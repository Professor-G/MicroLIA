#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 10:38:20 2023

@author: daniel
"""
from MicroLIA.data_processing import crop_image, concat_channels
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from scipy.ndimage.interpolation import zoom
from imblearn.over_sampling import SMOTE
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from warnings import warn
import numpy as np
import random
import cv2

def augmentation(channel1, channel2=None, channel3=None, batch=1, width_shift=0, height_shift=0, horizontal=False, vertical=False, 
    rotation=False, fill='nearest', image_size=None, zoom_range=None, mask_size=None, num_masks=None, blend_multiplier=0, blending_func='mean', 
    num_images_to_blend=2, skew_angle=0, return_stacked=False):
    """
    This function takes in a set of images, up to three filters. The ``channel1`` to ``channel3`` arguments
    are 3D arrays containing individual 2D images, all from the same band. 
    NOte that even though the filters must be input individually each time, the output
    can designated to a 4-D array if ``return_stacked``=``True``. The default is ``False``, which
    will make it so the function returns the same number of outputs as channel inputs. The number 
    of augmentations to perform PER INDIVIDUAL SAMPLE is determined by the ``batch`` argument. 

    Rotating (``rotation``), skewing (``skew_angle``), and flipping images (``horizontal`` & ``vertical``) can 
    make the training model more robust to variations in the orientation and perspective of the input images. 
    Likewise, shifting up (``widtht_shift``) and down (``height_shift``) will help make the model translation 
    invariant and thus robust to the position of the object of interest within the image.

    Image blending (``blend_multiplier``) can help to generate new samples through the combination of different images 
    using a variety of blending criteria (``blend_func``). Note that by default two random images (``num_images_to_blend``)
    will be blended together to create one synthetic sample, and since this procedure is applied post-batch creation,
    the same unique sample may be randomly blended, which could be a problem if the configured augmentation parameters
    do not generate sufficient training feature variety.

    Random cutouts (``mask_size``) can help increase the diversity of the training set and reduce overfitting, as applying
    this technique prevents the training model from relying too heavily on specific features of the image, thus 
    encouraging the model to learn more general image attributes.
    
    These techniques, when enabled, are applied in the following order:
        - Random shift + flip + rotation: Generates ``batch`` number of images.
        - Random zoom in or out.
        - If ``image_size`` is set, the image is resized so as to crop the distorted boundary.
        - Random image skewness is applied, with the ``skew_angle`` controlling the maximum angle,
            in degrees, to distort the image from its original position.
        - The batch size is now increased by a factor of ``blend_multiplier``, where each unique sample is generated
            by randomly merging ``num_images_to_blend`` together according to the blending function ``blend_func``. 
            As per the random nature, an original sample may be blended together at this stage,
            but with enough variation this may not be a problem.
        - Circular cutouts of size ``mask_size`` are randomly placed in the image, whereby
            the cutouts replace the pixel values with zeroes. Note that as per the random nature
            of the routine, if ``num_masks`` is greater than 1, overlap between each cutout may occur,
            depending on the corresponding image size to ``mask_size`` ratio.
    Note:
        This function is used for offline data augmentation! In practice, online augmentation may be preferred 
        as that exposes the training model to significantly more samples. If multiple channels are input,
        this method will save the seeds from the augmentation of the first filter, after which the seeds 
        will be applied to the remaining filters, thus ensuring the same augmentation procedure is applied across all channels.

    Args:
        channel1 (ndarray): 2D array of containing a single image, or a 3D array containing
            multiple images. 
        channel2 (ndarray, optional): 2D array of containing a single image, or a 3D array containing
            multiple images. Must correspond with channel1. Default is None.
        channel3 (ndarray, optional): 2D array of containing a single image, or a 3D array containing
            multiple images. Must correspond with channel2. Default is None.
        batch (int): How many augmented images to create and save, per individual smaple in the input data.
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
            the image in its original size is returned. Defaults to None.
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

    Returns:
        Array containing the augmented images. When input, channel2 and channel3 yield 
        additionl outputs, respectively.
    """

    if batch == 0: #Setting this in case the negative class is not set to be augmented during the CNN optimization routine.
        if channel2 is None:
            return channel1 
        else:
            if channel3 is None:
                return channel1, channel2
            else:
                return channel1, channel2, channel3

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
        return rotate(data, np.random.choice(range(rotation+1), 1)[0], reshape=False, order=0, prefilter=True) #Prefilter is useful but slows things down slightly
        
    #Tensorflow Image Data Generator with shifts and flips. While fill is also an option, in practice it is best to simply crop an oversized image 
    datagen = ImageDataGenerator(
        width_shift_range=width_shift,
        height_shift_range=height_shift,
        horizontal_flip=horizontal,
        vertical_flip=vertical,
        fill_mode=fill
        )

    #The rotation child function is only added to the Image Data Generator if rotation parameter is input
    if rotation != 0:
        datagen.preprocessing_function = image_rotation

    if len(channel1.shape) == 3: 
        data = np.array(np.expand_dims(channel1, axis=-1))
    elif len(channel1.shape) == 2:
        data = np.array(np.expand_dims(channel1, axis=-1))
        data = data.reshape((1,) + data.shape)
    else:
        raise ValueError("Input data must be 2D for single sample or 3D for multiple samples")

    augmented_data, seeds = [], [] #Seeds will store the rotation/translation/shift and/or zoom augmentations for multi-band reproducibility
    for i in np.arange(0, len(data)):
        original_data = data[i].reshape((1,) + data[-i].shape)
        for j in range(batch):
            seed = int(random.sample(range(int(1e9)), 1)[0])
            seeds.append(seed)
            augment = datagen.flow(original_data, batch_size=1, seed=seed) #returns 3D (width, height, num)
            augmented_data_batch = augment.next()[0]
            width, height = augmented_data_batch.shape[:2]
            augmented_data_reshaped = np.reshape(augmented_data_batch, (width, height))
            if zoom_range is not None:
                augmented_data.append(resize(random_zoom(augmented_data_reshaped, zoom_min=zoom_range[0], zoom_max=zoom_range[1], seed=seed), image_size))
            else:
                augmented_data.append(resize(augmented_data_reshaped, image_size))

    augmented_data = np.array(augmented_data)

    if skew_angle != 0:
        seeds_skew, a = [], [] #Individual images will be input independently to the a list to ensure proper seed use
        for i in range(len(augmented_data)):
            seed = int(random.sample(range(int(1e9)), 1)[0])
            seeds_skew.append(seed)
            a.append(random_skew(augmented_data[i], max_angle=skew_angle, seed=seed))
        augmented_data = np.array(a)

    if blend_multiplier >= 1:
        seeds_blend, a = [], [] #Individual images will be input independently to a list along with the seed to ensure reproducibility across all channels
        for i in range(int(blend_multiplier*len(augmented_data))):
            seed = int(random.sample(range(int(1e9)), 1)[0])
            seeds_blend.append(seed)
            blended_image = image_blending(augmented_data, num_augmentations=1, blending_func=blending_func, num_images_to_blend=num_images_to_blend, seed=seed)
            a.append(blended_image[0])
        augmented_data = np.array(a)

    if mask_size is not None:
        seeds_mask, a = [], [] #Individual images will be input independently to the a list to ensure proper seed use
        for i in range(len(augmented_data)):
            seed = int(random.sample(range(int(1e9)), 1)[0])
            seeds_mask.append(seed)
            a.append(random_cutout(augmented_data[i], mask_size=mask_size, num_masks=num_masks, seed=seed))
        augmented_data = np.array(a)

    if channel2 is None:
        return augmented_data
    else:
        if len(channel2.shape) == 3: 
            data = np.array(np.expand_dims(channel2, axis=-1))
        elif len(channel2.shape) == 2:
            data = np.array(np.expand_dims(channel2, axis=-1))
            data = data.reshape((1,) + data.shape)
     
        augmented_data2, k = [], 0
        for i in np.arange(0, len(data)):
            original_data = data[i].reshape((1,) + data[-i].shape)
            for j in range(batch):
                augment = datagen.flow(original_data, batch_size=1, seed=seeds[k])
                augmented_data_batch = augment.next()[0]
                width, height = augmented_data_batch.shape[:2]
                augmented_data_reshaped = np.reshape(augmented_data_batch, (width, height))
                if zoom_range is not None:
                    augmented_data2.append(resize(random_zoom(augmented_data_reshaped, zoom_min=zoom_range[0], zoom_max=zoom_range[1], seed=seeds[k]), image_size))
                else:
                    augmented_data2.append(resize(augmented_data_reshaped, image_size))
                k += 1

    augmented_data2 = np.array(augmented_data2)

    if skew_angle != 0:
        a = [] #Individual images will be input independently to ensure proper seed use
        for i in range(len(augmented_data2)):
            a.append(random_skew(augmented_data2[i], max_angle=skew_angle, seed=seeds_skew[i]))
        augmented_data2 = np.array(a)

    if blend_multiplier >= 1:
        a = [] #Individual images will be input independently to a list to ensure reproducibility across all channels
        for i in range(int(blend_multiplier*len(augmented_data2))):
            blended_image = image_blending(augmented_data2, num_augmentations=1, blending_func=blending_func, num_images_to_blend=num_images_to_blend, seed=seeds_blend[i])
            a.append(blended_image[0])
        augmented_data2 = np.array(a)

    if mask_size is not None:
        a = [] #Individual images will be input independently to ensure proper seed use
        for i in range(len(augmented_data2)):
            a.append(random_cutout(augmented_data2[i], mask_size=mask_size, num_masks=num_masks, seed=seeds_mask[i]))
        augmented_data2 = np.array(a)

    if channel3 is None:
        if return_stacked:
            return concat_channels(augmented_data, augmented_data2)
        else:
            return augmented_data, augmented_data2
    else:
        if len(channel3.shape) == 3: 
            data = np.array(np.expand_dims(channel3, axis=-1))
        elif len(channel3.shape) == 2:
            data = np.array(np.expand_dims(channel3, axis=-1))
            data = data.reshape((1,) + data.shape)
   
        augmented_data3, k = [], 0
        for i in np.arange(0, len(data)):
            original_data = data[i].reshape((1,) + data[-i].shape)
            for j in range(batch):
                augment = datagen.flow(original_data, batch_size=1, seed=seeds[k])
                augmented_data_batch = augment.next()[0]
                width, height = augmented_data_batch.shape[:2]
                augmented_data_reshaped = np.reshape(augmented_data_batch, (width, height))
                if zoom_range is not None:
                    augmented_data3.append(resize(random_zoom(augmented_data_reshaped, zoom_min=zoom_range[0], zoom_max=zoom_range[1], seed=seeds[k]), image_size))
                else:
                    augmented_data3.append(resize(augmented_data_reshaped, image_size))
                k += 1

    augmented_data3 = np.array(augmented_data3)

    if skew_angle != 0:
        a = [] #Individual images will be input independently to ensure proper seed use
        for i in range(len(augmented_data3)):
            a.append(random_skew(augmented_data3[i], max_angle=skew_angle, seed=seeds_skew[i]))
        augmented_data3 = np.array(a)

    if blend_multiplier >= 1:
        a = [] #Individual images will be input independently to a list to ensure reproducibility across all channels
        for i in range(int(blend_multiplier*len(augmented_data3))):
            blended_image = image_blending(augmented_data3, num_augmentations=1, blending_func=blending_func, num_images_to_blend=num_images_to_blend, seed=seeds_blend[i])
            a.append(blended_image[0])
        augmented_data3 = np.array(a)

    if mask_size is not None:
        a = [] #Individual images will be input independently to ensure proper seed use
        for i in range(len(augmented_data3)):
            a.append(random_cutout(augmented_data3[i], mask_size=mask_size, num_masks=num_masks, seed=seeds_mask[i]))
        augmented_data3 = np.array(a)

    if return_stacked:
        return concat_channels(augmented_data, augmented_data2, augmented_data3)
    else:
        return augmented_data, augmented_data2, augmented_data3

def random_cutout(images, mask_size=16, num_masks=1, seed=None, mask_type='circle'):
    """
    Applies the cutout data augmentation technique to a sample of 2D images.
    This method applies `num_masks` random positioned (mask_size x mask_size) black squares or
    circles to each image.

    Args:
        images (numpy array): A 3D array of shape (num_images, height, width).
        mask_size (int): The size of the cutout mask. Defaults to 16.
        num_masks (int): Number of masks to apply to each image. Defaults to 1.
        seed (int): Seed for the random number generator. Defaults to None.
        mask_type (str): Type of mask to create. Can be 'square' or 'circle'. Defaults to 'square'.

    Returns:
        A new 3D array of the same shape as data, with cutout applied.
    """

    if seed is not None:
        np.random.seed(seed)

    if images.ndim == 3:
        num_images, height, width = images.shape
    elif images.ndim == 2:
        height, width = images.shape
        num_images = 1
    else:
        raise ValueError('Input array must be either 2D (single image) or 3D (multiple images)')

    #Reshape input from (num_images, height, width) to (num_images, height, width, 1)
    images = images.reshape(-1, height, width, 1)

    new_images = np.copy(images)

    for i in range(num_images):
        for j in range(num_masks):
            if mask_type == 'square':
                if height - 2*mask_size > 0 and width - 2*mask_size > 0:
                    h = np.random.randint(mask_size, height - mask_size)
                    w = np.random.randint(mask_size, width - mask_size)
                    new_images[i, h-mask_size:h+mask_size, w-mask_size:w+mask_size, :] = 0
                else:
                    raise ValueError('Mask size is too large for the image input!')
            elif mask_type == 'circle':
                if height - 2*mask_size > 0 and width - 2*mask_size > 0:
                    h = np.random.randint(mask_size, height - mask_size)
                    w = np.random.randint(mask_size, width - mask_size)
                    y, x = np.ogrid[-h:height-h, -w:width-w]
                    mask = x*x + y*y <= mask_size*mask_size
                    new_images[i][mask, :] = 0
                else:
                    raise ValueError('Mask size is too large for the image input!')
            else:
                raise ValueError('Invalid mask_type, options are "square" or "circle".')

    #Reshape output from (num_images, height, width, 1) to (num_images, height, width)
    new_images = new_images.reshape(num_images, height, width)
    if num_images == 1:
        return new_images[0]
    else:
        return new_images

def image_blending(images, num_augmentations=1, blend_ratio=0.5, blending_func='mean', normalize_blend=True,
    num_images_to_blend=5, seed=None):
    """
    Perform image blending augmentation on a set of single-band images, combining up to num_images_to_blend images to generate each augmentation.
    
    After all the blended images are generated, the code normalizes each image by dividing each pixel value by 
    the number of images that were blended to create that image. By dividing the sum of pixel values by the 
    number of images blended, the resulting pixel values will be the average of the corresponding pixel values 
    in the original images. This ensures that the overall intensity of the resulting image is similar to the 
    original images and does not become too bright or too dark due to the blending process.
    
    Note:
        The blend_ratio is a parameter that determines the proportion of the two images that are blended together.

        For example, suppose we have two images A and B that we want to blend. If blend_ratio is set to 0.5, then the 
        resulting blended image will be a 50-50 mix of A and B. Specifically, each pixel in the blended image will be computed as:

        blended_pixel = 0.5 * pixel_A + 0.5 * pixel_B

        If blend_ratio is set to 0.7, then the resulting blended image will contain 70% of A and 30% of B. Specifically, 
        each pixel in the blended image will be computed as:

        blended_pixel = 0.7 * pixel_A + 0.3 * pixel_B

        Thus the blend_ratio determines the weighting of each image in the blend, with values between 0 and 1 indicating the proportion of each image.

    Args:
        images (numpy array): A 3D array of shape (num_images, height, width).
        num_augmentations (int): The number of augmented images to generate.
        blend_ratio (float): The proportion of the two images to blend together. Must be between 0 and 1.
        blending_func (str): The blending function to use. Options are 'mean', 'max', 'min', and 'random'.
        num_images_to_blend (int): The number of images to randomly select for blending. Defaults to 2.
        seed (int): Seed for the random number generator. Defaults to None.

    Returns:
        ndarray: A 3D array of the augmented images, with dimensions (num_images, height, width).
    """

    #assert images.ndim != 3, "Input images must have dimensions (num_images, height, width)"
    assert isinstance(num_augmentations, int) and num_augmentations > 0, "num_augmentations must be a positive integer"
    assert 0 <= blend_ratio <= 1, "blend_ratio must be between 0 and 1"
    assert isinstance(num_images_to_blend, int) and num_images_to_blend > 0 and num_images_to_blend <= len(images), "num_images_to_blend must be a positive integer less than or equal to the number of input images"
    
    #Define blending function
    if blending_func == 'mean':
        blend_func = lambda x, y: (1 - blend_ratio) * x + blend_ratio * y
    elif blending_func == 'max':
        blend_func = lambda x, y: np.maximum(x.astype(np.float64), y.astype(np.float64)).astype(x.dtype)
    elif blending_func == 'min':
        blend_func = lambda x, y: np.minimum(x.astype(np.float64), y.astype(np.float64)).astype(x.dtype)
    elif blending_func == 'random':
        random_func = random.choice(['mean', 'max', 'min'])
        if random_func == 'mean':
            blend_func = lambda x, y: (1 - blend_ratio) * x + blend_ratio * y
        elif random_func == 'max':
            blend_func = lambda x, y: np.maximum(x.astype(np.float64), y.astype(np.float64)).astype(x.dtype)
        elif random_func == 'min':
            blend_func = lambda x, y: np.minimum(x.astype(np.float64), y.astype(np.float64)).astype(x.dtype) 
    else:
        raise ValueError(f"Blending function '{blending_func}' not recognized, options are 'mean', 'max', 'min', or 'random'.")

    if seed is not None:
        np.random.seed(seed)
    
    #Initialize output array
    if images.ndim == 3:
        num_images, height, width = images.shape
    elif images.ndim == 2:
        height, width = images.shape 
        num_images = 1
    else:
        raise ValueError('Incorrect input shape!')

    output_images = np.zeros((num_augmentations, height, width), dtype=np.float32)
    
    #Perform image blending augmentation
    for i in range(num_augmentations):
        #Randomly select number of images to blend with, up to the num_images_to_blend
        num_images_selected = np.random.randint(2, num_images_to_blend+1)
        blend_indices = np.random.choice(num_images, size=num_images_selected, replace=False)
        blend_images = images[blend_indices]
        #Apply image blending
        blended_image = blend_images[0]
        for j in range(1, num_images_selected):
            blended_image = blend_func(blended_image, blend_images[j])
        output_images[i, :, :] += blended_image.astype(np.float32)

        if normalize_blend: #Normalize the blended images to avoid overlapping extreme pixels
            output_images[i, :, :] /= num_images_selected

    return output_images

def smote_oversampling(X_train, Y_train, smote_sampling=1, binary=True, seed=None):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) oversampling to the training data to address class imbalance.

    SMOTE generates synthetic samples by interpolating between neighboring minority class samples. It does this by choosing 
    two samples from the minority class at random, then choosing a random point between them and adding it as a new sample. 
    By doing this, SMOTE creates new synthetic samples that are similar to existing samples in the minority class, while 
    introducing some degree of randomness to ensure that the synthetic samples are not exact duplicates.
    
    If binary=True, the function assumes binary classification only therefore only expects two classes. 

    Note:
        The smote_sampling options are defined as follows:
        
        0-1: 0 means no sampling applied, if 0.5, then the minority class will be oversampled using the SMOTE 
        algorithm until it is 50% the size of the majority class, whereas if it's set to 0.75 it will be oversampled
        until the minority class is 75% the size of the majority, etc. Maximum value is 1.

        If using the following string options the number of samples in both classes will always be equalized.

        "minority": This is the default option, and it generates synthetic samples only for the minority class.

        "not minority": This option generates synthetic samples for all classes except the majority class.

        "not majority": This option generates synthetic samples for all classes except the minority class.

        "all": This option generates synthetic samples for all classes.

    Args:
        X_train (numpy array): A 4D array of shape (num_images, height, width, num_channels) containing the training data.
        Y_train (numpy array): A 1D array of shape (num_images,) containing the training labels.
        smote_sampling (str or float): Determines the target ratio of samples in the minority class after resampling. 
            Valid options are 'minority', 'not minority', 'not majority', 'all', or a float in the range (0, 1) that 
            specifies the desired ratio of minority class samples. Defaults to 1.
        binary (bool): Set to True if there are only two classes, used to one-hot-encode the labels array. Defaults to True.
        seed (int): Seed for the random number generator. Defaults to None.

    Returns:
        tuple: A tuple containing the oversampled X_train and Y_train arrays.

    Raises:
        ValueError: If X_train and Y_train have incompatible shapes or if smote_sampling is not a valid option.
    """

    if smote_sampling == 0:
        return X_train, Y_train

    if X_train.shape[0] != Y_train.shape[0]:
        raise ValueError("X_train and Y_train must have the same number of samples")
    
    #Check smote_sampling parameter
    valid_sampling_options = ['minority', 'not minority', 'not majority', 'all']
    if isinstance(smote_sampling, str) and smote_sampling not in valid_sampling_options:
        raise ValueError(f"Invalid smote_sampling option '{smote_sampling}'. Valid options are {valid_sampling_options}, or a float in the range (0, 1)")
    elif isinstance(smote_sampling, float) and (smote_sampling < 0 or smote_sampling > 1):
        raise ValueError("smote_sampling must be a float in the range (0, 1)")
    
    #Reshape X_train to 2D array
    if len(X_train.shape) == 4:
        num_images, height, width, num_channels = X_train.shape
    else:
        num_images, height, width = X_train.shape
        num_channels = 1

    X_2d = np.reshape(X_train, (num_images, -1))
    
    #Apply SMOTE oversampling
    smote = SMOTE(sampling_strategy=smote_sampling, random_state=seed)
    X_resampled, Y_resampled = smote.fit_resample(X_2d, Y_train)
    
    #Reshape X_resampled to 4D array
    num_resampled = X_resampled.shape[0]
    X_train_resampled = np.reshape(X_resampled, (num_resampled, height, width, num_channels))
    if binary:
        Y_resampled = to_categorical(Y_resampled, num_classes=2)

    return X_train_resampled, Y_resampled

def resize(data, size=50):
    """
    Resizes the data by cropping out the outer boundaries outside the size x size limit.
    Can be either a 2D array containing one sample, or a 3D array for multiple samples.
    
    Note:
        By design this function will not work if the data is a single sample, multiple channels (img_width, img_height, img_num_channels). 
        In this case, reshape to be 4-D: data = data.reshape(1, img_width, img_height, img_num_channels)

    Args:
        data (array): 2D array to resize.
        size (int): The length/width of the output array. Defaults to 50 pixels. 
            Can be set to None to return the same image size.

    Returns:
        The cropped out array.
    """

    if size is None:
        return data 

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

def random_skew(image, max_angle=15, intensity=0.1, seed=None):
    """
    Apply random skewness to a 2D image.

    Args: 
        image (array): The input 2D image to be skewed.
        max_angle (float): The maximum absolute value of the skew angle, 
            in degrees. Defaults to 15.
        intensity (float): The maximum amount of skew to apply. A lower intensity 
            value will result in a more subtle skew, while a higher intensity value 
            will result in a more significant skew. Defaults to 0.1. Must be <= 1.
        seed (int): Seed for the random number generator. Defaults to None.

    Returns:
        The skewed image.
    """

    if seed is not None:
        np.random.seed(seed)

    skew_x = np.random.uniform(-intensity, intensity)
    skew_y = np.random.uniform(-intensity, intensity)

    # Get image dimensions
    rows, cols = image.shape

    # Define source points for affine transformation
    src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])

    # Define destination points for affine transformation
    max_skew_angle = np.deg2rad(max_angle)
    dst_x = np.random.uniform(-max_skew_angle, max_skew_angle)
    dst_y = np.random.uniform(-max_skew_angle, max_skew_angle)
    dst_x = skew_x * cols + dst_x * rows
    dst_y = skew_y * rows + dst_y * cols
    dst_points = src_points + np.float32([[dst_x, dst_y], [dst_x + cols, dst_y], [dst_x, dst_y + rows]])

    # Define affine transformation matrix
    matrix = cv2.getAffineTransform(src_points, dst_points)

    # Apply affine transformation
    skewed_image = cv2.warpAffine(image, matrix, (cols, rows))

    return skewed_image

def random_zoom(images, zoom_min=0.9, zoom_max=1.1, seed=None):
    """
    Randomly applies zoom in/out to a 3D array of images (or 2D for a single image)
    along the height and width dimensions.

    Args:
        images (ndarray): 2D or 3D numpy array of shape (height, width) or (num_images, height, width)
        zoom_min (float): Minimum ratio to decrease the image by. Defaults to 0.9.
        zoom_max (float): Maximum ratio to increase the image by. Defaults to 1.1.
        seed (int): Seed for the random number generator. Defaults to None.

    Returns:
        2D or 3D numpy array of image(s) with the random zoom applied.
    """

    if seed is not None:
        np.random.seed(seed)

    if images.ndim == 2:
        images = np.expand_dims(images, axis=0)

    zoom_factor = np.random.uniform(zoom_min, zoom_max)
    zoomed_images = zoom(images, zoom=(1.0, zoom_factor, zoom_factor), mode='nearest', order=0, prefilter=True)

    if zoomed_images.shape[0] == 1:
        zoomed_images = np.squeeze(zoomed_images, axis=0)

    return zoomed_images

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