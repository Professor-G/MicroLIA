.. LIA documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:15:14 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MicroLIA's documentation!
===============================

MicroLIA is an open-source program for microlensing detection in wide-field surveys. This engine uses the machine learning RF model using a variety of lightcurve statistics, therefore a training set with survey parameters must be created before MicroLIA can begin its query. See example below.

Installation
==================

The current stable version can be installed via pip:

.. code-block:: bash

    pip install MicroLIA


Example: OGLE IV
==================
The lightcurves for 219 OGLE IV microlensing events can be downloaded here:

Each file contains three columns: time, mag, magerr

We will train MicroLIA for OGLE IV microlensing detection, and record how many of these 219 events we successfully recover.

OGLE IV: Training Set
==================
Adaptive cadence is important as this allows MicroLIA to detect microlensing events even if the survey footprint is sparse. In this example we need to train MicroLIA using OGLE IV cadence, which we can take to be the timestamps of these 219 lightcurves. We will append the time array of each lightcurve to a list.

.. code-block:: python

    import os
    import numpy as np

    filenames = os.listdir('/data')

    timestamps = []

    for name in filenames:
      time = np.loadtxt(name)[:,0]
      timestamps.append(time)


This time list will be used to simulate our training data, as each time an event is simulated a random timestamp from the list which will be chosen. We can now create our training data using the training_set module -- in this example we will set the min_mag of the survey to be 15, and the max_mag to be 20. We will also set n_class=500, this is the size of our training classes.

.. code-block:: python

   from MicroLIA import training_set

   training_set.create(timestamps, min_mag=15, max_mag=20, n_class=500)


There are a number of other parameters we can control when creating the training set, including exposure time and zeropoint of the survey telescope. Setting these parameters carefully will ensure that our training set matches what will be observed. 

To be more accurate we will set these optional parameters, and even include a noise model using the rms and median mag of our OGLE IV data.

.. code-block:: python

   from MicroLIA import training_set
   from MicroLIA import noise_models

   rms_mag = []
   median_mag = []

   for i in filenames:
      mag = np.loadtxt(i)[:,1]
      rms = 0.5*np.abs(np.percentile(mag,84) - np.percentile(mag,16))
      rms_mag.append(rms)
      median_mag.append(np.median(mag))

   ogle_noise = noise_models.create_noise(median, rms)

   training_set.create(timestamps, min_mag=np.min(median_mag), 
      max_mag=np.max(median_mag), noise=ogle_noise, zp=22, 
      exptime=30, n_class=500)

This will simulate the lightcurves for our training set, all of which will be saved in the 'lightcurves.fits' file, organized by ID. The other two files 'pca_features.txt', and 'all_features.txt' include the data that will be used to train our machine learning model. To learn about these files click here.

OGLE IV: Microlensing Classification
==================

We will create our Random Forest machine learning model using the statistical features only:

.. code-block:: python

   from MicroLIA import models

   model = models.create_models('all_features.txt', model='rf')


With the model saved we can begin classifying any lightcurve! Let's load the first OGLE IV microlensing lightcurve and check what the prediction is:

.. code-block:: python

   from MicroLIA import microlensing_classifier

   data = np.loadtxt(filenames[0])
   time, mag, magerr = data[:,0], data[:,1], data[:,2]

   prediction = microlensing_classifier.predict(time, mag, magerr, model)

   print(prediction)

The prediction output is the probability prediction of each class. Finally, let's load all 219 lightcurves and check the overall prediction accuracy:

.. code-block:: python

   predictions = []
   for name in filenames:
      data = np.loadtxt(name)
      time, mag, magerr = data[:,0], data[:,1], data[:,2]

      prediction = microlensing_classifier.predict(time, mag, magerr, model)
      predictions.append(prediction[0][np.argmax(prediction[1])])

   print('total accuracy '+str(len(predictions)/len(np.argwhere(predictions == 'ML'))))

OGLE IV: From start to finish
==================

.. code-block:: python

   import os
   import numpy as np
   from MicroLIA import training_set, microlensing_classifier
   from MicroLIA import models, noise_models

   filenames = os.listdir('/data') #219

   timestamps = [] #
   rms_mag = []
   median_mag = []

   for name in filenames:
      time = np.loadtxt(name)[:,0]
      mag = np.loadtxt(name)[:,1]

      rms = 0.5*np.abs(np.percentile(mag,84) - np.percentile(mag,16))
      timestamps.append(time)
      median_mag.append(np.median(mag))
      rms_mag.append(rms)

   ogle_noise = noise_models.create_noise(median_mag, rms_mag)

   training_set.create(timestamps, min_mag=np.min(median_mag), max_mag=np.max(median_mag), noise=ogle_noise, zp=22, exptime=30, n_class=500)

   model = models.create_models('all_features.txt', model='rf')

   predictions = []
   for name in filenames:
      data = np.loadtxt(name)
      time, mag, magerr = data[:,0], data[:,1], data[:,2]

      prediction = microlensing_classifier.predict(time, mag, magerr, model)
      predictions.append(prediction[0][np.argmax(prediction[1])])

   print('total accuracy '+str(len(predictions)/len(np.argwhere(predictions == 'ML'))))


