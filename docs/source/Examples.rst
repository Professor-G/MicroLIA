.. _Examples:

Example: OGLE II
==================
The lightcurves for 214 OGLE II microlensing events can be downloaded here:

Each file contains three columns: time, mag, magerr

We will train MicroLIA for OGLE II microlensing detection, and record how many of these 214 events we successfully recover.

OGLE II: Training Set
==================
Adaptive cadence is important as this allows MicroLIA to detect microlensing events even if the survey footprint is sparse. In this example we need to train MicroLIA using OGLE IV cadence, which we can take to be the timestamps of these 214 lightcurves. We will append the time array of each lightcurve to a list.

.. code-block:: python

    import os
    import numpy as np
    from pathlib import Path

    path = str(Path.home())+'/OGLE_II/data/'

    filenames = os.listdir(path)

    timestamps = []

    for name in filenames:
      time = np.loadtxt(path+name)[:,0]
      timestamps.append(time)


This time list will be used to simulate our training data, as each time an event is simulated a random timestamp from the list which will be chosen. We can now create our training data using the training_set module -- in this example we will set the min_mag of the survey to be 15, and the max_mag to be 20. We will also set n_class=500, this is the size of our training classes.

.. code-block:: python

   from MicroLIA import training_set

   data_x, data_y = training_set.create(timestamps, min_mag=15, max_mag=20, n_class=500)


There are a number of other parameters we can control when creating the training set, including exposure time and zeropoint of the survey telescope. Setting these parameters carefully will ensure that our training set matches what will be observed. 

To be more accurate we will set these optional parameters, and even include a noise model using the rms and median mag of our OGLE IV data.

.. code-block:: python

   from MicroLIA import training_set, noise_models

   rms_mag = []
   median_mag = []

   for name in filenames:
      mag = np.loadtxt(path+name)[:,1]
      rms = 0.5*np.abs(np.percentile(mag,84) - np.percentile(mag,16))
      rms_mag.append(rms)
      median_mag.append(np.median(mag))

   ogle_noise = noise_models.create_noise(median_mag, rms_mag)

   data_x, data_y = training_set.create(timestamps, min_mag=np.min(median_mag), max_mag=np.max(median_mag), noise=ogle_noise, zp=22, exptime=30, n_class=1000)

This will simulate the lightcurves for our training set, all of which will be saved by default in the 'lightcurves.fits' file, organized by class and ID. The other file is called 'all_features.txt' and contains the statistical metrics of each lightcurve. The first two columns of this file contain the class of each lightcurve and its unique ID, which allows us to inspect the individual lightcurves in the .fits file. This function will return the statistical metrics (data_x) and the corresponding labels (data_y), which can be loaded directly from the 'all_features.txt' file.

There are additional parameters that can be controlled when creating the training set, the most imporant being the arguments that control the "quality" of the simulated microlensing and cataclysmic variable classes. These parameters control the number of data points that must be within the signals, this is especially important to tune if the cadence of the survey is sparse, as per the random nature of the simulations some signals may contain too few points within the transient event to be reasonably detectable. Please refer to the API for more information on these parameters.

`Please refer to the API documentation for more information on these parameters and how to control them <https://microlia.readthedocs.io/en/latest/autoapi/MicroLIA/training_set/index.html>`_.


OGLE II: Classification Engine
==================

We will create our machine learning model using the statistical features of the lightcurves, which are saved by default in the 'all_features.txt' file when we created our training set. The first column is the lightcurve class, and therefore will be loaded as our training labels. The second column is the unique ID of the simulated lightcurve, which will be ignored. 

We can load this file and create our data_x and data_y arrays, although note above that these variables were created when we made our training set. This is just to show how we could generally load the saved training data:

.. code-block:: python
   
   home = str(Path.home())+'/' #By default the file is saved in the home directory

   data = np.loadtxt(home+'all_features.txt', dtype=str)
   data_x = data[:,2:].astype('float')
   data_y = data[:,0]
   
With our training data loaded we can create our machine learning engine with MicroLIA's `models module <https://microlia.readthedocs.io/en/latest/autoapi/MicroLIA/models/index.html>`_.

Unless set otherwise, when creating the model three optimization procedures will automatically run, in the following order:

-  Any missing values (NaN) will be imputed using the `sklearn implementation of the k Nearest Neighbors imputation algorithm <https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html>`_. The imputer will be saved so that it can be applied to transform new, unseen data, therefore solving the issue of missing values. 

-  The features that contain information will be selected using the Boruta algorithm developed by `Kursa and Rudnicki 2011 <https://arxiv.org/pdf/1106.5112.pdf>`_. While bagging algorithms like the Forest Random are robust to irrelevant features, computation-wise it is imperative that we compute only the features that are helpful.

-  Finally, the model hyperparameters will be optimized using the hyperparameter optimization software `Optuna <https://optuna.org/>`_, developed by `Akiba et al 2019 <https://arxiv.org/abs/1907.10902>`_. The default sampler Optuna employs is the Tree Parzen Estimator, a Bayesian optimization approach that effectively reduces the error by narrowing the search space according to the performance of previous iterations.

Since these are turned on by default, we can create and optimize a Random Forest clasifier using the following:

.. code-block:: python

   from MicroLIA import models

   model, imputer, feats_to_use = models.create(data_x, data_y, clf='rf')

Note that MicroLIA currently supports three machine learning algorithms: Random Forest, Extreme Gradient Boosting, and Neural Network. While clf='rf' for Random Forest is the default input, we can also set this to 'xgb' or 'nn'. 

Since neural networks require more tuning to properly identify the optimal number of layers and neurons, it is recommended to set n_iter to at least 100, as by default only 25 trials are performed when optimizing the hyperparameters:

.. code-block:: python

   model, imputer, feats_to_use = models.create(data_x, data_y, clf='nn', n_iter=100)

There has been particular interest in the XGBoost algorithm, which can outperform the Random Forest:

.. code-block:: python

   model, imputer, feats_to_use = models.create(data_x, data_y, clf='xgb')

`For details please refer to the function documentation <https://microlia.readthedocs.io/en/latest/autoapi/MicroLIA/models/index.html#MicroLIA.models.create>`_.


OGLE II: Classification Accuracy
==================

With the optimized model saved, as well as our imputer and indices of features to use, we can begin classifying any lightcurve using the predict() function. Let's load the first OGLE IV microlensing lightcurve and check what the prediction is:

.. code-block:: python

   data = np.loadtxt(filenames[0])
   time, mag, magerr = data[:,0], data[:,1], data[:,2]

   prediction = models.predict(time, mag, magerr, model=model, imputer=imputer, feats_to_use=feats_to_use, convert=True, zp=22)

Note that since by default convert=True, which will convert the magnitude input to flux, therefore we must set the appropriate zeropoint argument. This zp must match whatever value was used when creating the training set, in this example zp=22. 

The prediction output is the lable probability prediction of each class, ordered in alphabetical order:

.. code-block:: python

   print(prediction)

The predicted class in this case is 'ML', as the corresponding classification accuracy of () is higher than all the others. Finally, let's load all 214 lightcurves and check the overall prediction accuracy by selecting whatever class has the largest probability prediction:

.. code-block:: python

   predictions = []
   for name in filenames:
      data = np.loadtxt(path+name)
      time, mag, magerr = data[:,0], data[:,1], data[:,2]

      prediction = models.predict(time, mag, magerr, model=model, imputer=imputer, feats_to_use=feats_to_use, convert=True, zp=22)
      predictions.append(prediction[0][np.argmax(prediction[1])])

   accuracy = len(np.argwhere(predictions == 'ML'))/len(predictions)
   print('Total accuracy :{}'.format(np.round(accuracy, 4)))

OGLE II: From Start to Finish
==================

.. code-block:: python

   import os
   import numpy as np
   from pathlib import Path
   from MicroLIA import training_set, noise_models, models

   path = str(Path.home())+'/OGLE_II/data/'
   filenames = os.listdir(path)

   timestamps = []
   for name in filenames:
      time = np.loadtxt(path+name)[:,0]
      timestamps.append(time)

   rms_mag = []
   median_mag = []

   for name in filenames:
      mag = np.loadtxt(path+name)[:,1]
      rms = 0.5*np.abs(np.percentile(mag,84) - np.percentile(mag,16))
      rms_mag.append(rms)
      median_mag.append(np.median(mag))

   ogle_noise = noise_models.create_noise(median_mag, rms_mag)

   training_set.create(timestamps, min_mag=np.min(median_mag), 
         max_mag=np.max(median_mag), noise=ogle_noise, zp=22, 
         exptime=30, n_class=1000)
   
   home = str(Path.home())+'/' #By default the file is saved in the home directory
   data = np.loadtxt(home+'all_features.txt', dtype=str)

   data_x = data[:,2:].astype('float')
   data_y = data[:,0]
   
   model, imputer, feats_to_use = models.create(data_x, data_y, clf='rf')

   predictions = []
   for name in filenames:
      data = np.loadtxt(path+name)
      time, mag, magerr = data[:,0], data[:,1], data[:,2]

      prediction = models.predict(time, mag, magerr, model=model, 
         imputer=imputer, feats_to_use=feats_to_use, convert=True, zp=22)

      predictions.append(prediction[0][np.argmax(prediction[1])])

   accuracy = len(np.argwhere(predictions == 'ML'))/len(predictions)
   print('Total accuracy :{}'.format(np.round(accuracy, 4)))