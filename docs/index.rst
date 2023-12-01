.. MicroLIA documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:15:14 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MicroLIA's documentation!
===============================
MicroLIA is an open-source program for microlensing detection in wide-field surveys. The original engine that was published used the machine learning Random Forest model, trained using a variety of lightcurve statistics. The current version of MicroLIA supports two additional models: eXtreme Gradient Boost and Neural Network. You can find information on the program development in the `paper <https://arxiv.org/abs/2004.14347>`_.

While the program is intended for microlensing signal detection, the API has been coded as a general-purpose tool and thus includes a function to generate a training set using any type of time-series data. To do this you can utilize the `load_all <https://microlia.readthedocs.io/en/latest/_modules/MicroLIA/training_set.html#load_all>`_ function available in the `training_set <https://microlia.readthedocs.io/en/latest/autoapi/MicroLIA/training_set/index.html>`_ module. This function requires that your training data contain at least 2-arrays (time + measurements), and while a third array is required (measurement errors), this can be zeros and will be ignored with the ``apply_weights`` parameter set to ``False``. For a complete list of statistical features the program calculates on your data, review the `features <https://microlia.readthedocs.io/en/latest/autoapi/MicroLIA/features/index.html>`_ module.

With a generated training set, you can easily train and assess a production-ready machine learning model using the `Classifier <https://microlia.readthedocs.io/en/latest/autoapi/MicroLIA/ensemble_model/index.html#MicroLIA.ensemble_model.Classifier>`_ class. For more information refer to the API documentation. 


Installation
==================
The current stable version can be installed via pip:

.. code-block:: bash

    pip install MicroLIA

You can also clone the development version:    

.. code-block:: bash

    git clone https://github.com/Professor-G/MicroLIA.git
    cd MicroLIA
    pip install .


Importing MicroLIA
==================
The most important variable to set when employing MicroLIA is the cadence of the survey -- we can construct this by appending timestamp arrays to an empty list. When a lightcurve is simulated, a timestamp from the list will be selected at random.

With our timestamps saved we can simulate our training data and generate an optimal machine learning model with the following imports:

.. code-block:: python

   from MicroLIA import training_set

   data_x, data_y = training_set.create(timestamps)

.. figure:: _static/simulation.jpg
    :align: center
|
Alternatively, if you wish to use your own lightcurves, use the load_all function which takes as input 
the path to a directory containing one subdirectory for each class in your training set. The class name
will be taken to be the subdirectory name.

.. code-block:: python

   data_x, data_y = training_set.load_all(path='/Users/daniel/Desktop/dummy')

With the lightcurves simulated or loaded, and the feature matrix saved as data_x, we can create our classifier object:

.. code-blocK:: python
      
      from MicroLIA import ensemble_model

      model = models.Classifier(data_x, data_y, impute=True, optimize=True, opt_cv=3, boruta_trials=25, n_iter=25)
      model.create()
      
.. figure:: _static/optimize.png
    :align: center
|
As the optimization routine has been enabled, the model creation process may take several hours depending on the size of the training set and the model being optimized. 

When the final model is output, we can predict new, unseen data, but note that if the input is in magnitude, an instrument zeropoint must be provided for proper flux conversion. If input is in flux, set convert=False:

.. code-block:: python

   prediction = model.predict(time, mag, magerr, convert=True, zp=24)

Example
==================
To review MicroLIA's functionality in detail please refer to the `example page <https://microlia.readthedocs.io/en/latest/source/Examples.html>`_, which outlines the options available when simulating the training data and creating/optimizing the classifiers. 

Science
==================
To learn about Gravitational Microlensing, including how to derive the magnification equation, please visit the `science page <https://microlia.readthedocs.io/en/latest/source/Gravitational%20Microlensing.html>`_. 

Pages
==================
.. toctree::
   :maxdepth: 1

   source/Gravitational Microlensing
   source/Examples

Documentation
==================
Here is the documentation for all the modules:

.. toctree::
   :maxdepth: 1

   source/MicroLIA