.. LIA documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:15:14 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MicroLIA's documentation!
===============================
MicroLIA is an open-source program for microlensing detection in wide-field surveys. The original engine that was published used the machine learning Random Forest model, trained using a variety of lightcurve statistics. The current version of MicroLIA supports two additional models: Extreme Gradient Boost and Neural Network. 

Below is the outline of the program we have designed:

Installation
==================
The current stable version can be installed via pip:

.. code-block:: bash

    pip install MicroLIA


Importing MicroLIA
==================
The most important variable to set when employing MicroLIA is the cadence of the survey -- we can construct this by appending timestamp arrays to an empty list. When a lightcurve is simulated, a timestamp from the list will be selected at random.

With our timestamps saved we can simulate our training data and generate an optimal machine learning model with the following import:

.. code-block:: python

   from MicroLIA import training_set, models

   data_x, data_y = training_set.create(timestamps)
   model, imputer, feats_to_use = models.create(data_x, data_y)

The above procedure may take ~1 hour given the default parameters. We can now make predictions of new, unseen data:

.. code-block:: python

   prediction = models.predict(time, mag, magerr, model=model, imputer=imputer, feats_to_use=feats_to_use)

Example
==================
To learn about MicroLIA's functionality please refer to the `example page <https://microlia.readthedocs.io/en/latest/source/Examples.html>`_, which covers in detail the options we have when simulating our training data and creating the classifiers. 

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
