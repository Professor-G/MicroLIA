 [![DOI](https://zenodo.org/badge/78798347.svg)](https://zenodo.org/badge/latestdoi/78798347)
# A Gravitational Microlensing Detection Algorithm 
<img src="https://user-images.githubusercontent.com/19847448/37119532-ae69efbc-2225-11e8-81bf-a21ae6a21978.jpg" width="900" height="500">


# LIA

LIA is an open source program for detecting microlensing events in wide-field surveys. The program first computes 47 statistical features from the lightcurve (mag+magerr), after which it applies a Principal Component Analysis (PCA) for feature dimensionality reduction. These PCA features are then run through a Random Forest ensemble to classify the lightcurve as either a microlensing event, a variable source, a cataclysmic variable (CV), or a constant source displaying no variability. We’ve adapted the code for use across any wide-field survey, and as such, a training set with adaptive cadence must first be created. We provide a default training set that was simulated using ideal cadence, but for optimal performance it’s imperative that a training set matching survey conditions is used.

# Creating a Training Set

The **simulate** module contains the framework necessary for simulating all individual classes. For simulating a complete training set, we’ve simplified the process by including all necessary steps within the **create_training** module. The ‘hard’ part is aggregating the necessary timestamps you expect the survey to measure in. These can be simulated, or be derived from real lightcurves if the survey is already underway. In this example we will assume a year-long survey with daily cadence, hence only one timestamp for which to simulate our classes. We will also assume the survey has limiting magnitudes of 15 and 20, and as I don’t know the noise model of this imaginary survey, we will default to applying a Gaussian model. Now, let’s simulate 500 of each class:

```
from LIA import training_set

time = range(0,366,1)
training_set.create_training(time, min_base = 15, max_base=20, noise=None, q=500)

```

This function will output a FITS file titled ‘lightcurves’ that will contain the photometry for your simulated classes, sorted by ID number. It will also save two text files, one titled ‘all_features’ containing all 47 statistical values, and the other titled ‘pca_features’ containing only the principal components. We need the two text files to construct the required models.

```
from LIA import models

rf, pca = models.create_models(`all_features.txt’, `pca_features.txt’)
```
With the RF model trained and the PCA transformation saved, we are ready to classify any light curve.

```
from LIA import microlensing_classifier

#create imaginary light curve
mag = np.array([18, 18.3, 18.1, 18, 18.4, 18.9, 19.2, 19.3, 19.5, 19.2, 18.8, 18.3, 18.6])
magerr = np.array([0.01, 0.01, 0.03, 0.09, 0.04, 0.1, 0.03, 0.13, 0.04, 0.06, 0.09, 0.1, 0.35])

class, ml_pred = microlensing_classifier.predict_class(mag, magerr, rf, pca)[0:2]
```
We’re interested only in the predicted class and the probability it’s microlensing, but in principle you can output all predictions if you want.


# Test Script

To make sure that the algorithm is working, please run the following test scripts located in the **test** folder:

* test_script1
* test_script2

Both test scripts should classify the test lightcurve as microlensing. 
# How to Contribute?

Want to contribute? Bug detections? Comments? Please email us : dg7541@bard.edu, etibachelet@gmail.com, rstreet@lcogt.net
