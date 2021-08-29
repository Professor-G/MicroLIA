[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2541465.svg)](https://doi.org/10.5281/zenodo.2541465)
[![arXiv](https://img.shields.io/badge/arXiv-2004.14347-b31b1b.svg)](https://arxiv.org/abs/2004.14347)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/LGPL-3.0)

# LIA: Lens Identification Algorithm
<img src="https://user-images.githubusercontent.com/19847448/51231407-4cce2a80-1918-11e9-8c4b-aaafeddbd335.jpg" width="900" height="500">


# LIA

Please download the official Version 1.0 release from Zenodo, this GitHub rep is under development.

LIA is an open-source program for detecting microlensing events in wide-field surveys — it’s currently adapted for single lens detection only. The program first computes 47 statistical features from the lightcurve (mag+magerr), after which it applies a Principal Component Analysis (PCA) for feature dimensionality reduction. These PCA features are then run through a Random Forest ensemble to classify the lightcurve as either a microlensing event, a variable source, a cataclysmic variable (CV), or a constant source displaying no variability. We’ve adapted the code for use across any wide-field survey, and as such, a training set with adaptive cadence must first be created.

# Installation

Requires Python3.7 -- to install all dependencies run

```
python setup.py install
```

from the LIA directory.

# Creating Training Data & Constructing Models 

The **simulate** module contains the framework necessary for simulating all individual classes. For simulating a complete training set, we’ve simplified the process by including all necessary steps within the **create_training** module. The "hard" part is aggregating the necessary survey timestamps; these can be simulated, or be derived from real lightcurves if the survey is already underway. In this example we will assume a year-long space survey with daily cadence, hence only one timestamp from which to simulate our classes (please note that the set of timestamps must be appended to a list, as in practice survey cadence may be irregular across different tiles). We will also assume the survey has limiting magnitudes of 15 and 20, and as we don’t know the noise model of this imaginary survey, we will default to applying a Gaussian model — although the **noise_models** module contains a function for creating your own. Now, let’s simulate 500 of each class:

```
from LIA import training_set
import numpy as np 

time=[]
time.append(np.arange(0,366,1))

training_set.create(time, min_mag=15, max_mag=20, noise=None, n_class=500)
```

This function will output a FITS file titled ‘lightcurves’ that will contain the photometry for your simulated classes, sorted by ID number and class. It will also save two text files with labeled classes. The file titled ‘all_features’ contains the class label and the ID number corresponding to each light curve in the FITS file, followed by the 47 statistical values computed, while the other titled ‘pca_features’ contains only the class label followed by the principal components. We need these two text files to construct the required models.

```
from LIA import models

rf, pca = models.create_models(‘all_features.txt’, ‘pca_features.txt’)
```
With the RF model trained and the PCA transformation saved, we are ready to classify any light curve.

```
from LIA import microlensing_classifier

#create imaginary light curve
mag = np.array([18, 18.3, 18.1, 18, 18.4, 18.9, 19.2, 19.3, 19.5, 19.2, 18.8, 18.3, 18.6])
magerr = np.array([0.01, 0.01, 0.03, 0.09, 0.04, 0.1, 0.03, 0.13, 0.04, 0.06, 0.09, 0.1, 0.35])

prediction, ml_pred = microlensing_classifier.predict(mag, magerr, rf, pca)[0:2]
```
We’re interested only in the first two outputs which are the predicted class and the probability it’s microlensing, but by default the **predict** function will output the probability predictions for all classes. For more information please refer to the documentation available in the specific modules.

# pyLIMA

We find that in practice the algorithm flags << 1% of lightcurves as microlensing, with false-alerts being prominent when data quality is bad. This is difficult to circumnavigate as we can only train with what we expect the survey to detect, and as such simple anomalies in the photometry can yield unpredictable results. We strongly recommend fitting each microlensing candidate LIA detects with [pyLIMA](https://github.com/ebachelet/pyLIMA), an open-source program for modeling microlensing events. By restricting microlensing parameters to reasonable observables, this fitting algorithm acts as a great additional filter in the search for these rare transient events. We’ve had great success by restricting our PSPL parameters to the following:

* 0 < t<sub>E;</sub> < 1000
* 0 <= tE <= 1000
* uo < 2.0
* Reduced Chi2 <= 10

As pyLIMA provides more information than this, we suggest you explore a parameter space that best fits your needs. 

# Test Script

To make sure that the algorithm is working, please run the following test scripts located in the **test** folder:

* test_features
* test_classifier

If both test scripts work you are good to go!
 
# How to Contribute?

Want to contribute? Bug detections? Comments? Suggestions? Please email us : danielgodinez123@gmail.com, etibachelet@gmail.com, rstreet@lcogt.net
