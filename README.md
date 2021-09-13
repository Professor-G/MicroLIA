[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2541465.svg)](https://doi.org/10.5281/zenodo.2541465)
[![arXiv](https://img.shields.io/badge/arXiv-2004.14347-b31b1b.svg)](https://arxiv.org/abs/2004.14347)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/LGPL-3.0)

# LIA: Lens Identification Algorithm
<img src="https://user-images.githubusercontent.com/19847448/51231407-4cce2a80-1918-11e9-8c4b-aaafeddbd335.jpg" width="900" height="500">

# LIA

Please download the official Version 1.0 release from Zenodo, this GitHub rep is under development.

LIA is an open-source program for detecting microlensing events in wide-field surveys — it’s currently adapted for single lens detection only. The program first computes 82 statistical features from the lightcurve (time, mag, magerr), 17 of which are computed in derivative space therefore we recommend sigma clipping the lightcurves to avoid exploding gradients. Using these features, you can create either a Random Forest of Neural Network classifier, with the option of applying a Principal Component Analysis (PCA) for feature dimensionality reduction. Once trained, LIA will classify new lightcurves as either a microlensing event, a variable source, a cataclysmic variable, a long period variable, or a constant source displaying no variability. We’ve adapted the code for use across any wide-field survey, and as such, a training set with adaptive cadence must first be created.

# Installation

Requires Python3.7 -- to install all dependencies run

```
    $ python setup.py install --old-and-unmanageable
```

from the LIA directory. After installing, you will be able to call LIA directly from any directory.

# Creating Training Data & Constructing Models 

The **simulate** module contains the framework necessary for simulating all individual classes. For simulating a complete training set, we’ve simplified the process by including all the necessary steps within the **create_training** module. The "hard" part is aggregating the necessary survey timestamps; these can be simulated, or be derived from real lightcurves if the survey is already underway. In this example we will assume a year-long survey with daily cadence, hence only one timestamp from which to simulate our classes (please note that the set of timestamps must be appended to a list, as in practice survey cadence may be irregular across different tiles). We will also assume that the survey has limiting magnitudes of 15 and 20, and as we don’t know the noise model of this imaginary survey, we will default to applying a Gaussian model — although the **noise_models** module contains a function for creating your own. Now, let’s simulate 500 of each class:

```python
from LIA import training_set
import numpy as np 

time=[]
time.append(np.arange(0,366,1))

training_set.create(time, min_mag=15, max_mag=20, noise=None, n_class=500)
```
<img src="https://user-images.githubusercontent.com/19847448/133037904-dced6505-af02-49bf-a6be-44c907716a21.png">

This function will output a FITS file titled ‘lightcurves’ that will contain the photometry for your simulated classes, sorted by ID number and class. It will also save two text files with labeled classes. The file titled ‘all_features’ contains the class label and the ID number corresponding to each lightcurve in the FITS file, followed by the statistical metrics that were computed, while the other titled ‘pca_features’ contains the class label, the ID, and the corresponding principal components. When a training set is created both the Random Forest and Neural Network classifiers will be tested, including with and without PCA -- this will allow you to determine what kind of model would perform best given your survey conditions. The output will be as follows:

<img src="https://user-images.githubusercontent.com/19847448/133038459-aa422912-9a01-4e05-af92-fd2abb418fb7.png">

We can see that the higest performance occurs when we use a Random Forest without PCA, or a Neural Network with PCA. To train a Neural Network with PCA we will run the following:

```python
from LIA import models

model, pca = models.create_models('all_features.txt', 'pca_features.txt', model='nn')
```

Then we can begin classifying any lightcurve:

```python
from LIA import microlensing_classifier

prediction, ml_pred = microlensing_classifier.predict(time, mag, magerr, model, pca)[0:2]
```

Or instead we could create a Random Forest model without PCA, we will do this by simply omitting the PCA option altogether:

```python
from LIA import models
from LIA import microlensing_classifier

model = models.create_models('all_features.txt', model='rf')
prediction, ml_pred = microlensing_classifier.predict(time, mag, magerr, model)[0:2]
```

We’re interested only in the first two outputs which are the predicted class and the probability it’s microlensing, but by default the **predict** function will output the probability predictions for all classes. For more information please refer to the documentation available in the specific modules.

# pyLIMA

We find that in practice the algorithm flags << 1% of lightcurves as microlensing, with false-alerts being prominent when data quality is bad. This is difficult to circumnavigate as we can only train with what we expect the survey to detect, and as such simple anomalies in the photometry can yield unpredictable results. We strongly recommend fitting each microlensing candidate LIA detects with [pyLIMA](https://github.com/ebachelet/pyLIMA), an open-source program for modeling microlensing events. By restricting microlensing parameters to reasonable observables, this fitting algorithm acts as a great additional filter in the search for these rare transient events. We’ve had great success by restricting our PSPL parameters to the following:

* 0 &le; t<sub>E</sub> &le; 1000
* u<sub>0</sub> &le; 2.0
* Reduced &chi;<sup>2</sup> &le; 10

As pyLIMA provides more information than this, we suggest you explore a parameter space that best fits your needs. 

# Test Script

To make sure that the algorithm is working, please run the following test scripts located in the **test** folder:

* test_features
* test_classifier

If both test scripts work you are good to go!
 
# How to Contribute?

Want to contribute? Bug detections? Comments? Suggestions? Please email us : danielgodinez123@gmail.com, etibachelet@gmail.com, rstreet@lcogt.net
