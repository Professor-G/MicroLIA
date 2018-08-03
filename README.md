 [![DOI](https://zenodo.org/badge/78798347.svg)](https://zenodo.org/badge/latestdoi/78798347)
# A Gravitational Microlensing Detection Algorithm 
<img src="https://user-images.githubusercontent.com/19847448/37119532-ae69efbc-2225-11e8-81bf-a21ae6a21978.jpg" width="900" height="500">


# LIA

LIA is an open source program for detecting microlensing events in wide-field surveys. The program first computes 47 statistical features from the lightcurve (mag+magerr), after which it applies a Principal Component Analysis (PCA) for feature dimensionality reduction. These PCA features are then run through a Random Forest ensemble to classify the lightcurve as either a microlensing event, a variable source, a cataclysmic variable (CV), or a constant source displaying no variability. We’ve adapted the code for use across any wide-field survey, and as such, a training set with adaptive cadence must first be created. We provide a default training set that was simulated using ideal cadence, but for optimal performance it’s imperative that a training set matching survey conditions is used.

# Creating a Training Set

The **simulate** module contains the framework necessary for simulating all individual classes. For simulating a complete training set, we’ve simplified the process by including all necessary steps within the **create_training** module. The “hard” part is aggregating the necessary timestamps you expect the survey to measure. These can be simulated, or be derived from real lightcurves if the survey is already underway. In this example we will assume a year-long survey with daily cadence, hence only one timestamp for which to simulate our classes; in addition to limiting magnitudes of 15 and 20. As I don’t know the noise model of this imaginary survey, we will default to applying a Gaussian model. Now, let’s simulate 500 of each class:

```
from LIA import create_training
time = np.arange(1,365,1)

create_training.create_training(time, min_base = 15, max_base=20, fn=None, q=500)

```

This function will output a FITS file title ‘lightcurves’ that will contain the photometry for your simulated classes, sorted by ID number. 







# Algorithm Performance
<img src="https://user-images.githubusercontent.com/19847448/37122785-bf46bf18-222f-11e8-8266-fa0bb1c48dbd.jpg" width="900" height="500">

# pyLIMA
The algorithm implements pyLIMA as an additional filter when a microlensing candidate is detected. This is an open source for modeling microlensing events, and must be installed for the algorithm to work (please see: https://github.com/ebachelet/pyLIMA). By restricitng fitted parameters, we are able to eliminate true alerts from misclassified transients and variables. Microlensing detections that don't fall within our parameter space are then classified as OTHER.

# Microlensing Detection
The main module that classifies the lightcurve is the **random_forest_classifier** using the **predict_class** function. It makes use of the module **stats_computation**, which computes the statistical metrics that were used to train the classifier. Statistics include robust and devised metrics, as well as variability indices found in the literature.  Information regarding individual metrics is available in the **stats_computation** module. Note that some of the statistics are not utilized, as they were low-performing and hence omitted to improve the efficiency of the program. 

When a lightcurve is flagged by the RF as a potential microlensing candidate, a PSPL fit is attempted using the Levenberg-Marquardt algorithm. Microlensing parameters (t<sub>o</sub>, u<sub>o</sub>, t<sub>E</sub>) as well as the reduced chi-squared must fall within a reasonable parameter space otherwise the algorithm forwards the event as a false-alert. If the LM fit fails, another attempt is made using the differential evolution algorithm -- in this case the reduced chi-squared is fixed to the maximum accepted value of 3.0, as outliers and poor data are to be expected. For documentation, see: https://ebachelet.github.io/pyLIMA/pyLIMA.microlfits.html.


# Installation + Libraries
Clone the repository or download to a local system as a ZIP file. It's best to work off the same directory in which the package (as well as pyLIMA) is saved, so that the modules can be called directly, such as: 

from **random_forest_classifier** import **predict_class**

Besides pyLIMA, the code is written entirely in python, and makes use of familiar packages including sklearn and scipy.


# Test Script

To make sure that the algorithm is working, please run the following test scripts located in the **test** folder:

* test_script1
* test_script2

Both test scripts should classify the test lightcurve as microlensing. 
# How to Contribute?

Want to contribute? Bug detections? Comments? Please email us : dg7541@bard.edu, etibachelet@gmail.com, rstreet@lcogt.net
