[![Documentation Status](https://readthedocs.org/projects/microlia/badge/?version=latest)](https://microlia.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2541465.svg)](https://doi.org/10.5281/zenodo.2541465)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/LGPL-3.0)
[![arXiv](https://img.shields.io/badge/arXiv-2004.14347-b31b1b.svg)](https://arxiv.org/abs/2004.14347)

# MicroLIA: MicroLensing Identification Algorithm
MicroLIA is an open-source program for detecting microlensing events in wide-field surveys. You can use the built-in modules to simulate lightcurves with adaptive cadence (the program only provides PSPL simulations), or you can utilize your own set of lightcurves.
<img src="https://user-images.githubusercontent.com/19847448/51231407-4cce2a80-1918-11e9-8c4b-aaafeddbd335.jpg" width="900" height="500">

# Installation

```
    $ pip install MicroLIA
```

# Version 2

As of version 2.7.0, MicroLIA provides the following new features and improvements:

* New time-series features (74 total). To enhance the analysis, we now take the derivative of the lightcurve and re-compute the features in derivative space, for a total of 148 statistical metrics.
* Lightcurve features can now be calculated by taking into account the flux/mag errors, thus allowing for proper weighting of data points.
* We include a feature selection procedure so as to identify the metrics that carry useful information given the training set, and remove features that contain no information for the classification task.
* Short-period variables can now be simulated using real RR-Lyrae templates.
* The training set can now be generated using your own directory of lightcurves, no limit on the amount of classes.
* After a training set is generated, a csv file is saved which can be directly input when creating the classifier; in addition, the training set module contains a plot function to visualize the simulated lightcurves.
* The machine learning engine hyperparameters can now be optimized using Bayesian analysis. 
* We added data imputation techniques to handle undefined values in the training data.
* Built-in class methods are now available to visualize the engine parameters and performance, as well as to save and load models post-processing.


# [Documentation](https://microlia.readthedocs.io/en/latest/?)

For technical details and an example of how to implement MicroLIA for a microlensing search, check out our [Documentation](https://microlia.readthedocs.io/en/latest/?).


# Additional Filtering: pyLIMA

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

# Citation

If you use MicroLIA in publication, we would appreciate citations to the paper, [Godines et al. 2019.](https://ui.adsabs.harvard.edu/abs/2019A%26C....2800298G/abstract)

 
# How to Contribute?

Want to contribute? Bug detections? Comments? Suggestions? Please email us : danielgodinez123@gmail.com, etibachelet@gmail.com, rstreet@lcogt.net
