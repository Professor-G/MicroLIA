# -*- coding: utf-8 -*-
"""
Created on Fri Aug 3 13:30:11 2018

@author: danielgodinez
"""
from setuptools import setup, find_packages

setup(
    name="MicroLIA",
    version="2.7.2",
    author="Daniel Godinez",
    author_email="danielgodinez123@gmail.com",
    description="Machine learning classifier for microlensing event detection",
    long_description="A machine learning pipeline for detecting microlensing events using tree-based models and feature extraction from OGLE data.",
    license="GPL-3.0",
    url="https://github.com/Professor-G/MicroLIA",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn==1.1.1",
        "scikit-optimize==0.9.0",
        "astropy==5.0.4",
        "scipy==1.7.3",
        "peakutils==1.3.3",
        "progress==1.6",
        "matplotlib==3.5.1",
        "optuna==3.1.0",
        "boruta==0.3",
        "BorutaShap==1.0.16",
        "xgboost==1.6.1",
        "scikit-plot==0.3.7",
        "opencv-python==4.7.0.68",
        "pandas==1.4.1",
        "dill",
        "gatspy==0.3",
        "astroML==1.0.2.post1",
        "tensorflow",
    ],
    python_requires=">=3.9, <3.10",
    include_package_data=True,
    package_data={
        "MicroLIA": [
            "data/Miras_vo.xml",
            "data/Sesar2010/*",
            "test/test_model_xgb/MicroLIA_ensemble_model/*",
            "test/test_classifier.py",
            "test/test_features.py",
            "test/MicroLIA_Training_Set_OGLE_IV.csv",
            "test/test_ogle_lc.dat"
        ]
    },
    test_suite="nose.collector",
)
