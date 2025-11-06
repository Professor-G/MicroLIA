# -*- coding: utf-8 -*-
"""
Created on Fri Aug 3 13:30:11 2018

@author: danielgodinez
"""
from setuptools import setup, find_packages

setup(
    name="MicroLIA",
    version="2.8.1",
    author="Daniel Godinez",
    author_email="danielgodinez123@gmail.com",
    description="Machine learning classifier for microlensing event detection",
    long_description="A machine learning pipeline for detecting microlensing events using tree-based models.",
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
        "numpy==2.2.6",
        "pandas==2.3.1",
        "matplotlib==3.10.3",
        "scikit-learn==1.7.1",
        "xgboost==3.0.2",
        "optuna==4.4.0",
        "shap==0.48.0",
        "tqdm==4.67.1",
        "scipy==1.16.0",
        "astropy==7.1.0",
        "gatspy==0.3",
        "progress==1.6.1",
        "astroML==1.0.2",
        "joblib==1.5.1",
    ],
    python_requires=">=3.9",
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
)
