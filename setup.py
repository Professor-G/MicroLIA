# -*- coding: utf-8 -*-
"""
Created on Fri Aug 3 13:30:11 2018

@author: danielgodinez
"""
from setuptools import find_packages, setup

setup(
	name = "LIA",
	version = 1.1,
	description = "Microlensing detection algorithm",
	author = "Daniel Godinez",
	author_email = "danielgodinez123@gmail.com",
	license = 'GPL-3.0',
	url = "https://github.com/dgodinez77/LIA",
	install_requires = ['numpy','scikit-learn','astropy','mpmath','scipy','PeakUtils','tsfresh','AstroML'],
	packages = find_packages(),
	)