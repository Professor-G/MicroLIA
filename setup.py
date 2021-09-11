# -*- coding: utf-8 -*-
"""
Created on Fri Aug 3 13:30:11 2018

@author: danielgodinez
"""
from setuptools import find_packages, setup, Extension

setup(
	name = "LIA",
	version = 1.2,
	description = "Microlensing detection algorithm",
	author = "Daniel Godinez",
	author_email = "danielgodinez123@gmail.com",
	license = 'GPL-3.0',
	url = "https://github.com/dgodinez77/LIA",
	packages = find_packages(),
	include_package_data=True,
	install_requires = ['numpy','scikit-learn','astropy','math','scipy','PeakUtils', 'progress'],
	test_suite="nose.collector",
	package_data={
    '': ['Miras_vo.xml'],
    },
	)
