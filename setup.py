# coding=utf-8

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow-gpu==2.12.0',
                     'fastprogress==0.1.21',
                     'pandas==0.25.0',
                     'transformers==2.1.1']

setup(
    name='bner',
    version='1.0.0',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='BERT for NER'
)