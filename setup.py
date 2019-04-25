# coding=utf-8

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow_hub>=0.3.0',
                     'bert-tensorflow>=1.0.1',
                     'pandas>=0.23.4']

setup(
    name='bner',
    version='1.0.0',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='BERT for NER'
)