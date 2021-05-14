from setuptools import setup
import os
import sys

if sys.version_info[0] < 3:
    with open('README.rst') as f:
        long_description = f.read()
else:
    with open('README.rst', encoding='utf-8') as f:
        long_description = f.read()


setup(
    name='EvolutionaryFS',
    version='0.0.2',
    description='Implementation of evolutionary algorithm for machine learning feature selection',
    long_description=long_description,
    long_description_content_type='text/markdown',  # This is important!
    author='StatguyUser',
    url='https://github.com/StatguyUser/EvolutionaryFS',
    install_requires=['numpy'],
    download_url='https://github.com/EvolutionaryFS/EvolutionaryFS.git',
    py_modules=["EvolutionaryFS"],
    package_dir={'':'src'},
)
