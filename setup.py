"""Install package."""
import re
import os
import sys
import subprocess
import traceback
from setuptools import setup, find_packages, Command
from setuptools.command.bdist_egg import bdist_egg as _bdist_egg
from setuptools.command.develop import develop as _develop
from distutils.command.build import build as _build


PYPI_REQUIREMENTS = []
if os.path.exists('requirements.txt'):
    for line in open('requirements.txt'):
        PYPI_REQUIREMENTS.append(line.strip())

scripts = ['bin/training_paccmann', 'bin/training_baseline']

setup(
    name='paccmann',
    version='0.1',
    description=(
        'paccmann - toolbox for prediction'
        ' of anticancer compound sensitivity using '
        'multimodal attention-based neural networks.'
    ),
    author='Ali Oskooei, Jannis Born, Matteo Manica',
    author_email='osk@zurich.ibm.com, jab@zurich.ibm.com, tte@zurich.ibm.com',
    long_description=open('README.md').read(),
    packages=find_packages('.'),
    install_requires=PYPI_REQUIREMENTS,
    zip_safe=False,
    scripts=scripts
)
