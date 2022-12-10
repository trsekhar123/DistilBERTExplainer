#!/usr/bin/env python
from setuptools import PEP420PackageFinder
from distutils.core import setup
import os
import os.path as op
ROOT = op.dirname(op.abspath(__file__))
SRC = op.join(ROOT,"src")

setup(name='explain_lib',
      version='1.0',
      description='Explainer Package',
      author='Rajasekhar Thiruthuvaraj',
      author_email='trsekhar.123@gmail.com',
      package_dir={'':'src'},
      packages=PEP420PackageFinder.find(where=str(SRC))
     )

