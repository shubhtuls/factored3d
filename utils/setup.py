# Usage:
'''
python setup.py build_ext --inplace
rm -rf build/
mv factored3d/utils/bbox_utils.so ./
rm -rf factored3d/
''' 
from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "Bbox utils",
    ext_modules = cythonize('bbox_utils.pyx'),  # accepts a glob pattern
)
