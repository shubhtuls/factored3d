# Installation Instructions

### Compilation
First, we need to compile some cython utilities.
```
cd utils
python setup.py build_ext --inplace
mv factored3d/utils/bbox_utils.so ./
rm -rf build/ # remove redundant folders
rm -rf factored3d/ # remove redundant folders
```

### Dependencies
