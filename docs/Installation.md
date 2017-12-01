# Installation Instructions

### We recommend using virtualenv:
```
virtualenv venv
source venv/bin/activate
pip install -U pip
deactivate
source venv/bin/activate
pip install -r docs/requirements.txt
```

### Install pytorch
```
pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl
pip install torchvision visdom dominate
```

### Compilation
First, we need to compile some cython utilities.
```
cd utils
python setup.py build_ext --inplace
mv factored3d/utils/bbox_utils.so ./
rm -rf build/ # remove redundant folders
rm -rf factored3d/ # remove redundant folders
```

### Download pre-trained models.
```
# Download pre-trained Resnet18 Model.
wget https://download.pytorch.org/models/resnet18-5c106cde.pth -O ~/.torch/models/resnet18-5c106cde.pth

# Download our models.
wget https://people.eecs.berkeley.edu/~shubhtuls/cachedir/factored3d/cachedir.tar.gz && tar -xf cachedir.tar.gz
wget https://people.eecs.berkeley.edu/~shubhtuls/cachedir/factored3d/blender.tar.gz && tar -xf blender.tar.gz && mv blender renderer/.
```

### External Dependencies
```
mkdir external; cd external;
# Python interface for binvox
git clone https://github.com/dimatura/binvox-rw-py ./binvox

# Piotr's toolbox
git clone https://github.com/pdollar/toolbox ./toolbox

# Edgeboxes code
git clone https://github.com/pdollar/edges ./edges

# SSC-Net code (used for computing voxelization for the baseline)
git clone https://github.com/shurans/sscnet ./sscnet
```
