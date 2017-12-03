# Instructions to evaluate models and baselines

### Pre-requisites
Install pcl and pcl-python. Instructions for Ubuntu:

```
# add pcl repo
sudo add-apt-repository ppa:v-launchpad-jochen-sprickerhof-de/pcl -y
sudo apt-get update -y

# install pcl
sudo apt-get install libpcl-all

# install python wrapper
cd CODE_ROOT/external
touch __init__.py
git clone git@github.com:s-gupta/python-pcl.git pythonpcl && cd pythonpcl && make
```

Note that the example scripts below are for the validation set. Please modify the arguments and plotting functions to use test set for the final evaluation.
### Comparing Scene Representations
```
# Launch comparison evaluation
# launch jobs from one level above code directory
python -m factored3d.benchmark.suncg.scene_comparison --num_train_epoch=1 --name=dwr_shape_ft --classify_rot --pred_voxels=True --use_context --eval_set=val

# Plot comparisons
cd CODE_ROOT/benchmark/suncg
python sc_plots.py
```

### Object Detection with Reconstruction Evaluation
```
# Launch detection setting object 3D prediction evaluation
# launch jobs from one level above code directory
python -m factored3d.benchmark.suncg.dwr --num_train_epoch=1 --name=dwr_shape_ft --classify_rot --pred_voxels=True --use_context   --eval_set=val

# Plot precision-recall curves
cd CODE_ROOT/benchmark/suncg
python pr_plots.py
```