# Instructions to train models and baselines

### Prerequisites
Make sure the data loading and preprocessing is complete. The training will also be visualized using visdom which can be started by
```
python -m visdom.server
```
Note that all the training jobs should be launched from one directory above CODE_ROOT. Additionally, the sample scripts below assume that the code folder is named 'factored3d'.


### Training factored 3D prediction models
We first train the object-centric 3D prediction module. Since training with proposals or predicting full voxels is computationally expensive, we train in stages to speed up the process.
```
# Download a pretrained object voxel auto-encoder
cd CODE_ROOT/cachedir/snapshots;
wget https://people.eecs.berkeley.edu/~shubhtuls/cachedir/factored3d/object_autoenc_32.tar.gz && tar -xvzf object_autoenc_32.tar.gz

# All jobs should be launched from one level above code directory
cd CODE_ROOT/..;

# First train the object-centric 3D prediction model on ground-truth boxes
python -m factored3d.experiments.suncg.box3d --plot_scalars --display_visuals --display_freq=2000 --save_epoch_freq=1 --batch_size=8  --name=box3d_base --use_context --pred_voxels=False --classify_rot --shape_loss_wt=10 --n_data_workers=0 --num_epochs=8

# Fine-tune the above model using proposals
python -m factored3d.experiments.suncg.dwr --name=dwr_base --classify_rot --pred_voxels=False --use_context --plot_scalars --display_visuals --save_epoch_freq=1 --display_freq=1000 --display_id=100 --box3d_ft --shape_loss_wt=10 --label_loss_wt=10  --batch_size=8 --num_epochs=1

# Finally, also learn to predict shape voxels instead of auto-encoder shape code
python -m factored3d.experiments.suncg.dwr --name=dwr_shape_ft --classify_rot --pred_voxels=True --shape_dec_ft --use_context --plot_scalars --display_visuals --save_epoch_freq=1 --display_freq=1000 --display_id=100 --shape_loss_wt=2  --label_loss_wt=10 --batch_size=8 --ft_pretrain_epoch=1 --num_epochs=1
```

We also train the layout (amodal inverse depth) prediction CNN
```
# job should be launched from one level above code directory
cd CODE_ROOT/..

python -m factored3d.experiments.suncg.layout --plot_scalars --display_visuals --save_epoch_freq=1 --batch_size=8 --name=layout_pred --display_freq=2000 --suncg_dl_out_layout=true --suncg_dl_out_depth=false --display_id=40 --num_epochs=8
```

### Training (inverse) depth prediction baseline
```
python -m factored3d.experiments.suncg.layout --plot_scalars --display_visuals --save_epoch_freq=1 --batch_size=8 --name=depth_baseline --display_freq=2000 --suncg_dl_out_layout=false --suncg_dl_out_depth=true --display_id=20 --num_epochs=8
```

### Training scene voxel prediction baseline
```
# job should be launched from one level above code directory
cd CODE_ROOT/..;

python -m factored3d.experiments.suncg.voxels --plot_scalars --display_visuals --save_epoch_freq=1 --batch_size=8 --name=voxels_baseline --display_freq=2000 --num_epochs=8
```
