# Instructions to download SUNCG

### SUNCG Dataset
Donwload the [SUNCG dataset](http://suncg.cs.princeton.edu/) and extract the contents to SUNCG_DIR. There should be 5 folders named 'house', 'room', 'object', 'texture' and 'object_vox' in SUNCG_DIR. We now download additional meta-data.
```
cd SUNCG_DIR;

# Download data splits
mkdir splits
cd splits
wget https://people.eecs.berkeley.edu/~shubhtuls/cachedir/factored3d/suncg_split.pkl
cd ..

# Download layout data (suncg houses with objects removed)
# we use this data to render the amodal depths
wget https://people.eecs.berkeley.edu/~shubhtuls/cachedir/factored3d/layout.tar.gz
tar -zxvf layout.tar.gz
mv houseLayout layout
```


### Physically-based Renderings
To use the [physically-based renderings](http://pbrs.cs.princeton.edu/) provided by Zhang et. al., we need to download the images, associated camera viewpoints and depth images (for training the baseline).

```
cd SUNCG_DIR;

mkdir zipfiles; cd zipfiles;

# Download camera viewpoints
wget http://pbrs.cs.princeton.edu/pbrs_release/data/camera_v2.zip
unzip camera_v2.zip -d ../camera

# Download LDR renderings
wget http://pbrs.cs.princeton.edu/pbrs_release/data/mlt_v2.zip
unzip mlt_v2.zip -d ../renderings_ldr

# meta-data
wget http://pbrs.cs.princeton.edu/pbrs_release/data/data_goodlist_v2.txt


# Download depth images (needed to train the depth baseline)
wget http://pbrs.cs.princeton.edu/pbrs_release/data/depth_v2.zip
unzip depth_v2.zip -d ../renderings_depth
```
