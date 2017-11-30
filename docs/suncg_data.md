# Instructions to download SUNCG

### SUNCG Dataset
Donwload the [SUNCG dataset](http://suncg.cs.princeton.edu/) and extract the contents to SUNCG_DIR. There should be 5 folders named 'house', 'room', 'object', 'layout' and 'object_vox' in SUNCG_DIR.

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

# Download depth images
wget http://pbrs.cs.princeton.edu/pbrs_release/data/depth_v2.zip
unzip depth_v2.zip -d ../renderings_depth
```
