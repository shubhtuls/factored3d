# Instructions to precompute data required for training

### Compiling SUNCG Toolbox

```
cd SUNCG_DIR;

# Download the toolbox
git clone https://github.com/shurans/SUNCGtoolbox ./toolbox
cd toolbox

# Use our modified rendering function
cp CODE_ROOT/preprocess/suncg/scn2img.cpp ./gaps/apps/scn2img/

# Compile
make

# (or optionally instead of above) compile with offscreen mesa support
make mesa
```
In case you compile with offscreen support, you might need to edit [this line](https://github.com/shurans/SUNCGtoolbox/blob/master/gaps/makefiles/Makefile.apps#L42) to specify additional lib directories if you're using a locally compiled version of mesa, and possibly also specify a CPLUS_INCLUDE_PATH. Though note that locally compiling mesa with offscreen support can get a bit tricky.

We highly recommend using the offscreen version, as otherwise the rendering behaviour is often stochastic.


### Rendering Layout and Node Images
You'll first need to edit the 'sunCgDir' variable in both the python scripts below. Note that both the rendering jobs can take a while. If you managed to compile the gaps toolbox with offscreen mesa, you can add --mesa=True to the commands below, else you'll need to run the rendering jobs in an onscreen mode.
```
cd CODE_ROOT/preprocess/suncg

# Render amodal depths (edit the 'sunCgDir' variable before running)
python render_layout_depth.py --min=1 --nc=1

# Render node indices (edit the 'sunCgDir' variable before running)
python render_node_indices.py --min=1 --nc=1

```

### Voxelize Objects and Scenes
Please download binvox from [here](http://www.patrickmin.com/binvox/) and store the binary as SUNCG_DIR/toolbox/binvox.

```
# Voxelize the objects (edit the 'sunCgDir' variable before running)
# This needs to be run in onscreen mode with a display/desktop connected
python voxelize_objects.py

# Compute voxelizations for the full scene (required for training the baseline)
# Edit the 'suncgDir' in globals.m before running
precompute_scene_voxels(1, 0);
```

### Compute object proposals
```
# Extract ground-truth object boxes
# Edit the 'suncgDir' in globals.m before running
precompute_gt_bboxes(1, 0);

# Extract edgebox proposals
# Edit the 'suncgDir' in globals.m before running
precompute_edge_boxes(1, 0);
```
