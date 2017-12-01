# Compiling SUNCG Toolbox

```
cd SUNCG_DIR;
mkdir zipfiles; cd zipfiles;

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