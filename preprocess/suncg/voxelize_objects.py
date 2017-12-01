# This needs to be executed onscreen
import os,sys
import os.path as osp
import numpy as np
import scipy.io as sio

sys.path.append('/data0/shubhtuls/code/factored3d/external/binvox')
sun_cg_dir = '/data0/shubhtuls/datasets/suncg_pbrs_release'
binvox_exec_file = '/data0/shubhtuls/datasets/suncg_pbrs_release/toolbox/binvox'

import binvox_rw

def sub_dirs(d):
    return [o for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]


obj_ids = sub_dirs(osp.join(sun_cg_dir,'object'))
obj_ids = [o for o in obj_ids if 'copy' not in o]
grid_size = 64
dc1 = 'find {} -name "*.binvox" -type f -delete'.format(osp.join(sun_cg_dir,'object'))
dc2 = 'find {} -name "*.mat" -type f -delete'.format(osp.join(sun_cg_dir,'object'))
os.system(dc1) #delete old .binvox files
os.system(dc2) #delete old .mat files

for ix in range(len(obj_ids)):
    obj_id = obj_ids[ix]
    print(obj_id)
    object_dir = osp.join(sun_cg_dir, 'object', obj_id)
    binvox_file_interior = osp.join(object_dir, obj_id + '.binvox')
    binvox_file_surface = osp.join(object_dir, obj_id + '_1.binvox')

    cmd_interior = '{} -cb -d {} {}'.format(binvox_exec_file, grid_size, osp.join(object_dir, obj_id + '.obj'))
    cmd_surface = '{} -cb -e -d {} {}'.format(binvox_exec_file, grid_size, osp.join(object_dir, obj_id + '.obj'))
    os.system(cmd_interior)
    os.system(cmd_surface)

    with open(binvox_file_interior, 'rb') as f0:
        with open(binvox_file_surface, 'rb') as f1:
            vox_read_interior = binvox_rw.read_as_3d_array(f0)
            vox_read_surface = binvox_rw.read_as_3d_array(f1)

            #need to add translation corresponding to voxel centering
            shape_vox = vox_read_interior.data.astype(np.bool) + vox_read_surface.data.astype(np.bool)
            if(np.max(shape_vox) > 0):
                Xs, Ys, Zs = np.where(shape_vox)
                trans_centre = np.array([1.0*np.min(Xs)/(np.size(shape_vox,0)), 1.0*np.min(Ys)/(np.size(shape_vox,1)), 1.0*np.min(Zs)/(np.size(shape_vox,2)-1)] )
                translate = vox_read_surface.translate - trans_centre*vox_read_surface.scale
                sio.savemat(osp.join(object_dir, obj_id + '.mat'), {'voxels' : shape_vox, 'scale' : vox_read_surface.scale, 'translation' : translate})