from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import collections

import scipy.misc
import scipy.linalg
import scipy.io as sio
import scipy.ndimage.interpolation
from absl import flags

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from ..utils import suncg_parse

from ..renderer import utils as render_utils

#-------------- flags -------------#
#----------------------------------#
flags.DEFINE_string('suncg_dir', '/data0/shubhtuls/datasets/suncg_pbrs_release', 'Suncg Data Directory')
flags.DEFINE_boolean('filter_objects', True, 'Restrict object classes to main semantic classes.')
flags.DEFINE_integer('max_views_per_house', 0, '0->use all views. Else we randomly select upto the specified number.')

flags.DEFINE_boolean('suncg_dl_out_codes', True, 'Should the data loader load codes')
flags.DEFINE_boolean('suncg_dl_out_layout', False, 'Should the data loader load layout')
flags.DEFINE_boolean('suncg_dl_out_depth', False, 'Should the data loader load modal depth')
flags.DEFINE_boolean('suncg_dl_out_fine_img', True, 'We should output fine images')
flags.DEFINE_boolean('suncg_dl_out_voxels', False, 'We should output scene voxels')
flags.DEFINE_boolean('suncg_dl_out_proposals', False, 'We should edgebox proposals for training')
flags.DEFINE_boolean('suncg_dl_out_test_proposals', False, 'We should edgebox proposals for testing')
flags.DEFINE_integer('suncg_dl_max_proposals', 40, 'Max number of proposals per image')

flags.DEFINE_integer('img_height', 128, 'image height')
flags.DEFINE_integer('img_width', 256, 'image width')

flags.DEFINE_integer('img_height_fine', 480, 'image height')
flags.DEFINE_integer('img_width_fine', 640, 'image width')

flags.DEFINE_integer('layout_height', 64, 'amodal depth height : should be half image height')
flags.DEFINE_integer('layout_width', 128, 'amodal depth width : should be half image width')

flags.DEFINE_integer('voxels_height', 32, 'scene voxels height. Should be half of width and depth.')
flags.DEFINE_integer('voxels_width', 64, 'scene voxels width')
flags.DEFINE_integer('voxels_depth', 64, 'scene voxels depth')
flags.DEFINE_boolean('suncg_dl_debug_mode', False, 'Just running for debugging, should not preload ojects')

#------------- Dataset ------------#
#----------------------------------#
class SuncgDataset(Dataset):
    '''SUNCG data loader'''
    def __init__(self, house_names, opts):
        self._suncg_dir = opts.suncg_dir

        self._house_names = house_names
        self.img_size = (opts.img_height, opts.img_width)
        self.output_fine_img = opts.suncg_dl_out_fine_img
        if self.output_fine_img:
            self.img_size_fine = (opts.img_height_fine, opts.img_width_fine)
        self.output_codes = opts.suncg_dl_out_codes
        self.output_layout = opts.suncg_dl_out_layout
        self.output_modal_depth = opts.suncg_dl_out_depth
        self.output_voxels = opts.suncg_dl_out_voxels
        self.output_proposals = opts.suncg_dl_out_proposals
        self.output_test_proposals = opts.suncg_dl_out_test_proposals

        if self.output_layout or self.output_modal_depth:
            self.layout_size = (opts.layout_height, opts.layout_width)
        if self.output_voxels:
            self.voxels_size = (opts.voxels_width, opts.voxels_height, opts.voxels_depth)

        if self.output_proposals:
            self.max_proposals = opts.suncg_dl_max_proposals
        if self.output_codes:
            self.max_rois = opts.max_rois
            self._obj_loader = suncg_parse.ObjectLoader(osp.join(opts.suncg_dir, 'object'))
            if not opts.suncg_dl_debug_mode:
                self._obj_loader.preload()
            if opts.filter_objects:
                self._meta_loader = suncg_parse.MetaLoader(osp.join(opts.suncg_dir, 'ModelCategoryMappingEdited.csv'))
            else:
                self._meta_loader = None

        data_tuples = []
        for hx, house in enumerate(house_names):
            if (hx % 1000) == 0:
                print('Reading image names from house {}/{}'.format(hx, len(house_names)))
            imgs_dir = osp.join(opts.suncg_dir, 'renderings_ldr', house)
            view_ids = [f[0:6] for f in os.listdir(imgs_dir)]

            rng = np.random.RandomState([ord(c) for c in house])
            rng.shuffle(view_ids)

            if (opts.max_views_per_house > 0) and (opts.max_views_per_house < len(view_ids)):
                view_ids = view_ids[0:opts.max_views_per_house]
            for view_id in view_ids:
                data_tuples.append((house, view_id))
        self.n_imgs = len(data_tuples)
        self._data_tuples = data_tuples
        self._preload_cameras(house_names)

    def forward_img(self, index):
        house, view_id = self._data_tuples[index]
        img = scipy.misc.imread(osp.join(self._suncg_dir, 'renderings_ldr', house, view_id + '_mlt.png'))
        if self.output_fine_img:
            img_fine = scipy.misc.imresize(img, self.img_size_fine)
            img_fine = np.transpose(img_fine, (2,0,1))

        img = scipy.misc.imresize(img, self.img_size)
        img = np.transpose(img, (2,0,1))
        if self.output_fine_img:
            return img/255, img_fine/255, house, view_id
        else:
            return img/255, house, view_id

    def _preload_cameras(self, house_names):
        self._house_cameras = {}
        for hx, house in enumerate(house_names):
            if (hx % 200) == 0:
                print('Pre-loading cameras from house {}/{}'.format(hx, len(house_names)))
            cam_file = osp.join(self._suncg_dir, 'camera', house, 'room_camera.txt')
            camera_poses = suncg_parse.read_camera_pose(cam_file)
            self._house_cameras[house] = camera_poses

    def forward_codes(self, house_name, view_id):
        #print('Loading Codes for {}_{}'.format(house_name, view_id))
        campose = self._house_cameras[house_name][int(view_id)]
        cam2world = suncg_parse.campose_to_extrinsic(campose)
        world2cam = scipy.linalg.inv(cam2world)

        house_data = suncg_parse.load_json(
            osp.join(self._suncg_dir, 'house', house_name, 'house.json'))
        bbox_data = sio.loadmat(
            osp.join(self._suncg_dir, 'bboxes_node', house_name, view_id + '_bboxes.mat'))
        objects_data, objects_bboxes = suncg_parse.select_ids(
            house_data, bbox_data, meta_loader=self._meta_loader, min_pixels=500)
        objects_codes = suncg_parse.codify_room_data(
            objects_data, world2cam, self._obj_loader)
        objects_bboxes -= 1 #0 indexing to 1 indexing
        if len(objects_codes) > self.max_rois:
            select_inds = np.random.permutation(len(objects_codes))[0:self.max_rois]
            objects_bboxes = objects_bboxes[select_inds, :]
            objects_codes = [objects_codes[ix] for ix in select_inds]
        return objects_codes, objects_bboxes

    def forward_proposals(self, house_name, view_id, codes_gt, bboxes_gt):
        proposals_data = sio.loadmat(
            osp.join(self._suncg_dir, 'edgebox_proposals', house_name, view_id + '_proposals.mat'))
        bboxes_proposals = proposals_data['proposals'][:,0:4]
        bboxes_proposals -= 1 #zero indexed
        codes, bboxes, labels = suncg_parse.extract_proposal_codes(
            codes_gt, bboxes_gt, bboxes_proposals, self.max_proposals)
        return codes, bboxes, labels
    
    def forward_test_proposals(self, house_name, view_id):
        proposals_data = sio.loadmat(
            osp.join(self._suncg_dir, 'edgebox_proposals', house_name, view_id + '_proposals.mat'))
        bboxes_proposals = proposals_data['proposals'][:,0:4]
        bboxes_proposals -= 1 #zero indexed
        return bboxes_proposals

    def forward_layout(self, house_name, view_id, bg_depth=1e4):
        depth_im = scipy.misc.imread(osp.join(
            self._suncg_dir, 'renderings_layout', house_name, view_id + '_depth.png'))
        depth_im =  depth_im.astype(np.float)/1000.0  # depth was saved in mm
        depth_im += bg_depth*np.equal(depth_im,0).astype(np.float)
        disp_im = 1./depth_im
        amodal_depth = scipy.ndimage.interpolation.zoom(
            disp_im, (self.layout_size[0]/disp_im.shape[0], self.layout_size[1]/disp_im.shape[1]), order=0)
        amodal_depth = np.reshape(amodal_depth, (1, self.layout_size[0], self.layout_size[1]))
        return amodal_depth

    def forward_depth(self, house_name, view_id, bg_depth=1e4):
        depth_im = scipy.misc.imread(osp.join(
            self._suncg_dir, 'renderings_depth', house_name, view_id + '_depth.png'))
        depth_im =  depth_im.astype(np.float)/1000.0  # depth was saved in mm
        depth_im += bg_depth*np.equal(depth_im,0).astype(np.float)
        disp_im = 1./depth_im
        modal_depth = scipy.ndimage.interpolation.zoom(
            disp_im, (self.layout_size[0]/disp_im.shape[0], self.layout_size[1]/disp_im.shape[1]), order=0)
        modal_depth = np.reshape(modal_depth, (1, self.layout_size[0], self.layout_size[1]))
        return modal_depth

    def forward_voxels(self, house_name, view_id):
        scene_voxels = sio.loadmat(osp.join(
            self._suncg_dir, 'scene_voxels', house_name, view_id + '_voxels.mat'))
        scene_voxels = render_utils.downsample(
            scene_voxels['sceneVox'].astype(np.float32),
            64//self.voxels_size[1], use_max=True)
        return scene_voxels

    def __len__(self):
        return self.n_imgs

    def __getitem__(self, index):
        if self.output_fine_img:
            img, img_fine, house_name, view_id = self.forward_img(index)
        else:
            img, house_name, view_id = self.forward_img(index)

        elem = {
            'img': img,
            'house_name': house_name,
            'view_id': view_id,
        }
        if self.output_layout:
            layout = self.forward_layout(house_name, view_id)
            elem['layout'] = layout

        if self.output_voxels:
            voxels = self.forward_voxels(house_name, view_id)
            elem['voxels'] = voxels

        if self.output_modal_depth:
            depth = self.forward_depth(house_name, view_id)
            elem['depth'] = depth

        if self.output_codes:
            codes_gt, bboxes_gt = self.forward_codes(house_name, view_id)
            elem['codes'] = codes_gt
            elem['bboxes'] = bboxes_gt

        if self.output_proposals:
            codes_proposals, bboxes_proposals, labels_proposals = self.forward_proposals(
                house_name, view_id, codes_gt, bboxes_gt)
            if labels_proposals.size == 0:
                print('No proposal found: ', house_name, view_id)
            elem['codes_proposals'] = codes_proposals
            elem['bboxes_proposals'] = bboxes_proposals
            elem['labels_proposals'] = labels_proposals

        if self.output_test_proposals:
            bboxes_proposals = self.forward_test_proposals(house_name, view_id)
            if bboxes_proposals.size == 0:
                print('No proposal found: ', house_name, view_id)
            elem['bboxes_test_proposals'] = bboxes_proposals

        if self.output_fine_img:
            elem['img_fine'] = img_fine

        #print('House : {}, View : {}, Code Length : {}'.format(house_name, view_id, len(code)))
        return elem

#-------- Collate Function --------#
#----------------------------------#    
def recursive_convert_to_torch(elem):
    if torch.is_tensor(elem):
        return elem
    elif type(elem).__module__ == 'numpy':
        if elem.size == 0:
            return torch.zeros(elem.shape).type(torch.DoubleTensor)
        else:
            return torch.from_numpy(elem)
    elif isinstance(elem, int):
        return torch.LongTensor([elem])
    elif isinstance(elem, float):
        return torch.DoubleTensor([elem])
    elif isinstance(elem, collections.Mapping):
        return {key: recursive_convert_to_torch(d[key]) for key in elem}
    elif isinstance(elem, collections.Sequence):
        return [recursive_convert_to_torch(samples) for samples in elem]
    else:
        return elem

def collate_fn(batch):
    '''SUNCG data collater.
    
    Assumes each instance is a dict.
    Applies different collation rules for each field.

    Args:
        batch: List of loaded elements via Dataset.__getitem__
    '''
    collated_batch = {}
    # iterate over keys
    for key in batch[0]:
        if key =='codes' or key=='bboxes' or key=='codes_proposals' or key=='bboxes_proposals' or key=='bboxes_test_proposals':
            collated_batch[key] = [recursive_convert_to_torch(elem[key]) for elem in batch]
        elif key == 'labels_proposals':
            collated_batch[key] = torch.cat([default_collate(elem[key]) for elem in batch if elem[key].size > 0])
        else:
            collated_batch[key] = default_collate([elem[key] for elem in batch])
    return collated_batch

#----------- Data Loader ----------#
#----------------------------------#
def suncg_data_loader(house_names, opts):
    dset = SuncgDataset(house_names, opts)
    return DataLoader(
        dset, batch_size=opts.batch_size,
        shuffle=True, num_workers=opts.n_data_workers,
        collate_fn=collate_fn)


def suncg_data_loader_benchmark(house_names, opts):
    dset = SuncgDataset(house_names, opts)
    return DataLoader(
        dset, batch_size=opts.batch_size,
        shuffle=False, num_workers=opts.n_data_workers,
        collate_fn=collate_fn)