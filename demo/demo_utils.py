"""Testing class for the demo.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gflags as flags
import os
import os.path as osp
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
import scipy.misc
import pdb
import copy
import scipy.io as sio

from ..nnutils import test_utils
from ..nnutils import net_blocks
from ..nnutils import voxel_net
from ..nnutils import oc_net
from ..nnutils import disp_net

from ..utils import suncg_parse
from ..utils import metrics

from ..renderer import utils as render_utils


curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'cachedir')
flags.DEFINE_string('rendering_dir', osp.join(cache_path, 'rendering'), 'Directory where intermittent renderings are saved')

flags.DEFINE_integer('voxel_size', 32, 'Spatial dimension of shape voxels')
flags.DEFINE_integer('n_voxel_layers', 5, 'Number of layers ')
flags.DEFINE_integer('voxel_nc_max', 128, 'Max 3D channels')
flags.DEFINE_integer('voxel_nc_l1', 8, 'Initial shape encder/decoder layer dimension')
flags.DEFINE_float('voxel_eval_thresh', 0.25, 'Voxel evaluation threshold')

flags.DEFINE_string('shape_pretrain_name', 'object_autoenc_32', 'Experiment name for pretrained shape encoder-decoder')
flags.DEFINE_integer('shape_pretrain_epoch', 800, 'Experiment name for shape decoder')

flags.DEFINE_string('layout_name', 'layout_pred', 'Experiment name for layout predictor')
flags.DEFINE_integer('layout_train_epoch', 8, 'Experiment name for layout predictor')

flags.DEFINE_string('depth_name', 'depth_baseline', 'Experiment name for layout predictor')
flags.DEFINE_integer('depth_train_epoch', 8, 'Experiment name for layout predictor')

flags.DEFINE_string('scene_voxels_name', 'voxels_baseline', 'Experiment name for layout predictor')
flags.DEFINE_integer('scene_voxels_train_epoch', 8, 'Experiment name for layout predictor')
flags.DEFINE_float('scene_voxels_thresh', 0.25, 'Threshold for scene voxels prediction')

flags.DEFINE_integer('img_height', 128, 'image height')
flags.DEFINE_integer('img_width', 256, 'image width')

flags.DEFINE_integer('img_height_fine', 480, 'image height')
flags.DEFINE_integer('img_width_fine', 640, 'image width')

flags.DEFINE_integer('layout_height', 64, 'amodal depth height : should be half image height')
flags.DEFINE_integer('layout_width', 128, 'amodal depth width : should be half image width')

flags.DEFINE_integer('voxels_height', 32, 'scene voxels height. Should be half of width and depth.')
flags.DEFINE_integer('voxels_width', 64, 'scene voxels width')
flags.DEFINE_integer('voxels_depth', 64, 'scene voxels depth')

class DemoTester(test_utils.Tester):
    def load_oc3d_model(self):
        opts = self.opts
        self.voxel_encoder, nc_enc_voxel = net_blocks.encoder3d(
            opts.n_voxel_layers, nc_max=opts.voxel_nc_max, nc_l1=opts.voxel_nc_l1, nz_shape=opts.nz_shape)

        self.voxel_decoder = net_blocks.decoder3d(
            opts.n_voxel_layers, opts.nz_shape, nc_enc_voxel, nc_min=opts.voxel_nc_l1)

        self.oc3d_model = oc_net.OCNet(
            (opts.img_height, opts.img_width),
            roi_size=opts.roi_size,
            use_context=opts.use_context, nz_feat=opts.nz_feat,
            pred_voxels=False, nz_shape=opts.nz_shape, pred_labels=True,
            classify_rot=opts.classify_rot, nz_rot=opts.nz_rot)
        self.oc3d_model.add_label_predictor()

        if opts.pred_voxels:
            self.oc3d_model.code_predictor.shape_predictor.add_voxel_decoder(
                copy.deepcopy(self.voxel_decoder))

        self.load_network(self.oc3d_model, 'pred', self.opts.num_train_epoch)
        self.oc3d_model.eval()
        self.oc3d_model = self.oc3d_model.cuda(device_id=self.opts.gpu_id)

        if opts.pred_voxels:
             self.voxel_decoder = copy.deepcopy(self.oc3d_model.code_predictor.shape_predictor.decoder)

    def load_layout_model(self):
        opts = self.opts
        ## Load depth prediction network
        self.depth_model = disp_net.dispnet()
        network_dir = osp.join(opts.cache_dir, 'snapshots', opts.depth_name)
        self.load_network(
            self.depth_model, 'pred', opts.depth_train_epoch, network_dir=network_dir)
        self.depth_model.eval()
        self.depth_model = self.depth_model.cuda(device_id=self.opts.gpu_id)

    def load_depth_model(self):
        opts = self.opts
        self.layout_model = disp_net.dispnet()
        network_dir = osp.join(opts.cache_dir, 'snapshots', opts.layout_name)
        self.load_network(
            self.layout_model, 'pred', opts.layout_train_epoch, network_dir=network_dir)
        self.layout_model.eval()
        self.layout_model = self.layout_model.cuda(device_id=self.opts.gpu_id)

    def load_scene_voxels_model(self):
        opts = self.opts
        self.scene_voxels_model = voxel_net.VoxelNet(
            [opts.img_height, opts.img_width],
            [opts.voxels_width, opts.voxels_height, opts.voxels_depth],
            nz_voxels=opts.nz_voxels,
            n_voxels_upconv=opts.n_voxels_upconv
        )
        network_dir = osp.join(opts.cache_dir, 'snapshots', opts.scene_voxels_name)
        self.load_network(
            self.scene_voxels_model, 'pred', opts.layout_train_epoch, network_dir=network_dir)
        self.scene_voxels_model.eval()
        self.scene_voxels_model = self.scene_voxels_model.cuda(device_id=self.opts.gpu_id)

    def define_model(self):
        self.load_oc3d_model()
        self.load_layout_model()
        self.load_depth_model()
        self.load_scene_voxels_model()
        return

    def init_dataset(self):
        opts = self.opts
        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        if opts.voxel_size < 64:
            self.downsample_voxels = True
            self.downsampler = render_utils.Downsample(
                64//opts.voxel_size, use_max=True, batch_mode=True
            ).cuda(device_id=self.opts.gpu_id)
        else:
            self.downsampler = None

        if opts.classify_rot:
            self.quat_medoids = torch.from_numpy(
                scipy.io.loadmat(osp.join(opts.cache_dir, 'quat_medoids.mat'))['medoids']).type(torch.FloatTensor)

        if not opts.pred_voxels:
            network_dir = osp.join(opts.cache_dir, 'snapshots', opts.shape_pretrain_name)
            self.load_network(
                self.voxel_decoder,
                'decoder', opts.shape_pretrain_epoch, network_dir=network_dir)
            self.voxel_decoder.eval()
            self.voxel_decoder = self.voxel_decoder.cuda(device_id=self.opts.gpu_id)

    def decode_shape(self, pred_shape):
        opts = self.opts
        pred_shape = torch.nn.functional.sigmoid(
            self.voxel_decoder.forward(pred_shape)
        )
        return pred_shape

    def decode_rotation(self, pred_rot):
        opts = self.opts
        if opts.classify_rot:
            _, bin_inds = torch.max(pred_rot.data.cpu(), 1)
            pred_rot = Variable(suncg_parse.bininds_to_quats(
                bin_inds, self.quat_medoids), requires_grad=False)
        return pred_rot

    def set_input(self, batch):
        opts = self.opts
        rois = suncg_parse.bboxes_to_rois(batch['bboxes_test_proposals'])

        # Inputs for prediction
        input_imgs_fine = batch['img_fine'].type(torch.FloatTensor)
        input_imgs = batch['img'].type(torch.FloatTensor)

        self.input_imgs_orig = Variable(
            input_imgs.cuda(device=opts.gpu_id), requires_grad=False)

        for b in range(input_imgs_fine.size(0)):
            input_imgs_fine[b] = self.resnet_transform(input_imgs_fine[b])
            input_imgs[b] = self.resnet_transform(input_imgs[b])

        self.input_imgs = Variable(
            input_imgs.cuda(device=opts.gpu_id), requires_grad=False)

        self.input_imgs_fine = Variable(
            input_imgs_fine.cuda(device=opts.gpu_id), requires_grad=False)

        self.rois = Variable(
            rois.type(torch.FloatTensor).cuda(device=opts.gpu_id), requires_grad=False)

    def filter_pos(self, codes, pos_inds):
        pos_inds = torch.from_numpy(np.array(pos_inds)).squeeze()
        pos_inds = torch.autograd.Variable(
                pos_inds.type(torch.LongTensor).cuda(), requires_grad=False)
        filtered_codes = [torch.index_select(code, 0, pos_inds) for code in codes]
        return filtered_codes

    def predict_factored3d(self):
        codes_pred_all, labels_pred = self.oc3d_model.forward(
            (self.input_imgs_fine, self.input_imgs, self.rois))
        scores_pred = labels_pred.cpu().data.numpy()
        bboxes_pred = self.rois.data.cpu().numpy()[:, 1:]
        min_score_vis = np.minimum(0.7, np.max(scores_pred))
        pos_inds_vis = metrics.nms(
            np.concatenate((bboxes_pred, scores_pred), axis=1),
            0.3, min_score=min_score_vis)
        
        codes_pred_vis = self.filter_pos(codes_pred_all, pos_inds_vis)
        rois_pos_vis = self.filter_pos([self.rois], pos_inds_vis)[0]
        codes_pred_vis[0] = self.decode_shape(codes_pred_vis[0])
        codes_pred_vis[2] = self.decode_rotation(codes_pred_vis[2])

        layout_pred = self.layout_model.forward(self.input_imgs_orig)
        return codes_pred_vis, layout_pred
    
    def predict_depth(self):
        depth_pred = self.depth_model.forward(self.input_imgs_orig)
        return depth_pred

    def predict_scene_voxels(self):
        scene_voxels_pred = self.scene_voxels_model.forward(self.input_imgs_orig)
        return scene_voxels_pred


class DemoRenderer():
    def __init__(self, opts):
        self.opts = opts
        self.mesh_dir = osp.join(opts.rendering_dir, opts.name)
        if not os.path.exists(self.mesh_dir):
            os.makedirs(self.mesh_dir)

    def save_layout_mesh(self, mesh_dir, layout, prefix='layout'):
        opts = self.opts
        layout_vis = layout.data[0].cpu().numpy().transpose((1,2,0))
        vs, fs = render_utils.dispmap_to_mesh(
            layout_vis,
            suncg_parse.cam_intrinsic(),
            scale_x=self.opts.layout_width/640,
            scale_y=self.opts.layout_height/480
        )
        mesh_file = osp.join(self.mesh_dir, prefix + '.obj')
        fout = open(mesh_file, 'w')
        render_utils.append_obj(fout, vs, fs)
        fout.close()

    def save_codes_mesh(self, mesh_dir, code_vars, prefix='codes'):
        n_rois = code_vars[0].size()[0]
        code_list = suncg_parse.uncollate_codes(code_vars, 1, torch.Tensor(n_rois).fill_(0))
        mesh_file = osp.join(mesh_dir, prefix + '.obj')
        render_utils.save_parse(mesh_file, code_list[0], save_objectwise=False, thresh=0.1)

    def render_visuals(self, mesh_dir, obj_name=None):
        png_dir = osp.join(mesh_dir, 'rendering')
        if obj_name is not None:
            render_utils.render_mesh(osp.join(mesh_dir, obj_name + '.obj'), png_dir)
            im_view1 = scipy.misc.imread(osp.join(png_dir, '{}_render_000.png'.format(obj_name)))
            im_view2 = scipy.misc.imread(osp.join(png_dir, '{}_render_003.png'.format(obj_name)))
        else:
            render_utils.render_directory(mesh_dir, png_dir)
            im_view1 = scipy.misc.imread(osp.join(png_dir, 'render_000.png'))
            im_view2 = scipy.misc.imread(osp.join(png_dir, 'render_003.png'))
        return im_view1, im_view2

    def render_factored3d(self, codes, layout):
        os.system('rm {}/*.obj'.format(self.mesh_dir))
        self.save_codes_mesh(self.mesh_dir, codes)
        self.save_layout_mesh(self.mesh_dir, layout)
        return self.render_visuals(self.mesh_dir)
    
    def render_scene_vox(self, scene_vox):
        opts = self.opts
        os.system('rm {}/*.obj'.format(self.mesh_dir))
        voxels = scene_vox.data.cpu()[0,0].numpy()

        mesh_file = osp.join(self.mesh_dir, 'scene_vox.obj')
        vs, fs = render_utils.voxels_to_mesh(voxels.astype(np.float32), thresh=0.25)
        vs[:,0] -= voxels.shape[0]/2.0
        vs[:,1] -= voxels.shape[1]/2.0
        vs *= 0.04*(64//opts.voxels_height)
        fout = open(mesh_file, 'w')
        render_utils.append_obj(fout, vs, fs)
        fout.close()
        return self.render_visuals(self.mesh_dir, obj_name='scene_vox')

    def render_depth(self, dmap):
        opts = self.opts
        os.system('rm {}/*.obj'.format(self.mesh_dir))
        dmap_pred = dmap.data[0].cpu().numpy().transpose((1,2,0))
        mesh_file = osp.join(self.mesh_dir, 'depth.obj')
        dmap_points = render_utils.dispmap_to_points(
            dmap_pred,
            suncg_parse.cam_intrinsic(),
            scale_x=self.opts.layout_width/640,
            scale_y=self.opts.layout_height/480
        )

        vs, fs = render_utils.points_to_cubes(dmap_points)
        fout = open(mesh_file, 'w')
        render_utils.append_obj(fout, vs, fs)
        fout.close()

        return self.render_visuals(self.mesh_dir, obj_name='depth')
