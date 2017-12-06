"""Script for box3d prediction experiment.
"""
# Sample usage: python -m factored3d.experiments.suncg.box3d --plot_scalars --display_visuals --display_freq=2000 --save_epoch_freq=1 --batch_size=8  --name=box3d_base --use_context --pred_voxels=False --classify_rot --shape_loss_wt=10

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags
import os
import os.path as osp
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
import time
import scipy.misc
import pdb
import copy

from ...data import suncg as suncg_data
from ...utils import suncg_parse
from ...nnutils import train_utils
from ...nnutils import net_blocks
from ...nnutils import loss_utils
from ...nnutils import oc_net
from ...nnutils import disp_net
from ...utils import visutil
from ...renderer import utils as render_utils

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', '..', 'cachedir')
flags.DEFINE_string('rendering_dir', osp.join(cache_path, 'rendering'), 'Directory where intermittent renderings are saved')

flags.DEFINE_integer('voxel_size', 32, 'Spatial dimension of shape voxels')
flags.DEFINE_integer('n_voxel_layers', 5, 'Number of layers ')
flags.DEFINE_integer('voxel_nc_max', 128, 'Max 3D channels')
flags.DEFINE_integer('voxel_nc_l1', 8, 'Initial shape encder/decoder layer dimension')

flags.DEFINE_string('shape_pretrain_name', 'object_autoenc_32', 'Experiment name for pretrained shape encoder-decoder')
flags.DEFINE_integer('shape_pretrain_epoch', 800, 'Experiment name for shape decoder')
flags.DEFINE_boolean('shape_dec_ft', False, 'If predicting voxels, should we pretrain from an existing deocder')

flags.DEFINE_string('ft_pretrain_name', 'box3d_base', 'Experiment name from which we will pretrain the OCNet')
flags.DEFINE_integer('ft_pretrain_epoch', 0, 'Network epoch from which we will finetune')

flags.DEFINE_integer('max_rois', 5, 'If we have more objects than this per image, we will subsample.')
flags.DEFINE_integer('max_total_rois', 40, 'If we have more objects than this per batch, we will reject the batch.')

FLAGS = flags.FLAGS


class Box3dTrainer(train_utils.Trainer):
    def define_model(self):
        '''
        Define the pytorch net 'model' whose weights will be updated during training.
        '''
        opts = self.opts
        assert(not (opts.ft_pretrain_epoch > 0 and opts.num_pretrain_epochs > 0))

        self.voxel_encoder, nc_enc_voxel = net_blocks.encoder3d(
            opts.n_voxel_layers, nc_max=opts.voxel_nc_max, nc_l1=opts.voxel_nc_l1, nz_shape=opts.nz_shape)

        self.voxel_decoder = net_blocks.decoder3d(
            opts.n_voxel_layers, opts.nz_shape, nc_enc_voxel, nc_min=opts.voxel_nc_l1)

        self.model = oc_net.OCNet(
            (opts.img_height, opts.img_width),
            roi_size=opts.roi_size,
            use_context=opts.use_context, nz_feat=opts.nz_feat,
            pred_voxels=opts.pred_voxels, nz_shape=opts.nz_shape,
            classify_rot=opts.classify_rot, nz_rot=opts.nz_rot)

        if opts.ft_pretrain_epoch > 0:
            network_dir = osp.join(opts.cache_dir, 'snapshots', opts.ft_pretrain_name)
            self.load_network(
                self.model, 'pred', opts.ft_pretrain_epoch, network_dir=network_dir)

        if opts.pred_voxels:
            self.model.code_predictor.shape_predictor.add_voxel_decoder(
                copy.deepcopy(self.voxel_decoder))

        if opts.pred_voxels and opts.shape_dec_ft:
            network_dir = osp.join(opts.cache_dir, 'snapshots', opts.shape_pretrain_name)
            self.load_network(
                self.model.code_predictor.shape_predictor.decoder,
                'decoder', opts.shape_pretrain_epoch, network_dir=network_dir)

        if self.opts.num_pretrain_epochs > 0:
            self.load_network(self.model, 'pred', self.opts.num_pretrain_epochs-1)
        self.model = self.model.cuda(device_id=self.opts.gpu_id)
        return

    def init_dataset(self):
        opts = self.opts
        self.real_iter = 1 # number of iterations we actually updated the net for
        self.data_iter = 1 # number of iterations we called the data loader
        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        split_dir = osp.join(opts.suncg_dir, 'splits')
        self.split = suncg_parse.get_split(split_dir, house_names=os.listdir(osp.join(opts.suncg_dir, 'camera')))
        self.dataloader = suncg_data.suncg_data_loader(self.split['train'], opts)

        if not opts.pred_voxels:
            network_dir = osp.join(opts.cache_dir, 'snapshots', opts.shape_pretrain_name)
            self.load_network(
                self.voxel_encoder,
                'encoder', opts.shape_pretrain_epoch, network_dir=network_dir)
            self.load_network(
                self.voxel_decoder,
                'decoder', opts.shape_pretrain_epoch, network_dir=network_dir)
            self.voxel_encoder.eval()
            self.voxel_encoder = self.voxel_encoder.cuda(device_id=self.opts.gpu_id)
            self.voxel_decoder.eval()
            self.voxel_decoder = self.voxel_decoder.cuda(device_id=self.opts.gpu_id)

        if opts.voxel_size < 64:
            self.downsample_voxels = True
            self.downsampler = render_utils.Downsample(
                64//opts.voxel_size, use_max=True, batch_mode=True
            ).cuda(device_id=self.opts.gpu_id)

        if opts.classify_rot:
            self.quat_medoids = torch.from_numpy(
                scipy.io.loadmat(osp.join(opts.cache_dir, 'quat_medoids.mat'))['medoids']).type(torch.FloatTensor)


    def define_criterion(self):
        self.smoothed_factor_losses = {
            'shape': 0, 'scale': 0, 'quat': 0, 'trans': 0
        }

    def set_input(self, batch):
        opts = self.opts
        rois = suncg_parse.bboxes_to_rois(batch['bboxes'])
        self.data_iter += 1
        if rois.numel() <= 5 or rois.numel() >= 5*opts.max_total_rois: #with just one element, batch_norm will screw up
            self.invalid_batch = True
            return
        else:
            self.invalid_batch = False
            self.real_iter += 1

        input_imgs_fine = batch['img_fine'].type(torch.FloatTensor)
        input_imgs = batch['img'].type(torch.FloatTensor)
        for b in range(input_imgs_fine.size(0)):
            input_imgs_fine[b] = self.resnet_transform(input_imgs_fine[b])
            input_imgs[b] = self.resnet_transform(input_imgs[b])

        self.input_imgs = Variable(
            input_imgs.cuda(device=opts.gpu_id), requires_grad=False)

        self.input_imgs_fine = Variable(
            input_imgs_fine.cuda(device=opts.gpu_id), requires_grad=False)

        self.rois = Variable(
            rois.type(torch.FloatTensor).cuda(device=opts.gpu_id), requires_grad=False)

        code_tensors = suncg_parse.collate_codes(batch['codes'])
        code_tensors[0] = code_tensors[0].unsqueeze(1) #unsqueeze voxels

        if opts.classify_rot:
            quats_gt = code_tensors[2].clone()
            code_tensors[2] = suncg_parse.quats_to_bininds(code_tensors[2], self.quat_medoids)
            quats_binned = suncg_parse.bininds_to_quats(code_tensors[2], self.quat_medoids)
            # q_diff_loss = (quats_gt-quats_binned).pow(2).sum(1)
            # q_sum_loss = (quats_gt+quats_binned).pow(2).sum(1)
            # q_loss, _ = torch.stack((q_diff_loss, q_sum_loss), dim=1).min(1)
            # print(quats_gt, quats_binned)
            # print(q_loss)


        self.codes_gt = [
            Variable(t.cuda(device=opts.gpu_id), requires_grad=False) for t in code_tensors]

        if self.downsample_voxels:
            self.codes_gt[0] = self.downsampler.forward(self.codes_gt[0])

        if not opts.pred_voxels:
            self.codes_gt[0] = self.voxel_encoder.forward(self.codes_gt[0])

    def get_current_scalars(self):
        loss_dict = {'total_loss': self.smoothed_total_loss, 'iter_frac': self.real_iter/self.data_iter}
        for k in self.smoothed_factor_losses.keys():
            loss_dict['loss_' + k] = self.smoothed_factor_losses[k]
        return loss_dict

    def render_codes(self, code_vars, prefix='mesh'):
        opts = self.opts
        code_list = suncg_parse.uncollate_codes(code_vars, self.input_imgs.data.size(0), self.rois.data.cpu()[:,0])

        mesh_dir = osp.join(opts.rendering_dir, opts.name)
        if not os.path.exists(mesh_dir):
            os.makedirs(mesh_dir)
        mesh_file = osp.join(mesh_dir, prefix + '.obj')
        render_utils.save_parse(mesh_file, code_list[0], save_objectwise=False)

        png_dir = mesh_file.replace('.obj', '/')
        render_utils.render_mesh(mesh_file, png_dir)

        return scipy.misc.imread(osp.join(png_dir, prefix + '_render_000.png'))


    def get_current_visuals(self):
        visuals = {}
        opts = self.opts
        visuals['img'] = visutil.tensor2im(visutil.undo_resnet_preprocess(
            self.input_imgs_fine.data))

        codes_gt_vis = [t for t in self.codes_gt]
        if not opts.pred_voxels:
            codes_gt_vis[0] = torch.nn.functional.sigmoid(
                self.voxel_decoder.forward(self.codes_gt[0])
            )

        if opts.classify_rot:
            codes_gt_vis[2] = Variable(suncg_parse.bininds_to_quats(
                codes_gt_vis[2].cpu().data, self.quat_medoids), requires_grad=False)

        visuals['codes_gt'] = self.render_codes(codes_gt_vis, prefix='gt')

        codes_pred_vis = [t for t in self.codes_pred]
        if not opts.pred_voxels:
            codes_pred_vis[0] = torch.nn.functional.sigmoid(
                self.voxel_decoder.forward(self.codes_pred[0])
            )

        if opts.classify_rot:
            _, bin_inds = torch.max(codes_pred_vis[2].data.cpu(), 1)
            codes_pred_vis[2] = Variable(suncg_parse.bininds_to_quats(
                bin_inds, self.quat_medoids), requires_grad=False)

        visuals['codes_pred'] = self.render_codes(codes_pred_vis, prefix='pred')

        return visuals


    def get_current_points(self):
        pts_dict = {}
        return pts_dict

    def forward(self):
        opts = self.opts

        self.codes_pred = self.model.forward((self.input_imgs_fine, self.input_imgs, self.rois))
        self.total_loss, self.loss_factors = loss_utils.code_loss(
            self.codes_pred, self.codes_gt,
            pred_voxels=opts.pred_voxels,
            classify_rot=opts.classify_rot,
            shape_wt=opts.shape_loss_wt,
            scale_wt=opts.scale_loss_wt,
            quat_wt=opts.quat_loss_wt,
            trans_wt=opts.trans_loss_wt
        )
        for k in self.smoothed_factor_losses.keys():
            self.smoothed_factor_losses[k] = 0.99*self.smoothed_factor_losses[k] + 0.01*self.loss_factors[k].data[0]


def main(_):
    torch.manual_seed(0)
    if FLAGS.classify_rot:
        FLAGS.nz_rot = 24
    else:
        FLAGS.nz_rot = 4
    FLAGS.n_data_workers = 0 # code crashes otherwise due to json not liking parallelization
    trainer = Box3dTrainer(FLAGS)
    trainer.init_training()
    trainer.train()


if __name__ == '__main__':
    app.run()