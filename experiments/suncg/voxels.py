"""Script for scene level voxels prediction experiment.
"""
# example usage : python -m factored3d.experiments.suncg.voxels --plot_scalars --display_visuals --save_epoch_freq=1 --batch_size=8 --name=voxels_baseline --display_freq=2000
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags
import os
import os.path as osp
import numpy as np
import scipy.misc
import torch
from torch.autograd import Variable
import time
import pdb

from ...data import suncg as suncg_data
from ...nnutils import train_utils
from ...nnutils import voxel_net
from ...utils import visutil
from ...utils import suncg_parse
from ...renderer import utils as render_utils

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', '..', 'cachedir')
flags.DEFINE_string('rendering_dir', osp.join(cache_path, 'rendering'), 'Directory where intermittent renderings are saved')

FLAGS = flags.FLAGS

class VoxelTrainer(train_utils.Trainer):
    def define_model(self):
        opts = self.opts
        self.model = voxel_net.VoxelNet(
            [opts.img_height, opts.img_width],
            [opts.voxels_width, opts.voxels_height, opts.voxels_depth],
            nz_voxels=opts.nz_voxels,
            n_voxels_upconv=opts.n_voxels_upconv
        )
        if self.opts.num_pretrain_epochs > 0:
            self.load_network(self.model, 'pred', self.opts.num_pretrain_epochs)
        self.model = self.model.cuda(device_id=self.opts.gpu_id)
        return

    def init_dataset(self):
        opts = self.opts
        split_dir = osp.join(opts.suncg_dir, 'splits')
        self.split = suncg_parse.get_split(split_dir, house_names=os.listdir(osp.join(opts.suncg_dir, 'camera')))
        self.dataloader = suncg_data.suncg_data_loader(self.split['train'], opts)

    def define_criterion(self):
        self.criterion = torch.nn.BCEWithLogitsLoss().cuda(device_id=self.opts.gpu_id)

    def set_input(self, batch):
        opts = self.opts
        img_tensor = batch['img'].type(torch.FloatTensor)

        # batch_size=1 messes with batch norm
        self.invalid_batch = (img_tensor.size(0) == 1)

        if self.invalid_batch:
            return
        else:
            self.input_imgs = Variable(
                img_tensor.cuda(device=self.opts.gpu_id), requires_grad=False)

            trg_tensor = batch['voxels'].unsqueeze(1)
            self.trg_voxels = Variable(
                trg_tensor.type(torch.FloatTensor).cuda(device=self.opts.gpu_id), requires_grad=False)

    def forward(self):
        self.pred_voxels = self.model.forward(self.input_imgs)
        self.total_loss = self.criterion.forward(self.pred_voxels, self.trg_voxels)

    def render_voxels(self, voxels, prefix='mesh'):
        opts = self.opts
        voxels = voxels.data.cpu()[0,0].numpy()

        mesh_dir = osp.join(opts.rendering_dir, opts.name)
        if not os.path.exists(mesh_dir):
            os.makedirs(mesh_dir)

        mesh_file = osp.join(mesh_dir, prefix + '.obj')
        vs, fs = render_utils.voxels_to_mesh(voxels.astype(np.float32))
        vs[:,0] -= voxels.shape[0]/2.0
        vs[:,1] -= voxels.shape[1]/2.0
        vs *= 0.04*(64//opts.voxels_height)
        fout = open(mesh_file, 'w')
        render_utils.append_obj(fout, vs, fs)
        fout.close()

        png_dir = mesh_file.replace('.obj', '/')
        render_utils.render_mesh(mesh_file, png_dir)

        return scipy.misc.imread(osp.join(png_dir, prefix + '_render_000.png'))

    def get_current_visuals(self):
        visuals = {
            'img':visutil.tensor2im(self.input_imgs.data)
        }
        visuals['voxels_gt'] = self.render_voxels(self.trg_voxels, prefix='gt')
        visuals['voxels_pred'] = self.render_voxels(
            torch.nn.functional.sigmoid(self.pred_voxels), prefix='pred')
        return visuals
    
    def get_current_points(self):
        return {}

    def get_current_scalars(self):
        return {'total_loss': self.smoothed_total_loss, 'total_loss_repeat': self.smoothed_total_loss}

def main(_):
    FLAGS.suncg_dl_out_codes = False
    FLAGS.suncg_dl_out_fine_img = False
    FLAGS.suncg_dl_out_voxels = True
    FLAGS.suncg_dl_out_layout = False
    FLAGS.suncg_dl_out_depth = False
    torch.manual_seed(0)
    trainer = VoxelTrainer(FLAGS)
    trainer.init_training()
    trainer.train()

if __name__ == '__main__':
    app.run()