"""Script for layout prediction predictor experiment.
"""
# example usage (depth baseline) : python -m factored3d.experiments.suncg.layout --plot_scalars --display_visuals --save_epoch_freq=1 --batch_size=8 --name=depth_baseline --display_freq=2000 --suncg_dl_out_layout=false --suncg_dl_out_depth=true --display_id=20 --gpu_id=1

# example usage (layout prediction) : python -m factored3d.experiments.suncg.layout --plot_scalars --display_visuals --save_epoch_freq=1 --batch_size=8 --name=layout_pred --display_freq=2000 --suncg_dl_out_layout=true --suncg_dl_out_depth=false --display_id=40 --gpu_id=1

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from google.apputils import app
import gflags as flags
import os
import os.path as osp
import numpy as np
import torch
from torch.autograd import Variable
import time
import pdb

from ...data import suncg as suncg_data
from ...utils import suncg_parse
from ...nnutils import train_utils
from ...nnutils import disp_net
from ...utils import visutil
from ...renderer import utils as render_utils

FLAGS = flags.FLAGS

class LayoutTrainer(train_utils.Trainer):
    def define_model(self):
        self.model = disp_net.dispnet().cuda(device_id=self.opts.gpu_id)
        if self.opts.num_pretrain_epochs > 0:
            self.load_network(self.model, 'pred', self.opts.num_pretrain_epochs-1)
        return

    def init_dataset(self):
        opts = self.opts
        split_dir = osp.join(opts.suncg_dir, 'splits')
        self.split = suncg_parse.get_split(split_dir, house_names=os.listdir(osp.join(opts.suncg_dir, 'camera')))
        self.dataloader = suncg_data.suncg_data_loader(self.split['train'], opts)

    def define_criterion(self):
        self.criterion = torch.nn.L1Loss().cuda(device_id=self.opts.gpu_id)

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

            if opts.suncg_dl_out_layout:
                trg_tensor = batch['layout']
            else:
                assert(opts.suncg_dl_out_depth)
                trg_tensor = batch['depth']

            self.trg_layout = Variable(
                trg_tensor.type(torch.FloatTensor).cuda(device=self.opts.gpu_id), requires_grad=False)

    def forward(self):
        self.pred_layout = self.model.forward(self.input_imgs)
        self.total_loss = self.criterion.forward(self.pred_layout, self.trg_layout)

    def get_current_points(self):
        pts_dict = {}
        #for b in range(self.opts.batch_size):
        for b in range(1):
            dmap_gt = self.trg_layout.data[b].cpu().numpy().transpose((1,2,0))
            dmap_pred = self.pred_layout.data[b].cpu().numpy().transpose((1,2,0))
            keys = ['gt_layout_' + str(b), 'pred_layout_' + str(b)]
            dmaps = [dmap_gt, dmap_pred]
            min_disp = 1e-2
            for kx in range(2):
                dmap_points = render_utils.dispmap_to_points(
                    dmaps[kx],
                    suncg_parse.cam_intrinsic(),
                    scale_x=self.opts.layout_width/640,
                    scale_y=self.opts.layout_height/480,
                    min_disp = min_disp
                )
                pts_dict[keys[kx]] = dmap_points
                if kx == 0:
                    min_disp = 0.8/np.max(dmap_points[:, 2])

        return pts_dict

    def get_current_visuals(self):
        return {
            'img':visutil.tensor2im(self.input_imgs.data),
            'gt_layout':visutil.tensor2im(self.trg_layout.data),
            'pred_layout':visutil.tensor2im(self.pred_layout.data)
        }

    def get_current_scalars(self):
        return {'total_loss': self.smoothed_total_loss, 'total_loss_repeat': self.smoothed_total_loss}

def main(_):
    FLAGS.suncg_dl_out_codes = False
    FLAGS.suncg_dl_out_fine_img = False
    FLAGS.suncg_dl_out_voxels = False
    torch.manual_seed(0)
    trainer = LayoutTrainer(FLAGS)
    trainer.init_training()
    trainer.train()

if __name__ == '__main__':
    app.run()