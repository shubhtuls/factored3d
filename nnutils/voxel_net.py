'''
Scene level voxels prediction net.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import flags
import torch
import torch.nn as nn
from . import net_blocks as nb
import torchvision
#from oc3d.nnutils import roi_pooling
import pdb

#-------------- flags -------------#
#----------------------------------#
flags.DEFINE_integer('nz_voxels', 2000, 'Number of latent feat dimension for shape prediction')
flags.DEFINE_integer('n_voxels_upconv', 5, 'Number of upconvolution layers')

#------------- Modules ------------#
#----------------------------------#
class ResNetConv(nn.Module):
    def __init__(self, n_blocks=4):
        super(ResNetConv, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.n_blocks=n_blocks

    def forward(self, x):
        n_blocks = self.n_blocks
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        if n_blocks >= 1:
            x = self.resnet.layer1(x)
        if n_blocks >= 2:
            x = self.resnet.layer2(x)
        if n_blocks >= 3:
            x = self.resnet.layer3(x)
        if n_blocks >= 4:
            x = self.resnet.layer4(x)
        return x

#------------ Voxel Net -----------#
#----------------------------------#
class VoxelNet(nn.Module):
    def __init__(
        self, img_size,
        voxel_size, nz_voxels=2000,
        nz_init=256, n_voxels_upconv=5
    ):
        super(VoxelNet, self).__init__()

        self.resnet_conv = ResNetConv(n_blocks=4)
        nc_inp = 512*(img_size[0]//32)*(img_size[1]//32)

        self.encoder = nb.fc_stack(nc_inp, nz_voxels, 2)

        upsamp_factor = pow(2, n_voxels_upconv)
        self.spatial_size_init = [voxel_size[0]//upsamp_factor, voxel_size[1]//upsamp_factor, voxel_size[2]//upsamp_factor]
        nz_spatial = self.spatial_size_init[0]*self.spatial_size_init[1]*self.spatial_size_init[2]
        self.nz_init = nz_init

        self.decoder_reshape = nb.fc_stack(nz_voxels, nz_init*nz_spatial, 1)
        self.decoder = nb.decoder3d(n_voxels_upconv, None, nz_init, init_fc=False)

    def forward(self, imgs_inp):
        img_feat = self.resnet_conv.forward(imgs_inp)
        img_feat = img_feat.view(imgs_inp.size(0), -1)
        img_feat = self.encoder.forward(img_feat)
        img_feat = self.decoder_reshape.forward(img_feat)
        img_feat = img_feat.view(
            imgs_inp.size(0),
            self.nz_init,
            self.spatial_size_init[0],
            self.spatial_size_init[1],
            self.spatial_size_init[2]
        )
        voxels_pred = self.decoder.forward(img_feat)
        return voxels_pred
