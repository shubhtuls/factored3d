'''
Object-centric prediction net.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gflags as flags
import torch
import torch.nn as nn
import torchvision
from . import net_blocks as nb
from ..external.roi_pooling.modules import roi_pool_py as roi_pool
#from oc3d.nnutils import roi_pooling
import pdb

#-------------- flags -------------#
#----------------------------------#
flags.DEFINE_integer('roi_size', 4, 'RoI feat spatial size.')
flags.DEFINE_integer('nz_shape', 20, 'Number of latent feat dimension for shape prediction')
flags.DEFINE_integer('nz_feat', 300, 'RoI encoded feature size')
flags.DEFINE_boolean('use_context', True, 'Should we use bbox + full image features')
flags.DEFINE_boolean('pred_voxels', True, 'Predict voxels, or code instead')
flags.DEFINE_boolean('classify_rot', False, 'Classify rotation, or regress quaternion instead')
flags.DEFINE_integer('nz_rot', 4, 'Number of outputs for rot prediction. Value overriden in code.')


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


class ShapePredictor(nn.Module):
    def __init__(self, nz_feat, nz_shape, pred_voxels=True):
        super(ShapePredictor, self).__init__()
        self.pred_layer = nb.fc(True, nz_feat, nz_shape)
        self.pred_voxels = pred_voxels

    def forward(self, feat):
        # pdb.set_trace()
        shape = self.pred_layer.forward(feat)
        # print('shape: ( Mean = {}, Var = {} )'.format(shape.mean().data[0], shape.var().data[0]))
        if self.pred_voxels:
            shape = torch.nn.functional.sigmoid(self.decoder.forward(shape))
        return shape
    
    def add_voxel_decoder(self, voxel_decoder=None):
        # if self.pred_voxels:
        self.decoder = voxel_decoder        


class QuatPredictor(nn.Module):
    def __init__(self, nz_feat, nz_rot, classify_rot=True):
        super(QuatPredictor, self).__init__()
        self.pred_layer = nn.Linear(nz_feat, nz_rot)
        self.classify_rot = classify_rot

    def forward(self, feat):
        quat = self.pred_layer.forward(feat)
        if self.classify_rot:
            quat = torch.nn.functional.log_softmax(quat)
        else:
            quat = torch.nn.functional.normalize(quat)
        return quat


class ScalePredictor(nn.Module):
    def __init__(self, nz):
        super(ScalePredictor, self).__init__()
        self.pred_layer = nn.Linear(nz, 3)
    
    def forward(self, feat):
        scale = self.pred_layer.forward(feat) + 1 #biasing the scale to 1
        scale = torch.nn.functional.relu(scale) + 1e-12
        # print('scale: ( Mean = {}, Var = {} )'.format(scale.mean().data[0], scale.var().data[0]))
        return scale


class TransPredictor(nn.Module):
    def __init__(self, nz):
        super(TransPredictor, self).__init__()
        self.pred_layer = nn.Linear(nz, 3)
    
    def forward(self, feat):
        #pdb.set_trace()
        trans = self.pred_layer.forward(feat)
        # print('trans: ( Mean = {}, Var = {} )'.format(trans.mean().data[0], trans.var().data[0]))
        return trans


class LabelPredictor(nn.Module):
    def __init__(self, nz_feat, classify_rot=True):
        super(LabelPredictor, self).__init__()
        self.pred_layer = nn.Linear(nz_feat, 1)

    def forward(self, feat):
        pred = self.pred_layer.forward(feat)
        pred = torch.nn.functional.sigmoid(pred)
        return pred


class CodePredictor(nn.Module):
    def __init__(
        self, nz_feat=200,
        pred_voxels=True, nz_shape=100,
        classify_rot=True, nz_rot=4
    ):
        super(CodePredictor, self).__init__()
        self.quat_predictor = QuatPredictor(nz_feat, classify_rot=classify_rot, nz_rot=nz_rot)
        self.shape_predictor = ShapePredictor(nz_feat, nz_shape=nz_shape, pred_voxels=pred_voxels)
        self.scale_predictor = ScalePredictor(nz_feat)
        self.trans_predictor = TransPredictor(nz_feat)

    def forward(self, feat):
        shape_pred = self.shape_predictor.forward(feat)
        scale_pred = self.scale_predictor.forward(feat)
        quat_pred = self.quat_predictor.forward(feat)
        trans_pred = self.trans_predictor.forward(feat)
        return shape_pred, scale_pred, quat_pred, trans_pred


class RoiEncoder(nn.Module):
    def __init__(self, nc_inp_fine, nc_inp_coarse, use_context=True, nz_joint=300, nz_roi=300, nz_coarse=300, nz_box=50):
        super(RoiEncoder, self).__init__()

        self.encoder_fine = nb.fc_stack(nc_inp_fine, nz_roi, 2)
        self.encoder_coarse = nb.fc_stack(nc_inp_coarse, nz_coarse, 2)
        self.encoder_bbox = nb.fc_stack(4, nz_box, 3)

        self.encoder_joint = nb.fc_stack(nz_roi+nz_coarse+nz_box, nz_joint, 2)
        self.use_context = use_context

    def forward(self, feats):
        roi_img_feat, img_feat_coarse, rois_inp = feats
        feat_fine = self.encoder_fine.forward(roi_img_feat)
        feat_coarse = self.encoder_coarse.forward(img_feat_coarse)

        #dividing by img_height that the inputs are not too high
        feat_bbox = self.encoder_bbox.forward(rois_inp[:, 1:5]/480.0)
        if not self.use_context:
            feat_bbox = feat_bbox*0
            feat_coarse = feat_coarse*0
        feat_coarse_rep = torch.index_select(feat_coarse, 0, rois_inp[:, 0].type(torch.LongTensor).cuda())

        # print(feat_fine.size(), feat_coarse_rep.size(), feat_bbox.size())
        feat_roi = self.encoder_joint.forward(torch.cat((feat_fine, feat_coarse_rep, feat_bbox), dim=1))
        return feat_roi


#------------- OC Net -------------#
#----------------------------------#
class OCNet(nn.Module):
    def __init__(
        self, img_size_coarse,
        roi_size=4,
        use_context=True, nz_feat=1000,
        pred_voxels=True, nz_shape=100,
        classify_rot=False, nz_rot=4,
        pred_labels=False, filter_positives=False
    ):
        super(OCNet, self).__init__()
        self.pred_labels = pred_labels
        self.filter_positives = filter_positives
        self.nz_feat = nz_feat

        self.resnet_conv_fine = ResNetConv(n_blocks=3)
        self.resnet_conv_coarse = ResNetConv(n_blocks=4)
        self.roi_size = roi_size
        self.roi_pool = roi_pool.RoIPool(roi_size, roi_size, 1/16)
        nc_inp_fine = 256*roi_size*roi_size
        nc_inp_coarse = 512*(img_size_coarse[0]//32)*(img_size_coarse[1]//32)

        self.roi_encoder = RoiEncoder(nc_inp_fine, nc_inp_coarse, use_context=use_context, nz_joint=nz_feat)

        self.code_predictor = CodePredictor(
            nz_feat=nz_feat,
            pred_voxels=pred_voxels, nz_shape=nz_shape,
            classify_rot=classify_rot, nz_rot=nz_rot)
        nb.net_init(self.roi_encoder)
        nb.net_init(self.code_predictor)

    def add_label_predictor(self):
        self.label_predictor = LabelPredictor(self.nz_feat)
        nb.net_init(self.label_predictor)

    def forward(self, imgs_rois):
        imgs_inp_fine = imgs_rois[0]
        imgs_inp_coarse = imgs_rois[1]
        rois_inp = imgs_rois[2]

        img_feat_coarse = self.resnet_conv_coarse.forward(imgs_inp_coarse)
        img_feat_coarse = img_feat_coarse.view(img_feat_coarse.size(0), -1)

        img_feat_fine = self.resnet_conv_fine.forward(imgs_inp_fine)

        roi_img_feat = self.roi_pool.forward(img_feat_fine, rois_inp)
        roi_img_feat = roi_img_feat.view(roi_img_feat.size(0), -1)

        roi_feat = self.roi_encoder.forward((roi_img_feat, img_feat_coarse, rois_inp))

        if self.pred_labels:
            labels_pred = self.label_predictor.forward(roi_feat)

        if self.filter_positives:
            pos_inds = imgs_rois[3].squeeze().data.nonzero().squeeze()
            pos_inds = torch.autograd.Variable(
                pos_inds.type(torch.LongTensor).cuda(), requires_grad=False)
            roi_feat = torch.index_select(roi_feat, 0, pos_inds)

        codes_pred = self.code_predictor.forward(roi_feat)

        if self.pred_labels:
            return codes_pred, labels_pred
        else:
            return codes_pred