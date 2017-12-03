"""Script for dwr prediction benchmarking.
"""
# Sample usage:
# (shape_ft) : python -m factored3d.benchmark.suncg.dwr --num_train_epoch=1 --name=dwr_shape_ft --classify_rot --pred_voxels=True --use_context  --save_visuals --visuals_freq=50 --eval_set=val  --suncg_dl_debug_mode  --max_eval_iter=20
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from google.apputils import app
import gflags as flags
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
import json
import matplotlib.pyplot as plt
import time

from ...data import suncg as suncg_data
from . import evaluate_detection
from ...utils import bbox_utils
from ...utils import suncg_parse
from ...nnutils import test_utils
from ...nnutils import net_blocks
from ...nnutils import loss_utils
from ...nnutils import oc_net
from ...nnutils import disp_net
from ...utils import metrics
from ...utils import visutil
from ...renderer import utils as render_utils

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', '..', 'cachedir')
flags.DEFINE_string('rendering_dir', osp.join(cache_path, 'rendering'), 'Directory where intermittent renderings are saved')

flags.DEFINE_integer('voxel_size', 32, 'Spatial dimension of shape voxels')
flags.DEFINE_integer('n_voxel_layers', 5, 'Number of layers ')
flags.DEFINE_integer('voxel_nc_max', 128, 'Max 3D channels')
flags.DEFINE_integer('voxel_nc_l1', 8, 'Initial shape encder/decoder layer dimension')
flags.DEFINE_float('voxel_eval_thresh', 0.25, 'Voxel evaluation threshold')

flags.DEFINE_string('shape_pretrain_name', 'object_autoenc_32', 'Experiment name for pretrained shape encoder-decoder')
flags.DEFINE_integer('shape_pretrain_epoch', 800, 'Experiment name for shape decoder')

flags.DEFINE_integer('max_rois', 100, 'If we have more objects than this per image, we will subsample.')
flags.DEFINE_integer('max_total_rois', 100, 'If we have more objects than this per batch, we will reject the batch.')

flags.DEFINE_string('layout_name', 'layout_pred', 'Experiment name for layout predictor')
flags.DEFINE_integer('layout_train_epoch', 8, 'Experiment name for layout predictor')

FLAGS = flags.FLAGS

EP_box_iou_thresh =     [0.5   , 0.5      , 0.5    , 0.5      , 0.       , 0.5      , 0.5     , 0.5         , 0.5           , 0.5           , 0.5           , 0.5               , ]
EP_rot_delta_thresh =   [30.   , 30.      , 400.   , 30.      , 30.      , 30.      , 400.    , 30.         , 400.          , 400.          , 400.          , 30                , ]
EP_trans_delta_thresh = [1.    , 1.       , 1.     , 1000.    , 1        , 1.       , 1000.   , 1000.       , 1.            , 1000.         , 1000.         , 1000.             , ]
EP_shape_iou_thresh =   [0.25  , 0        , 0.25   , 0.25     , 0.25     , 0.25     , 0       , 0           , 0             , 0.25          , 0             , 0.25              , ]
EP_scale_delta_thresh = [0.5   , 0.5      , 0.5    , 0.5      , 0.5      , 100.     , 100.    , 100         , 100           , 100           , 0.5           , 100               , ]
EP_ap_str =             ['all' , '-shape' , '-rot' , '-trans' , '-box2d' , '-scale' , 'box2d' , 'box2d+rot' , 'box2d+trans' , 'box2d+shape' , 'box2d+scale' , 'box2d+rot+shape' , ]

class DWRTester(test_utils.Tester):
    def define_model(self):
        '''
        Define the pytorch net 'model' whose weights will be updated during training.
        '''
        opts = self.opts

        self.voxel_encoder, nc_enc_voxel = net_blocks.encoder3d(
            opts.n_voxel_layers, nc_max=opts.voxel_nc_max, nc_l1=opts.voxel_nc_l1, nz_shape=opts.nz_shape)

        self.voxel_decoder = net_blocks.decoder3d(
            opts.n_voxel_layers, opts.nz_shape, nc_enc_voxel, nc_min=opts.voxel_nc_l1)

        self.model = oc_net.OCNet(
            (opts.img_height, opts.img_width),
            roi_size=opts.roi_size,
            use_context=opts.use_context, nz_feat=opts.nz_feat,
            pred_voxels=False, nz_shape=opts.nz_shape, pred_labels=True,
            classify_rot=opts.classify_rot, nz_rot=opts.nz_rot)
        self.model.add_label_predictor()

        if opts.pred_voxels:
            self.model.code_predictor.shape_predictor.add_voxel_decoder(
                copy.deepcopy(self.voxel_decoder))

        self.load_network(self.model, 'pred', self.opts.num_train_epoch)
        self.model.eval()
        self.model = self.model.cuda(device_id=self.opts.gpu_id)

        if opts.pred_voxels:
             self.voxel_decoder = copy.deepcopy(self.model.code_predictor.shape_predictor.decoder)

        self.layout_model = disp_net.dispnet()
        network_dir = osp.join(opts.cache_dir, 'snapshots', opts.layout_name)
        self.load_network(
            self.layout_model, 'pred', opts.layout_train_epoch, network_dir=network_dir)
        self.layout_model.eval()
        self.layout_model = self.layout_model.cuda(device_id=self.opts.gpu_id)

        return

    def init_dataset(self):
        opts = self.opts
        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        split_dir = osp.join(opts.suncg_dir, 'splits')
        self.split = suncg_parse.get_split(split_dir, house_names=os.listdir(osp.join(opts.suncg_dir, 'camera')))
        self.dataloader = suncg_data.suncg_data_loader_benchmark(self.split[opts.eval_set], opts)

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
        bboxes_gt = suncg_parse.bboxes_to_rois(batch['bboxes'])
        bboxes_proposals = suncg_parse.bboxes_to_rois(batch['bboxes_test_proposals'])
        rois = bboxes_proposals
        if rois.numel() <= 0 or bboxes_gt.numel() <= 0: #some proposals and gt objects should be there
            self.invalid_batch = True
            return
        else:
            self.invalid_batch = False

        # Inputs for prediction
        input_imgs_fine = batch['img_fine'].type(torch.FloatTensor)
        input_imgs = batch['img'].type(torch.FloatTensor)

        self.input_imgs_layout = Variable(
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

        # Useful for evaluation
        code_tensors = suncg_parse.collate_codes(batch['codes'])
        code_tensors[0] = code_tensors[0].unsqueeze(1) #unsqueeze voxels
        self.layout_gt = Variable(
            batch['layout'].cuda(device=opts.gpu_id), requires_grad=False)
        self.codes_gt = [
            Variable(t.cuda(device=opts.gpu_id), requires_grad=False) for t in code_tensors]
        self.rois_gt = Variable(
            bboxes_gt.type(torch.FloatTensor).cuda(device=opts.gpu_id), requires_grad=False)
        if self.downsample_voxels:
            self.codes_gt[0] = self.downsampler.forward(self.codes_gt[0])


    def save_layout_mesh(self, mesh_dir, layout, prefix='layout'):
        opts = self.opts
        layout_vis = layout.data[0].cpu().numpy().transpose((1,2,0))
        mesh_file = osp.join(mesh_dir, prefix + '.obj')
        vs, fs = render_utils.dispmap_to_mesh(
            layout_vis,
            suncg_parse.cam_intrinsic(),
            scale_x=self.opts.layout_width/640,
            scale_y=self.opts.layout_height/480
        )
        fout = open(mesh_file, 'w')
        mesh_file = osp.join(mesh_dir, prefix + '.obj')
        fout = open(mesh_file, 'w')
        render_utils.append_obj(fout, vs, fs)
        fout.close()

    def save_codes_mesh(self, mesh_dir, code_vars, prefix='codes'):
        opts = self.opts

        n_rois = code_vars[0].size()[0]
        code_list = suncg_parse.uncollate_codes(code_vars, self.input_imgs.data.size(0), torch.Tensor(n_rois).fill_(0))

        if not os.path.exists(mesh_dir):
            os.makedirs(mesh_dir)
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

    def get_current_visuals(self):
        visuals = {}
        opts = self.opts
        visuals['a_img'] = visutil.tensor2im(visutil.undo_resnet_preprocess(
            self.input_imgs_fine.data))

        mesh_dir = osp.join(opts.rendering_dir, opts.name)
        vis_codes = [self.codes_pred_vis, self.codes_gt]
        vis_layouts = [self.layout_pred, self.layout_gt]
        vis_names = ['b_pred', 'c_gt']
        for vx, v_name in enumerate(vis_names):
            os.system('rm {}/*.obj'.format(mesh_dir))
            self.save_codes_mesh(mesh_dir, vis_codes[vx])
            self.save_layout_mesh(mesh_dir, vis_layouts[vx])

            visuals['{}_layout_cam_view'.format(v_name)], visuals['{}_layout_novel_view'.format(v_name)] = self.render_visuals(
                mesh_dir, obj_name='layout')
            visuals['{}_objects_cam_view'.format(v_name)], visuals['{}_objects_novel_view'.format(v_name)] = self.render_visuals(
                mesh_dir, obj_name='codes')
            visuals['{}_scene_cam_view'.format(v_name)], visuals['{}_scene_novel_view'.format(v_name)] = self.render_visuals(
                mesh_dir)
        return visuals


    def filter_pos(self, codes, pos_inds):
        pos_inds = torch.from_numpy(np.array(pos_inds)).squeeze()
        pos_inds = torch.autograd.Variable(
                pos_inds.type(torch.LongTensor).cuda(), requires_grad=False)
        filtered_codes = [torch.index_select(code, 0, pos_inds) for code in codes]
        return filtered_codes

    def predict(self):
        codes_pred_all, labels_pred = self.model.forward((self.input_imgs_fine, self.input_imgs, self.rois))
        scores_pred = labels_pred.cpu().data.numpy()
        bboxes_pred = self.rois.data.cpu().numpy()[:, 1:]
        min_score_eval = np.minimum(0.05, np.max(scores_pred))
        pos_inds_eval = metrics.nms(
            np.concatenate((bboxes_pred, scores_pred), axis=1),
            0.3, min_score=min_score_eval)

        self.codes_pred_eval = self.filter_pos(codes_pred_all, pos_inds_eval)
        self.rois_pos_eval = self.filter_pos([self.rois], pos_inds_eval)[0]     # b x 5, 1:5 is box (x1 y1 x2 y2)
        self.codes_pred_eval[0] = self.decode_shape(self.codes_pred_eval[0])    # b x 1 x 32 x 32 x 32
        self.codes_pred_eval[2] = self.decode_rotation(self.codes_pred_eval[2]) # b x 4
        self.codes_pred_eval[1] # Probably scale b x 3
        self.codes_pred_eval[3] # Probably trans b x 3
        self.scores_pred_eval = scores_pred[pos_inds_eval,:]*1.

        min_score_vis = np.minimum(0.7, np.max(scores_pred))
        pos_inds_vis = metrics.nms(
            np.concatenate((bboxes_pred, scores_pred), axis=1),
            0.3, min_score=min_score_vis)
        self.codes_pred_vis = self.filter_pos(codes_pred_all, pos_inds_vis)
        self.rois_pos_vis = self.filter_pos([self.rois], pos_inds_vis)[0]
        self.codes_pred_vis[0] = self.decode_shape(self.codes_pred_vis[0])
        self.codes_pred_vis[2] = self.decode_rotation(self.codes_pred_vis[2])

        self.layout_pred = self.layout_model.forward(self.input_imgs_layout)

    def evaluate(self):
        # rois as numpy array
        # Get Predictions.
        shapes, scales, rots, trans = self.codes_pred_eval
        scores = self.scores_pred_eval
        boxes = self.rois_pos_eval.cpu().data.numpy()[:,1:]
        # Get Ground Truth.
        gt_shapes, gt_scales, gt_rots, gt_trans = self.codes_gt
        gt_boxes = self.rois_gt.cpu().data.numpy()[:,1:]

        iou_box = bbox_utils.bbox_overlaps(boxes.astype(np.float), gt_boxes.astype(np.float))

        trans_, gt_trans_ = trans.cpu().data.numpy(), gt_trans.cpu().data.numpy()
        err_trans = np.linalg.norm(np.expand_dims(trans_, 1) - np.expand_dims(gt_trans_, 0), axis=2)

        scales_, gt_scales_ = scales.cpu().data.numpy(), gt_scales.cpu().data.numpy()
        err_scales = np.mean(np.abs(np.expand_dims(np.log(scales_), 1) - np.expand_dims(np.log(gt_scales_), 0)), axis=2)
        err_scales /= np.log(2.0)

        ndt, ngt = err_scales.shape
        err_shapes = err_scales*0.
        err_rots = err_scales*0.

        for i in range(ndt):
          for j in range(ngt):
            err_shapes[i,j] = metrics.volume_iou(shapes[i,0].data, gt_shapes[j,0].data, thresh=self.opts.voxel_eval_thresh)
            err_rots[i,j] = metrics.quat_dist(rots[i].data.cpu(), gt_rots[j].data.cpu())

        # Run the benchmarking code here.
        threshs = [EP_box_iou_thresh, EP_rot_delta_thresh, EP_trans_delta_thresh, EP_shape_iou_thresh, EP_scale_delta_thresh]
        fn = [np.greater_equal, np.less_equal, np.less_equal, np.greater_equal, np.less_equal]
        overlaps = [iou_box, err_rots, err_trans, err_shapes, err_scales]

        _dt = {'sc': scores}
        _gt = {'diff': np.zeros((ngt,1), dtype=np.bool)}
        _bopts = {'minoverlap': 0.5}
        stats = []
        for i in range(len(EP_ap_str)):
          # Compute a single overlap that ands all the thresholds.
          ov = []
          for j in range(len(overlaps)):
            ov.append(fn[j](overlaps[j], threshs[j][i]))
          _ov = np.all(np.array(ov), 0).astype(np.float32)
          # Benchmark for this setting.
          tp, fp, sc, num_inst, dup_det, inst_id, ov = evaluate_detection.inst_bench_image(_dt, _gt, _bopts, _ov)
          stats.append([tp, fp, sc, num_inst, dup_det, inst_id, ov])
        return stats

    def test(self):
        opts = self.opts
        bench_stats = []
        # codes are (shapes, scales, quats, trans)
        n_iter = len(self.dataloader)
        for i, batch in enumerate(self.dataloader):
            if i % 100 == 0:
                print('{}/{} evaluation iterations.'.format(i, n_iter))
            if opts.max_eval_iter > 0 and (i >= opts.max_eval_iter):
                break
            self.set_input(batch)
            if not self.invalid_batch:
                self.predict()
                bench_image_stats = self.evaluate()
                bench_stats.append(bench_image_stats)
                if opts.save_visuals and (i % opts.visuals_freq == 0):
                    self.save_current_visuals()

        # Accumulate stats, write to FLAGS.results_eval_dir.
        bb = zip(*bench_stats)
        bench_summarys = []
        eval_params = {}
        for i in range(len(EP_ap_str)):
            tp, fp, sc, num_inst, dup_det, inst_id, ov = zip(*bb[i])
            ap, rec, prec, npos, details = evaluate_detection.inst_bench(None, None, None, tp, fp, sc, num_inst)
            bench_summary = {'prec': prec.tolist(), 'rec': rec.tolist(), 'ap': ap[0], 'npos': npos}
            print('{:>20s}: {:5.3f}'.format(EP_ap_str[i], ap[0]*100.))
            bench_summarys.append(bench_summary)
        eval_params['ap_str'] = EP_ap_str
        eval_params['set'] = opts.eval_set

        json_file = os.path.join(FLAGS.results_eval_dir, 'eval_set{}_{}.json'.format(opts.eval_set, 0))
        print('Writing results to file: {:s}'.format(json_file))
        with open(json_file, 'w') as f:
            json.dump({'eval_params': eval_params, 'bench_summary': bench_summarys}, f)

def main(_):
    FLAGS.suncg_dl_out_codes = True
    FLAGS.suncg_dl_out_fine_img = True
    FLAGS.suncg_dl_out_test_proposals = True
    FLAGS.suncg_dl_out_voxels = False
    FLAGS.suncg_dl_out_layout = True
    FLAGS.suncg_dl_out_depth = False
    FLAGS.n_data_workers = 0
    FLAGS.max_views_per_house = 2
    FLAGS.batch_size = 1
    assert(FLAGS.batch_size == 1)

    FLAGS.results_vis_dir = osp.join(FLAGS.results_vis_dir, 'dwr', FLAGS.eval_set, FLAGS.name)
    FLAGS.results_eval_dir = osp.join(FLAGS.results_eval_dir, 'dwr', FLAGS.eval_set, FLAGS.name)
    if not os.path.exists(FLAGS.results_eval_dir):
        os.makedirs(FLAGS.results_eval_dir)
    if not os.path.exists(FLAGS.results_vis_dir):
        os.makedirs(FLAGS.results_vis_dir)
    torch.manual_seed(0)

    if FLAGS.classify_rot:
        FLAGS.nz_rot = 24
    else:
        FLAGS.nz_rot = 4

    tester = DWRTester(FLAGS)
    tester.init_testing()
    tester.test()


if __name__ == '__main__':
    app.run()