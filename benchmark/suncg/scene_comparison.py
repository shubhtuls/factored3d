"""Script for icp based evaluation.
"""
# Sample usage:
# (shape_ft) : python -m factored3d.benchmark.suncg.scene_comparison --num_train_epoch=1 --name=dwr_shape_ft --classify_rot --pred_voxels=True --use_context --eval_set=val

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
import json
import time
import scipy.io as sio

from ...data import suncg as suncg_data
from . import evaluate_detection
from ...external.pythonpcl import pcl
from ...utils import bbox_utils
from ...utils import suncg_parse
from ...nnutils import test_utils
from ...nnutils import net_blocks
from ...nnutils import loss_utils
from ...nnutils import voxel_net
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
flags.DEFINE_integer('max_icp_iterations', 20, 'If we have more objects than this per image, we will subsample.')
flags.DEFINE_integer('max_total_rois', 100, 'If we have more objects than this per batch, we will reject the batch.')

flags.DEFINE_string('layout_name', 'layout_pred', 'Experiment name for layout predictor')
flags.DEFINE_integer('layout_train_epoch', 8, 'Experiment name for layout predictor')

flags.DEFINE_string('depth_name', 'depth_baseline', 'Experiment name for layout predictor')
flags.DEFINE_integer('depth_train_epoch', 8, 'Epoch for layout predictor')

flags.DEFINE_string('scene_voxels_name', 'voxels_baseline', 'Experiment name for scene voxels predictor')
flags.DEFINE_integer('scene_voxels_train_epoch', 8, 'Epoch for scene voxels predictor')
flags.DEFINE_float('scene_voxels_thresh', 0.1, 'Threshold for scene voxels prediction')
FLAGS = flags.FLAGS


class SceneComparisonTester(test_utils.Tester):
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

        # Useful for evaluation
        code_tensors = suncg_parse.collate_codes(batch['codes'])
        code_tensors[0] = code_tensors[0].unsqueeze(1) #unsqueeze voxels

        self.codes_gt = [
            Variable(t.cuda(device=opts.gpu_id), requires_grad=False) for t in code_tensors]
        self.depth_gt = Variable(
            batch['depth'].cuda(device=opts.gpu_id), requires_grad=False)
        self.layout_gt = Variable(
            batch['layout'].cuda(device=opts.gpu_id), requires_grad=False)
        self.scene_voxels_gt = Variable(
            batch['voxels'].unsqueeze(1).cuda(device=opts.gpu_id), requires_grad=False)
        self.rois_gt = Variable(
            bboxes_gt.type(torch.FloatTensor).cuda(device=opts.gpu_id), requires_grad=False)
        if self.downsample_voxels:
            self.codes_gt[0] = self.downsampler.forward(self.codes_gt[0])

    def filter_pos(self, codes, pos_inds):
        pos_inds = torch.from_numpy(np.array(pos_inds)).squeeze()
        pos_inds = torch.autograd.Variable(
                pos_inds.type(torch.LongTensor).cuda(), requires_grad=False)
        filtered_codes = [torch.index_select(code, 0, pos_inds) for code in codes]
        return filtered_codes

    def predict_factored3d(self):
        codes_pred_all, labels_pred = self.oc3d_model.forward((self.input_imgs_fine, self.input_imgs, self.rois))
        scores_pred = labels_pred.cpu().data.numpy()
        bboxes_pred = self.rois.data.cpu().numpy()[:, 1:]
        min_score_vis = np.minimum(0.5, np.max(scores_pred))
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
        
    def save_point_clouds(self, pts_dict, suffix=''):
        """Saves input point clouds.
        
        Args:
            pts_dict: dict of pt clouds
        """
        mesh_dir = osp.join(self.opts.rendering_dir, self.opts.name, 'iter{}'.format(self.vis_iter))
        if not os.path.exists(mesh_dir):
            os.makedirs(mesh_dir)
        for pt_name in pts_dict:
            if isinstance(pts_dict[pt_name], np.ndarray):
                mesh_file = osp.join(mesh_dir, pt_name + '{}.obj'.format(suffix))
                fout = open(mesh_file, 'w')
                render_utils.append_obj(fout, pts_dict[pt_name], np.zeros((0, 3)))
                fout.close()
            else:
                for ox in range(len(pts_dict[pt_name])):
                    mesh_file = osp.join(mesh_dir, pt_name + '_inst{}{}.obj'.format(ox, suffix))
                    fout = open(mesh_file, 'w')
                    render_utils.append_obj(fout, pts_dict[pt_name][ox], np.zeros((0, 3)))
                    fout.close()

    def scene_voxel_points(self, scene_voxels, thresh=0.25):
        """Returns points for occupied voxels.
        
        Args:
            scene_voxels: 1 X 1 X W X H X D occupancies, torch Variable
            thresh: voxelization threshold
        Returns:
            n X 3 numpy array
        """
        scene_voxels = scene_voxels.data.cpu()[0, 0].numpy()
        vs = render_utils.voxels_to_points(scene_voxels.astype(np.float32), thresh=thresh)
        vs[:,0] -= scene_voxels.shape[0]/2.0
        vs[:,1] -= scene_voxels.shape[1]/2.0
        vs *= 0.04*(64//self.opts.voxels_height)
        return vs
    
    def points_to_scene_voxels(self, points):
        """Returns scene_voxels given points.
        
        Args:
            points: n X 3 numpy array
        Returns:
            scene_voxels: W X H X D occupancies
        """
        scene_voxels = np.zeros((self.opts.voxels_width, self.opts.voxels_height, self.opts.voxels_depth))
        vs = np.copy(points)
        vs /= 0.04*(64//self.opts.voxels_height)
        vs[:,0] += scene_voxels.shape[0]/2.0
        vs[:,1] += scene_voxels.shape[1]/2.0
        vs = np.floor(vs).astype(np.int32)
        valid_inds = (vs[:, 0] >= 0) & (vs[:, 1] >= 0) & (vs[:, 2] >= 0) & (vs[:, 0] < self.opts.voxels_width) & (vs[:, 1] < self.opts.voxels_height) & (vs[:, 2] < self.opts.voxels_depth)
        vs = vs[valid_inds, :]
        for v in range(vs.shape[0]):
            x = vs[v, 0]
            y = vs[v, 1]
            z = vs[v, 2]
            if scene_voxels[x, y, z] == 0:
                scene_voxels[x, y, z] = 1
        return scene_voxels
    
    def fov_scene_voxels(self):
        """Returns scene_voxels within camera fov.
        
        Returns:
            valid_voxels: W X H X D occupancies
        """
        valid_voxels = np.zeros((self.opts.voxels_width, self.opts.voxels_height, self.opts.voxels_depth))
        cam_k = suncg_parse.cam_intrinsic()
        for x in range(self.opts.voxels_width):
            for y in range(self.opts.voxels_height):
                for z in range(self.opts.voxels_depth):
                    pt3d = np.array([x - valid_voxels.shape[0]/2.0, y - valid_voxels.shape[1]/2.0, z]) + 0.5
                    pt3d *= 0.04
                    cam_pt = np.matmul(cam_k, pt3d.reshape((3, 1))).reshape(3)
                    cam_pt /= cam_pt[2]
                    if cam_pt[0] >= 0 and cam_pt[1] >= 0 and cam_pt[0] <= 640 and cam_pt[1] <= 480:
                        valid_voxels[x, y, z] = 1
        return valid_voxels
    
    def dmap_points(self, dmap, min_disp=1e-1):
        """Returns points for depth map.
        
        Args:
            dmap: 1 X 1 X H X W depths, torch Variable
            thresh: voxelization threshold
        Returns:
            n X 3 numpy array
        """
        dmap = dmap.data[0].cpu().numpy().transpose((1,2,0))
        dmap_points = render_utils.dispmap_to_points(
            dmap,
            suncg_parse.cam_intrinsic(),
            scale_x=self.opts.layout_width/640,
            scale_y=self.opts.layout_height/480,
            min_disp=min_disp
        )
        return dmap_points
    
    def oc3d_points(self, codes, thresh=0.25, objectwise=False):
        """Returns points for depth map.
        
        Args:
            codes: list of torch Variables for shape, scale, rotation, translation
            thresh: voxelization threshold
            objectwise: return per object array or one single array
        Returns:
            n X 3 numpy array
        """
        n_rois = codes[0].size()[0]
        code_list = suncg_parse.uncollate_codes(codes, 1, torch.Tensor(n_rois).fill_(0))
        pts = render_utils.codes_to_points(code_list[0], thresh=thresh, objectwise=objectwise)
        return pts
    
    def get_current_visuals(self):
        visuals = {}
        return visuals

    def predict(self):
        codes_pred, layout_pred = self.predict_factored3d()
        self.pts_pred_factored = np.concatenate([
            self.oc3d_points(codes_pred), self.dmap_points(layout_pred, min_disp=1e-1)], axis=0)
        
        scene_voxels_pred = self.predict_scene_voxels()
        self.pts_pred_voxels = self.scene_voxel_points(scene_voxels_pred, thresh=self.opts.scene_voxels_thresh)
        
        disp_pred = self.predict_depth()
        self.pts_pred_depth = self.dmap_points(disp_pred, min_disp=1e-1)

    def _pcl_align(self, src, target, max_iter):
        icp = pcl.IterativeClosestPoint()
        converged, transf, estimate, fitness = icp.icp(src, target, max_iter=max_iter)
        return converged, transf, estimate, fitness

    def evaluate(self):
        pts_gt_objects = self.oc3d_points(self.codes_gt, thresh=0.25, objectwise=True)
        object_scales = self.codes_gt[1].cpu().norm(p=2, dim=1).pow(2)

        pts_gt_depth = [self.dmap_points(self.depth_gt)]        
        pts_gt_volume = [self.scene_voxel_points(self.scene_voxels_gt)]

        pts_gt_layout = [self.dmap_points(self.layout_gt)]

        foreground_layout = (self.layout_gt == self.depth_gt).double()
        pts_gt_fg_layout = [self.dmap_points(self.layout_gt*foreground_layout)]

        # For aligning objects to the point cloud of the scene.
        predictions = [self.pts_pred_factored, self.pts_pred_depth, self.pts_pred_voxels]
        if not all([p.size > 0 for p in predictions]):
            return None
        pred_methods = ['factored', 'depth', 'voxel']

        gt_names = ['objects', 'surface', 'volume', 'visible_layout', 'amodal_layout']

        gts = [
            pts_gt_objects, pts_gt_depth,
            self.scene_voxels_gt.data[0, 0].cpu().numpy(),
            pts_gt_fg_layout, pts_gt_layout]

        all_fitnesses = []
        for gt, gt_name in zip(gts, gt_names):
            fitnesses = []
            for prediction, pred_method in zip(predictions, pred_methods):
                if gt_name == 'objects':
                    icp_iteration = self.opts.max_icp_iterations
                else:
                    icp_iteration = 0
                if gt_name == 'volume':
                    gt_vol = gt
                    pred_vol = self.points_to_scene_voxels(prediction)
                    pred_vol = np.multiply(pred_vol, self.fov_valid_voxels)
                    intersection = np.multiply(gt_vol, pred_vol).sum()
                    union = gt_vol.sum() + pred_vol.sum() - intersection
                    #fitness = intersection/gt_vol.sum()
                    fitness = intersection/union
                else:
                    pc_scene = pcl.PointCloud()
                    pc_scene.from_array(prediction.astype(np.float32))
                    pts_aligned_objects, fitness, converged = [], [], []
                    for i, gt_obj in enumerate(gt):
                        pc_gt_obj = pcl.PointCloud()
                        pc_gt_obj.from_array(gt_obj.astype(np.float32))
                        converged_, transf_, estimate_, fitness_ = self._pcl_align(pc_gt_obj, pc_scene, icp_iteration)
                        # Transform the point cloud based on transformation
                        if gt_name == 'objects':
                            fitness.append(fitness_/object_scales[i])
                        else:
                            fitness.append(fitness_)
                        pts_aligned_objects.append(estimate_.to_array()*1.)
                fitnesses.append(fitness)
            fitnesses = np.array(fitnesses).reshape([3, -1])
            all_fitnesses.append(fitnesses)
        return all_fitnesses

    def test(self):
        opts = self.opts
        self.fov_valid_voxels = self.fov_scene_voxels()
        n_iter = len(self.dataloader)
        fitnesses = []
        for i, batch in enumerate(self.dataloader):
            if i % 10 == 0:
                print('{}/{} evaluation iterations.'.format(i, n_iter))
            if opts.max_eval_iter > 0 and (i >= opts.max_eval_iter):
                break
            self.set_input(batch)
            if not self.invalid_batch:
                self.predict()
                fitness = self.evaluate()
                if fitness is None:
                    continue
                self.save_current_visuals()
                fitnesses.append(fitness)
        fitness = zip(*fitnesses)
        fitness = [np.concatenate(ff, 1) for ff in fitness]
        print([np.mean(ff, 1) for ff in fitness])
        results_file = osp.join(FLAGS.results_eval_dir, 'results.mat')
        if not os.path.exists(opts.results_eval_dir):
            os.makedirs(opts.results_eval_dir)

        results = {
            'object_eval': fitness[0], 'depth_eval': fitness[1],
            'volume_overlap_eval': fitness[2],
            'layout_eval': fitness[3], 'layout_amodal_eval': fitness[4]
        }
        sio.savemat(results_file, results)


def main(_):
    FLAGS.suncg_dl_out_codes = True
    FLAGS.suncg_dl_out_fine_img = True
    FLAGS.suncg_dl_out_test_proposals = True
    FLAGS.suncg_dl_out_voxels = True
    FLAGS.suncg_dl_out_layout = True
    FLAGS.suncg_dl_out_depth = True
    FLAGS.n_data_workers = 0
    FLAGS.max_views_per_house = 2
    FLAGS.batch_size = 1
    assert(FLAGS.batch_size == 1)

    FLAGS.results_vis_dir = osp.join(FLAGS.results_vis_dir, 'icp', FLAGS.eval_set, FLAGS.name)
    FLAGS.results_eval_dir = osp.join(FLAGS.results_eval_dir, 'icp', FLAGS.eval_set, FLAGS.name)
    if not os.path.exists(FLAGS.results_eval_dir):
        os.makedirs(FLAGS.results_eval_dir)
    if not os.path.exists(FLAGS.results_vis_dir):
        os.makedirs(FLAGS.results_vis_dir)
    torch.manual_seed(0)

    if FLAGS.classify_rot:
        FLAGS.nz_rot = 24
    else:
        FLAGS.nz_rot = 4

    tester = SceneComparisonTester(FLAGS)
    tester.init_testing()
    tester.test()


if __name__ == '__main__':
    app.run()