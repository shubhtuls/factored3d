from __future__ import division
from __future__ import print_function
import copy
import csv
import json
import numpy as np
import scipy.linalg
import scipy.io as sio
import os
import os.path as osp
import pickle
import torch
from . import transformations
from . import bbox_utils

valid_object_classes = [
    'bed', 'sofa', 'table', 'chair', 'desk', 'television',
    #'cabinet', 'counter', 'refridgerator', 'night_stand', 'toilet', 'bookshelf', 'shelves', 'bathtub'
]
list.sort(valid_object_classes)

def load_json(json_file):
    '''
    Parse a json file and return a dictionary.

    Args:
        json_file: Absolute path of json file
    Returns:
        var: json data as a nested dictionary
    '''
    with open(json_file) as f:
        var = json.load(f)
        return var


#------------ Cameras -------------#
#----------------------------------#
def read_node_indices(node_file):
    '''
    Returns a dictionary mapping node index of entity name.

    Args:
        node_file: Absolute path of node indices file
    Returns:
        node_dict: output mapping
    '''
    node_dict = {}
    with open(node_file) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        for c in content:
            index, entity = c.split()
            node_dict[int(index)] = entity
    return node_dict


def read_camera_pose(cam_file):
    '''
    Returns a list of camera poses stored in the cam_file.

    Args:
        cam_file: Absolute path of camera file
    Returns:
        cam_data: List of cameras, each camera pose is a list of 12 numbers
    '''
    with open(cam_file) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        cam_data = [[float(v) for v in l.split()] for l in content]
        return cam_data


def cam_intrinsic():
    '''
    SUNCG rendering intrinsic matrix.

    Returns:
        3 X 3 intrinsic matrix
    '''
    return np.array([
        [517.97, 0, 320],
        [0, 517.97, 240],
        [0, 0, 1]
    ])


def campose_to_extrinsic(campose):
    '''
    Obtain an extrinsic matrix from campose format.

    Based on sscnet implementation
    (https://github.com/shurans/sscnet/blob/master/matlab_code/utils/camPose2Extrinsics.m)
    Args:
        campose: list of 12 numbers indicting the camera pose
    Returns:
        extrinsic: 4 X 4 extrinsic matrix
    '''
    tv = np.array(campose[3:6])
    uv = np.array(campose[6:9])
    rv = np.cross(tv,uv)
    trans = np.array(campose[0:3]).reshape(3, 1)

    extrinsic = np.concatenate((
        rv.reshape(3, 1),
        -1*uv.reshape(3, 1),
        tv.reshape(3, 1),
        trans.reshape(3, 1)), axis=1)
    extrinsic = np.concatenate((extrinsic, np.array([[0,0,0,1]])),axis=0)
    return extrinsic


#--------- Intersection -----------#
#----------------------------------#
def dehomogenize_coord(point):
    nd = point.size
    return point[0:nd-1]/point[nd]


def is_inside_box(point, bbox):
    c1 = np.all(point.reshape(-1) >= np.array(bbox['min']))
    c2 = np.all(point.reshape(-1) <= np.array(bbox['max']))
    return c1 and c2


def intersects(obj_box, grid_box, transform):
    '''
    Check if the trnsformed obj_box intersects the grid_box.

    Args:
        obj_box: 3D bbox of object in world frame
        grid_box: grid in camera frame
        transform: extrinsic matrix for world2camera frame
    Returns:
        True/False indicating if the boxes intersect
    '''
    for ix in range(8):
        point = [0, 0, 0, 1]
        point[0] = obj_box['min'][0] if ((ix//1)%2 == 0) else obj_box['max'][0]
        point[1] = obj_box['min'][1] if ((ix//2)%2 == 0) else obj_box['max'][1]
        point[2] = obj_box['min'][2] if ((ix//4)%2 == 0) else obj_box['max'][2]
        pt_transformed = np.matmul(transform, np.array(point).reshape(4,1))
        if is_inside_box(dehomogenize_coord(pt_transformed), grid_box):
            return True

    return False


#--------- House Parsing ----------#
#----------------------------------#
def select_ids(house_data, bbox_data, min_sum_dims=0, min_pixels=0, meta_loader=None):
    '''
    Selects objects which are indexed by bbox_data["node_ids"].

    Args:
        house_data: House Data
        bbox_data: ids for desired objects, and the number of pixels for each
    Returns:
        house_data with nodes only for the selected objects
    '''
    house_copy = {}
    room_node = {}
    for k in house_data.keys():
        if(k != 'levels'):
            house_copy[k] = copy.deepcopy(house_data[k])
            room_node[k] = copy.deepcopy(house_data[k])
    house_copy['levels'] = [{}] #empty dict
    room_node['nodeIndices'] = []
    house_copy['levels'][0]['nodes'] = [room_node]

    bboxes = bbox_data['bboxes']
    node_ids = bbox_data['ids']
    n_pixels = bbox_data['nPixels']
    n_selected = 0
    selected_inds = []
    for ix in range(n_pixels.size):
        lx = int(node_ids[ix]//10000) - 1
        nx = int(node_ids[ix]%10000) - 1

        obj_node = house_data['levels'][lx]['nodes'][nx]
        select_node = obj_node['type'] == 'Object'

        if min_sum_dims>0 and select_node:
            select_node = sum(obj_node['bbox']['max']) >= (sum(obj_node['bbox']['min']) + min_sum_dims)

        if min_pixels>0 and select_node:
            select_node = n_pixels[ix] > min_pixels
        
        if (meta_loader is not None) and select_node:
            object_class = meta_loader.lookup(obj_node['modelId'], field='nyuv2_40class')
            select_node = object_class in valid_object_classes

        if select_node:
            selected_inds.append(ix)
            n_selected = n_selected + 1
            house_copy['levels'][0]['nodes'][0]['nodeIndices'].append(n_selected)
            house_copy['levels'][0]['nodes'].append(copy.deepcopy(obj_node))

    return house_copy, np.copy(bboxes[selected_inds, :]).astype(np.float)


def select_layout(house_data):
    '''
    Selects all room nodes while ignoring objects.

    Args:
        house_data: House Data
    Returns:
        house_data with nodes only for the rooms
    '''
    house_copy = copy.deepcopy(house_data)
    for lx in range(len(house_data['levels'])):
        nodes_layout = []
        for nx in range(len(house_data['levels'][lx]['nodes'])):
            node = house_data['levels'][lx]['nodes'][nx]
            if node['type'] == 'Room':
                node_copy = copy.deepcopy(node)
                node_copy['nodeIndices'] = None
                nodes_layout.append(node_copy)
        house_copy['levels'][ls]['nodes'] = nodes_layout
    return house_copy


#-------- Voxel Processing --------#
#----------------------------------#
def prune_exterior_voxels(voxels, voxel_coords, grid_box, cam2world, slack=0.05):
    '''
    Sets occupancies of voxels outside grid box to 0.

    Args:
        voxels: grid indicating occupancy at each location
        voxel_coords: 4 X grid_size voxel coordinates (in camera frame)
        grid_box: area of interest (in world frame)
        cam2world: transformation matrix
        slack: The grid_box is assumed to be a bit bigger by this fraction
    Returns:
        voxels with some occupancies outside set to 0
    '''
    min_dims = np.array(grid_box['min']).reshape(-1, 1)
    max_dims = np.array(grid_box['max']).reshape(-1, 1)
    box_slack = (max_dims - min_dims)*slack
    min_dims -= box_slack
    max_dims += box_slack

    ndims = min_dims.size
    cam2world = cam2world[0:ndims-1, :]

    voxel_coords = np.matmul(cam2world, voxel_coords.reshape(ndims, -1))
    is_inside = (voxel_coords >= min_dims) & (voxel_coords <= max_dims)
    is_inside = np.all(is_inside, axis=0, keep_dims=False).reshape(voxels.shape)
    return np.multiply(voxels, is_inside.astype(voxels.dtype))


#------------- Codify -------------#
#----------------------------------#
def scale_transform(s):
    return np.diag([s, s, s, 1])


def trans_transform(t):
    t = t.reshape(3)
    tform = np.eye(4)
    tform[0:3,3] = t
    return tform


def codify_room_data(house_data, world2cam, obj_loader, use_shape_code=False):
    '''
    Coded form of objects.

    Args:
        house_data: nested dictionary corresponding to a single room
        world2cam: 4 X 4 transformation matrix
        obj_loader: pre-loading class to facilitate fast object lookup
    Returns:
        codes: list of (shape, transformation, scale, rot, trans) tuples
    '''
    n_obj = len(house_data['levels'][0]['nodes'])-1
    codes = []
    for nx in range(1,n_obj+1):
        obj_node = house_data['levels'][0]['nodes'][nx]
        model_name = obj_node['modelId']
        if(use_shape_code):
            shape = obj_loader.lookup(model_name, 'code')
        else:
            shape = obj_loader.lookup(model_name, 'voxels')
        vox_shift = trans_transform(np.ones((3,1))*0.5)
        if(obj_node.has_key('isMirrored') and obj_node['isMirrored'] == 1):
            vox_shift = np.matmul(vox_shift, np.diag([-1,1,1,1]))
        # print(model_name)
        # print(scale_transform(obj_loader.lookup(model_name, 'scale')))
        vox2obj = np.matmul(
            np.matmul(
                trans_transform(obj_loader.lookup(model_name, 'translation')),
                scale_transform(obj_loader.lookup(model_name, 'scale')[0, 0])
            ), vox_shift)
        obj2world = np.array(obj_node['transform']).reshape(4,4).transpose()
        obj2cam = np.matmul(world2cam, obj2world)
        vox2cam = np.matmul(obj2cam, vox2obj)

        rot_val, scale_val = np.linalg.qr(vox2cam[0:3, 0:3])
        scale_val = np.array([scale_val[0,0], scale_val[1,1], scale_val[2,2]])
        for d in range(3):
            if scale_val[d] < 0:
                scale_val[d] = -scale_val[d]
                rot_val[:,d] *= -1

        trans_val = vox2cam[0:3,3]
        rot_val = np.pad(rot_val, (0,1), 'constant')
        rot_val[3, 3] = 1
        code = (
            shape.astype(np.float32),
            vox2cam,
            scale_val,
            transformations.quaternion_from_matrix(rot_val, isprecise=True),
            trans_val
        )
        codes.append(code)

    return codes

def copy_code(code):
    return tuple([np.copy(c) for c in code])

def extract_proposal_codes(
    codes_gt, bboxes_gt, bboxes_proposals, max_proposals,
    add_gt_boxes=True, pos_thresh=0.7, neg_thresh=0.3, pos_ratio=0.25):

    # initialize counters and arrays
    ctr, n_pos, n_neg = 0, 0, 0
    codes = []
    labels = np.zeros((max_proposals))
    bboxes = np.zeros((max_proposals, 4))

    # Add gt boxes, compute positive and negative inds
    all_inds = np.array(range(bboxes_proposals.shape[0]))
    if len(codes_gt) > 0:
        if add_gt_boxes:
            for gx in range(len(bboxes_gt)):
                if gx < max_proposals:
                    codes.append(copy_code(codes_gt[gx]))
                    bboxes[ctr, :] = np.copy(bboxes_gt[gx, :])
                    labels[ctr] = 1
                    ctr += 1

        # Compute positive and negative indices
        ious = bbox_utils.bbox_overlaps(bboxes_gt.astype(np.float), bboxes_proposals.astype(np.float))
        max_ious = np.amax(ious, axis=0)
        gt_inds = np.argmax(ious, axis=0)
        pos_inds = all_inds[max_ious >= pos_thresh]
        neg_inds = all_inds[max_ious < neg_thresh]
    else:
        pos_inds = np.array([])
        neg_inds = all_inds
    
    # Add positive proposals
    pos_perm = np.random.permutation(pos_inds.size)
    while (n_pos < pos_inds.size) and (ctr < pos_ratio*max_proposals):
        px = pos_inds[pos_perm[n_pos]]
        gx = gt_inds[px]
        codes.append(copy_code(codes_gt[gx]))
        bboxes[ctr, :] = np.copy(bboxes_proposals[px, :])
        labels[ctr] = 1
        ctr += 1
        n_pos += 1

    # Add negative proposals
    neg_perm = np.random.permutation(neg_inds.size)
    while (n_neg < neg_inds.size) and (ctr < max_proposals):
        px = neg_inds[neg_perm[n_neg]]
        bboxes[ctr, :] = np.copy(bboxes_proposals[px, :])
        labels[ctr] = 0
        ctr += 1
        n_neg += 1

    bboxes = bboxes[0:ctr, :]
    labels = labels[0:ctr]
    # print(len(codes_gt), n_pos, n_neg)
    return codes, bboxes, labels


#---------- Object Loader ---------#
#----------------------------------#
class ObjectLoader:
    '''Pre-loading class to facilitate object lookup'''
    def __init__(self, object_dir):
        self._object_dir = object_dir
        object_names = [f for f in os.listdir(object_dir)]
        list.sort(object_names)
        self._object_names = object_names
        self._curr_obj_id = None
        self._preloaded_data = {}

    def lookup(self, obj_id, field):
        if obj_id != self._curr_obj_id:
            self._curr_obj_id = obj_id
            if obj_id in self._preloaded_data.keys():
                self._curr_obj_data = self._preloaded_data[obj_id]
            else:
                self._curr_obj_data = sio.loadmat(osp.join(
                    self._object_dir, obj_id, obj_id + '.mat'))
        return copy.copy(self._curr_obj_data[field])

    def preload(self):
        for ox in range(len(self._object_names)):
            obj_id = self._object_names[ox]
            obj_data = sio.loadmat(osp.join(
                self._object_dir, obj_id, obj_id + '.mat'))
            self._preloaded_data[obj_id] = obj_data


class MetaLoader:
    '''Pre-loading class for object metadata'''
    def __init__(self, csv_file):
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            self._metadata = []
            for row in reader:
                self._metadata.append(row)
        self._id_dict = {}
        for mx in range(len(self._metadata)):
            self._id_dict[self._metadata[mx]['model_id']] = mx

    def lookup(self, obj_id, field='nyuv2_40class'):
        if obj_id not in self._id_dict:
            return None
        mx = self._id_dict[obj_id]
        return self._metadata[mx][field]


class ObjectRetriever:
    '''Class for nearest neighbor lookup'''
    def __init__(self, obj_loader, encoder, gpu_id=0, downsampler=None, meta_loader=None):
        self.enc_shapes = []
        self.orig_shapes = []
        for model_name in obj_loader._object_names:
            if (meta_loader is not None) :
                object_class = meta_loader.lookup(model_name, field='nyuv2_40class')
                if object_class not in valid_object_classes:
                    continue

            shape = obj_loader.lookup(model_name, 'voxels').astype(np.float32)
            shape = torch.from_numpy(shape).unsqueeze(0).unsqueeze(0)
            shape_var = torch.autograd.Variable(shape.cuda(device=gpu_id), requires_grad=False)
            if downsampler is not None:
                shape_var = downsampler.forward(shape_var)
            shape = shape_var.data.cpu()[0]
            self.orig_shapes.append(shape)

            shape_enc = encoder.forward(shape_var)
            shape_enc = shape_enc.data.cpu()[0]
            self.enc_shapes.append(shape_enc)

        self.enc_shapes = torch.stack(self.enc_shapes)
        self.nz_shape = self.enc_shapes.shape[1]

    def retrieve(self, code):
        code = code.view([1, self.nz_shape])
        dists = (self.enc_shapes - code).norm(p=2, dim=1)
        _, inds = torch.min(dists, 0)
        return self.orig_shapes[inds[0]].clone()


#-------- Train/Val splits --------#
#----------------------------------#
def get_split(save_dir, house_names=None, train_ratio=0.75, val_ratio=0.1):
    ''' Loads saved splits if they exist. Otherwise creates a new split.

    Args:
        save_dir: Absolute path of where splits should be saved
    '''
    split_file = osp.join(save_dir, 'suncg_split.pkl')
    if os.path.isfile(split_file):
        return pickle.load(open(split_file, 'rb'))
    else:
        list.sort(house_names)
        house_names = np.ndarray.tolist(np.random.permutation(house_names))
        n_houses = len(house_names)
        n_train = int(n_houses*train_ratio)
        n_val = int(n_houses*val_ratio)
        splits = {
            'train': house_names[0:n_train],
            'val': house_names[n_train:(n_train+n_val)],
            'test': house_names[(n_train+n_val):]
        }
        pickle.dump(splits, open(split_file, 'wb'))
        return splits

#---------- Torch Utils -----------#
#----------------------------------#
def bboxes_to_rois(bboxes):
    ''' Concatenate boxes and associate batch index.
    '''
    rois = torch.cat(bboxes, 0).type(torch.FloatTensor)
    if rois.numel() == 0:
        return torch.zeros(0, 5).type(torch.FloatTensor)
    batch_inds = torch.ones((rois.size(0), 1)).type(torch.FloatTensor)
    ctr = 0
    for bx, boxes in enumerate(bboxes):
        nb = 0
        if boxes.numel() > 0:
            nb = boxes.size(0)
            batch_inds[ctr:(ctr+nb)] *= bx
            ctr += nb
    rois = torch.cat([batch_inds, rois], 1)
    return rois.type(torch.FloatTensor)


def collate_codes_instance(codes_b):
    ''' [(shape_i, tmat_i, scale_i, quat_i, trans_i)] => (shapes, scales, quats, trans)
    '''
    if len(codes_b) == 0:
        return None
    codes_out_b = []
    select_inds = [0,2,3,4] #won't output tmats
    for sx in select_inds:
        codes_out_b.append(torch.stack([code[sx] for code in codes_b], dim=0))
    return codes_out_b


def uncollate_codes_instance(code_tensors_b):
    '''
    (shapes, scales, quats, trans) => [(shape_i, tmat_i, scale_i, quat_i, trans_i)]
    '''
    codes_b = []
    if code_tensors_b[0].numel() == 0:
        return codes_b
    for cx in range(code_tensors_b[0].size(0)):
        code = []
        for t in code_tensors_b:
            code.append(t[cx].squeeze().numpy())
        codes_b.append(tuple(code))
    return codes_b


def collate_codes(codes):
    codes_instance = []
    codes_out = []
    for b in range(len(codes)):
        codes_b = collate_codes_instance(codes[b])
        if codes_b is not None:
            codes_instance.append(codes_b)
    for px in range(4):
        codes_out.append(torch.cat([code[px] for code in codes_instance]).type(torch.FloatTensor))
    return codes_out


def uncollate_codes(code_tensors, batch_size, batch_inds):
    '''
    Assumes batch inds are 0 indexed, increasing
    '''
    start_ind = 0
    codes = []
    for tx in range(len(code_tensors)):
        code_tensors[tx] = code_tensors[tx].data.cpu()

    for b in range(batch_size):
        codes_b = []
        nelems = torch.eq(batch_inds, b).sum()
        if nelems > 0:
            code_tensors_b = []
            for t in code_tensors:
                code_tensors_b.append(t[start_ind:(start_ind+nelems)])
            codes_b = uncollate_codes_instance(code_tensors_b)
            start_ind += nelems
        codes.append(codes_b)
    return codes


def quats_to_bininds(quats, medoids):
    '''
    Finds the closest bin for each quaternion.
    
    Args:
        quats: N X 4 tensor
        medoids: n_bins X 4  tensor
    Returns:
        bin_inds: N tensor with values in [0, n_bins-1]
    '''
    medoids = medoids.transpose(1,0)
    prod = torch.matmul(quats, medoids).abs()
    _, bin_inds = torch.max(prod, 1)
    return bin_inds


def bininds_to_quats(bin_inds, medoids):
    '''
    Select thee corresponding quaternion.
    
    Args:
        bin_inds: N tensor with values in [0, n_bins-1]
        medoids: n_bins X 4  tensor
    Returns:
        quats: N X 4 tensor
    '''
    n_bins = medoids.size(0)
    n_quats = bin_inds.size(0)
    bin_inds = bin_inds.view(-1, 1)
    inds_one_hot = torch.zeros(n_quats, n_bins)
    inds_one_hot.scatter_(1, bin_inds, 1)
    return torch.matmul(inds_one_hot, medoids)
