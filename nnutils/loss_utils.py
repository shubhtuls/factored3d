'''
Loss building blocks.
'''
import torch
import torch.nn as nn
import math
import gflags as flags

#-------------- flags -------------#
#----------------------------------#
flags.DEFINE_float('shape_loss_wt', 1, 'Shape loss weight.')
flags.DEFINE_float('scale_loss_wt', 1, 'Scale loss weight.')
flags.DEFINE_float('quat_loss_wt', 1, 'Quat loss weight.')
flags.DEFINE_float('trans_loss_wt', 1, 'Trans loss weight.')


def quat_loss(q1, q2):
    '''
    Anti-podal squared L2 loss.
    
    Args:
        q1: N X 4
        q2: N X 4
    Returns:
        loss : scalar
    '''
    q_diff_loss = (q1-q2).pow(2).sum(1)
    q_sum_loss = (q1+q2).pow(2).sum(1)
    q_loss, _ = torch.stack((q_diff_loss, q_sum_loss), dim=1).min(1)
    return q_loss.mean()


def code_loss(
    code_pred, code_gt,
    pred_voxels=True, classify_rot=True,
    shape_wt=1.0, scale_wt=1.0, quat_wt=1.0, trans_wt=1.0):
    '''
    Code loss

    Args:
        code_pred: [shape, scale, quat, trans]
        code_gt: [shape, scale, quat, trans]
    Returns:
        total_loss : scalar
    '''
    if pred_voxels:
        s_loss = torch.nn.functional.binary_cross_entropy(code_pred[0], code_gt[0])
    else:
        #print('Shape gt/pred mean : {}, {}'.format(code_pred[0].mean().data[0], code_gt[0].mean().data[0]))
        s_loss = (code_pred[0] - code_gt[0]).pow(2).mean()

    if classify_rot:
        q_loss = torch.nn.functional.nll_loss(code_pred[2], code_gt[2])
    else:
        q_loss = quat_loss(code_pred[2], code_gt[2])

    sc_loss = (code_pred[1].log() - code_gt[1].log()).abs().mean()
    tr_loss = (code_pred[3] - code_gt[3]).pow(2).mean()

    total_loss = sc_loss*scale_wt
    total_loss += q_loss*quat_wt
    total_loss += tr_loss*trans_wt
    total_loss += s_loss*shape_wt
    
    loss_factors = {
        'shape': s_loss*shape_wt, 'scale': sc_loss*scale_wt, 'quat': q_loss*quat_wt, 'trans': tr_loss*trans_wt
    }
    return total_loss, loss_factors
