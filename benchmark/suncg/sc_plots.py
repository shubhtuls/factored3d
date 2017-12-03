from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import json
import os
import os.path as osp
import scipy.io

eval_set = 'val'
netName = 'dwr_shape_ft'

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', '..', 'cachedir')
plots_dir = os.path.join(cache_path, 'evaluation', 'icp', eval_set, 'plots')

if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

def subplots(plt, Y_X, sz_y_sz_x=(10,10)):
    Y,X = Y_X
    sz_y,sz_x = sz_y_sz_x
    plt.rcParams['figure.figsize'] = (X*sz_x, Y*sz_y)
    fig, axes = plt.subplots(Y, X)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    return fig, axes

def toNpArray(matVar):
    out = np.zeros(len(matVar))
    for i in range(len(matVar)):
        out[i] = matVar[i][0]
    return out

def plotExperiment(expName, errors, representationNames, xLeg, varName, maxRange=1):
    with plt.style.context('fivethirtyeight'):
        fig, axes = subplots(plt, (1,1), (6,6))
        ax = axes

        legs = []
        for i in range(len(representationNames)):
            repName = representationNames[i]
            perf = np.sort(errors[varName][i, :])
            perf = perf[~np.isnan(perf)]
            perf = perf[perf < 1e6]
            medVal = np.median(perf)
            percentile = np.linspace(0,1,np.size(perf,0))
            ax.plot(percentile, perf, '-')
            legs.append('{:s}'.format(repName))
        ax.set_ylim([0, maxRange]); ax.set_xlim([0, 1]);
        ax.set_ylabel(xLeg, fontsize=20)
        ax.set_xlabel('Fraction of Data', fontsize=20)
        ax.set_title(expName, fontsize=20)

        l = ax.legend(legs, title="Scene Representations:", fontsize=14, bbox_to_anchor=(0,1), loc='upper left', framealpha=0.5, frameon=True)

        ax.plot([0,0], [0,maxRange], 'k-')
        ax.plot([0,1], [0,0], 'k-')
        plt.tick_params(axis='both', which='major', labelsize=20)
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plot_file = os.path.join(plots_dir, varName + '.pdf')
        plt.savefig(plot_file, bbox_inches='tight')
        plt.close(fig)
        
resultsDir = os.path.join(cache_path, 'evaluation', 'icp', eval_set, netName)
matFile = os.path.join(resultsDir, 'results.mat')
results = scipy.io.loadmat(matFile)

########################################
##############  Objects  ###############
expName = 'Object Representation Ability'
representationNames = ['Factored (ours)', 'Depth', 'Voxels']
xLeg = 'Scale-normalized Mean Squared Error'
varName = 'object_eval'
plotExperiment(expName, results, representationNames, xLeg, varName, maxRange=1e-2)

########################################
###############  Depth  ################
expName = 'Depth Representation Ability'
representationNames = ['Factored (ours)', 'Depth', 'Voxels']
xLeg = 'Mean Squared Error (in $m^2$)'
varName = 'depth_eval'
plotExperiment(expName, results, representationNames, xLeg, varName, maxRange=8e-1)

########################################
##############  Voxels  ################
expName = 'Volume Representation Ability'
representationNames = ['Factored (ours)', 'Depth', 'Voxels']
xLeg = 'IoU (Higher is better)'
varName = 'volume_overlap_eval'
plotExperiment(expName, results, representationNames, xLeg, varName, maxRange=1)

########################################
##########  Visible Layout  ############
expName = 'Visible Layout Representation Ability'
representationNames = ['Factored (ours)', 'Depth', 'Voxels']
xLeg = 'Mean Squared Error (in $m^2$)'
varName = 'layout_eval'
plotExperiment(expName, results, representationNames, xLeg, varName, maxRange=5e-1)

########################################
##########  Amodal Layout  #############
expName = 'Amodal Layout Representation Ability'
representationNames = ['Factored (ours)', 'Depth', 'Voxels']
xLeg = 'Mean Squared Error (in $m^2$)'
varName = 'layout_amodal_eval'
plotExperiment(expName, results, representationNames, xLeg, varName, maxRange=8e-1)

print('Plots saved in {}'.format(osp.abspath(plots_dir)))