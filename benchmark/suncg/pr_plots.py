import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import os.path as osp
import platform

eval_set = 'val'
net_name = 'dwr_shape_ft'

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', '..', 'cachedir')
plots_dir = os.path.join(cache_path, 'evaluation', 'icp', eval_set, 'plots')

def subplots(plt, Y_X, sz_y_sz_x=(10,10)):
  Y,X = Y_X
  sz_y,sz_x = sz_y_sz_x
  plt.rcParams['figure.figsize'] = (X*sz_x, Y*sz_y)
  fig, axes = plt.subplots(Y, X)
  plt.subplots_adjust(wspace=0.1, hspace=0.1)
  return fig, axes

def pr_plots(net_name, iter_number, set_number):
  dir_name = os.path.join(cache_path, 'evaluation', 'dwr')
  json_file = os.path.join(dir_name, set_number, net_name, 'eval_set{}_0.json'.format(set_number))

  with open(json_file, 'rt') as f:
    a = json.load(f)
  imset = a['eval_params']['set'].title()

  plot_file = os.path.join(dir_name, set_number, net_name, 'eval_set{}_0_back.pdf'.format(set_number))
  print('Saving plot to {}'.format(osp.abspath(plot_file)))
  # Plot 1 with AP for all, and minus other things one at a time.
  #with sns.axes_style("darkgrid"):
  with plt.style.context('fivethirtyeight'):
    fig, axes = subplots(plt, (1,1), (7,7))
    ax = axes
    legs = []
    i_order = [0, 1, 2, 3, 5, 4]
    # for i in np.arange(6, 12):
    for jx in range(6):
      i = i_order[jx]
      prec = np.array(a['bench_summary'][i]['prec'])
      rec = np.array(a['bench_summary'][i]['rec'])
      if i == 0:
        ax.plot(rec, prec, '-')
        legs.append('{:4.1f} {:s}'.format(100*a['bench_summary'][i]['ap'], a['eval_params']['ap_str'][i]))
      else:
        ax.plot(rec, prec, '--')
        legs.append('{:4.1f}   {:s}'.format(100*a['bench_summary'][i]['ap'], a['eval_params']['ap_str'][i]))
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1]);
    ax.set_xlabel('Recall', fontsize=20)
    ax.set_ylabel('Precision', fontsize=20)
    ax.set_title('Precision Recall Plots on {:s} Set'.format(imset), fontsize=20)

    l = ax.legend(legs, fontsize=18, bbox_to_anchor=(0,0), loc='lower left', framealpha=0.5, frameon=True)

    ax.plot([0,1], [0,0], 'k-')
    ax.plot([0,0], [0,1], 'k-')
    plt.tick_params(axis='both', which='major', labelsize=20)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close(fig)

  plot_file = os.path.join(dir_name, set_number, net_name, 'eval_set{}_0_frwd.pdf'.format(set_number))
  print('Saving plot to {}'.format(osp.abspath(plot_file)))

  with plt.style.context('fivethirtyeight'):
    fig, axes = subplots(plt, (1,1), (7,7))
    ax = axes
    legs = []
    i_order = [6, 9, 7, 8, 10, 11]
    # for i in np.arange(6, 12):
    for jx in range(6):
      i = i_order[jx]
      prec = np.array(a['bench_summary'][i]['prec'])
      rec = np.array(a['bench_summary'][i]['rec'])
      if i == 6:
        ax.plot(rec, prec, '-')
        legs.append('{:4.1f} {:s}'.format(100*a['bench_summary'][i]['ap'], a['eval_params']['ap_str'][i]))
      else:
        ax.plot(rec, prec, '--')
        str_ = '+'+'+'.join(a['eval_params']['ap_str'][i].split('+')[1:])
        legs.append('{:4.1f}   {:s}'.format(100*a['bench_summary'][i]['ap'], str_))
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1]);
    ax.set_xlabel('Recall', fontsize=20)
    ax.set_ylabel('Precision', fontsize=20)
    ax.set_title('Precision Recall Plots on {:s} Set'.format(imset), fontsize=20)

    l = ax.legend(legs, fontsize=18, bbox_to_anchor=(0,0), loc='lower left', framealpha=0.5, frameon=True)
    ax.plot([0,1], [0,0], 'k-')
    ax.plot([0,0], [0,1], 'k-')
    plt.tick_params(axis='both', which='major', labelsize=20)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
  pr_plots(net_name, 0, eval_set)
