"""Script for making html from a directory.
"""
# Sample usage:
# (box3d_shape_ft) python make_html.py --imgs_root_dir='/data0/shubhtuls/code/oc3d/cachedir/results_vis/box3d/val/box3d_shape_ft' --html_name=box3d_shape_ft --html_dir='/data0/shubhtuls/code/oc3d/cachedir/results_vis/pages/'

# (dwr_shape_ft) python make_html.py --imgs_root_dir='/data0/shubhtuls/code/oc3d/cachedir/results_vis/dwr/val/dwr_shape_ft' --html_name=dwr_shape_ft --html_dir='/data0/shubhtuls/code/oc3d/cachedir/results_vis/pages/'

# (depth_baseline) python make_html.py --imgs_root_dir='/data0/shubhtuls/code/oc3d/cachedir/results_vis/depth_baseline' --html_name=depth_baseline --html_dir='/data0/shubhtuls/code/oc3d/cachedir/results_vis/pages/'

# (voxels_baseline) python make_html.py --imgs_root_dir='/data0/shubhtuls/code/oc3d/cachedir/results_vis/voxels_baseline' --html_name=voxels_baseline --html_dir='/data0/shubhtuls/code/oc3d/cachedir/results_vis/pages/'

# (nyu) python make_html.py --imgs_root_dir='/data0/shubhtuls/code/oc3d/cachedir/results_vis/nyu/test/dwr_shape_ft' --html_name=nyu_dwr_shape_ft --html_dir='/data0/shubhtuls/code/oc3d/cachedir/results_vis/pages/'

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from google.apputils import app
import gflags as flags
import os
import os.path as osp
from yattag import Doc
from yattag import indent
import numpy as np

flags.DEFINE_string('imgs_root_dir', '', 'Directory where renderings are saved')
flags.DEFINE_string('html_name', '', 'Name of webpage')
flags.DEFINE_string('html_dir', '', 'Directory where output should be saved')

def main(_):
    opts = flags.FLAGS
    vis_dir_names = os.listdir(opts.imgs_root_dir)
    vis_dir_names.sort()
    img_keys = os.listdir(osp.join(opts.imgs_root_dir, vis_dir_names[0]))
    img_keys.sort()
    img_root_rel_path = osp.relpath(opts.imgs_root_dir, opts.html_dir)
    if not os.path.exists(opts.html_dir):
        os.makedirs(opts.html_dir)
    html_file = osp.join(opts.html_dir, opts.html_name + '.html')
    ctr = 0

    doc, tag, text = Doc().tagtext()
    with tag('html'):
        with tag('body'):
            with tag('table', style = 'width:100%', border="1"):
                with tag('tr'):
                    for img_name in img_keys:
                        with tag('td'):
                            text(img_name)

                for img_dir in vis_dir_names:
                    with tag('tr'):
                        for img_name in img_keys:
                            with tag('td'):
                                with tag('img', width="640px", src=osp.join(img_root_rel_path, img_dir, img_name)):
                                    ctr += 1

    r1 = doc.getvalue()
    r2 = indent(r1)

    with open(html_file, 'wt') as f:
        f.write(r2)
    

if __name__ == '__main__':
    app.run()
