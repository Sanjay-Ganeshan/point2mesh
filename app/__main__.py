import argparse

import os

from .util.tensops import gpu, gpui, cpu, cpui
from .data import load_3d, load_settings, make_scenes
from .loss.pyramid import ImagePyramidLoss
from .loss.laplacian import LaplacianLoss
from .loss.combined import CombinedLoss
from .reconstruction.learner import Learner
from .util.render import render, show_images, save_images
from .util.profiling import get_vram_usage
import pyredner as pyr
import torch


mydir = os.path.dirname(os.path.abspath(__file__))
outputdir = os.path.join(mydir, "output")

def run_tests(args):
    mug1 = load_3d("mug1")
    decimate(mug1)
    decimate(mug1)
    save_3d(mug1, "mug1_decimated")

    # Try out blurring / image pyramid
    # Get some camera settings
    mug1_settings = load_settings('mug2')
    mug1_scenes = make_scenes(mug1, mug1_settings)
    rendered = render(mug1_scenes, mug1_settings, grad=False)
    blur = GaussianBlur(sigma=7).to(pyr.get_device())
    # blur the image
    blurry = blur(rendered)
    show_images(rendered)
    show_images(blurry)
    
def tex_test(args):
    pyr.set_device(torch.device('cpu'))
    mug_textured = load_3d("mug1")
    mug_settings = load_settings('mug1')
    mug_scenes = make_scenes(mug_textured, mug_settings)
    rendered = render(mug_scenes, mug_settings, grad=False)
    show_images(rendered)

    #    pyr.set_device(torch.device('cpu'))
    pyr.set_print_timing(False)

    learner = Learner("mine").free_mesh().free_tex()
    with torch.no_grad():
        _, orig_img = learner.generate()
        show_images(orig_img)


def sine_test(args):
    pyr.set_device(torch.device('cpu'))
    learner = Learner("sine")
    loss = CombinedLoss("flat")
    with torch.no_grad():
        _, orig_img = learner.generate()
        targets = loss.get_targets()
        show_images(orig_img)
        show_images(targets)

def experiment_sine_image_only(args):
    from .experiments.learn_sine_mesh import image_loss_only
    image_loss_only(n_iterations=200,
                    initial_lr=0.01,
                    lr_mult=0.6,
                    lr_epochs_for_mult=15,
                    img_output=os.path.join(outputdir, "flat_to_sine_images_imgonly.mp4"),
                    loss_output=os.path.join(outputdir, "flat_to_sine_loss_imgonly.png"),
                    mesh_output="learned_sine_imgonly")

def experiment_sine_pyramid_3(args):
    from .experiments.learn_sine_mesh import image_loss_only
    image_loss_only(n_iterations=200,
                    initial_lr=0.01,
                    lr_mult=0.2,
                    lr_epochs_for_mult=20,
                    n_levels=3,
                    img_output=os.path.join(outputdir, "flat_to_sine_images_pyramid3.mp4"),
                    mesh_output="learned_sine_pyramid3",
                    loss_output=os.path.join(outputdir, "flat_to_sine_loss_pyramid3.png"))

def experiment_sine_pyramid_3_aggressive_lr(args):
    from .experiments.learn_sine_mesh import image_loss_only
    image_loss_only(n_iterations=200,
                    initial_lr=0.01,
                    lr_mult=0.1,
                    lr_epochs_for_mult=15,
                    lr_mult_until_epoch=60,
                    n_levels=3,
                    img_output=os.path.join(outputdir, "flat_to_sine_images_pyramid3_aggresivelr.mp4"),
                    loss_output=os.path.join(outputdir, "flat_to_sine_loss_pyramid3_aggressivelr.png"),
                    mesh_output="learned_sine_pyramid3_aggressivelr")

def experiment_sine_image_only_and_smooth(args):
    from .experiments.learn_sine_mesh import image_and_smooth_loss
    image_and_smooth_loss(n_iterations=200,
                    initial_lr=0.01,
                    lr_mult=0.1,
                    lr_epochs_for_mult=15,
                    lr_mult_until_epoch=60,
                    n_levels=1,
                    pyramid_scale = 1000.0,
                    pyramid_weight = 0.6,
                    smooth_scale = 6.0,
                    smooth_weight = 0.0,
                    edge_length_scale = None,
                    scale_by_orig = False,
                    img_output=os.path.join(outputdir, "flat_to_sine_images_image_and_smooth.mp4"),
                    loss_output=os.path.join(outputdir, "flat_to_sine_loss_image_and_smooth.png"),
                    mesh_output="learned_sine_image_and_smooth")


def outp(fname):
    return os.path.join(outputdir, fname)

def experiment_mug_image_nosmooth(args):
    from .experiments.learn_sine_mesh import image_and_smooth_loss
    image_and_smooth_loss(n_iterations = 500,
                          initial_lr=0.005,
                          lr_mult=0.2,
                          lr_epochs_for_mult=100,
                          n_levels=1,
                          pyramid_scale=1000.0,
                          pyramid_weight=1.0,
                          smooth_scale=0.0,
                          smooth_weight=0.0,
                          edge_length_scale=None,
                          scale_by_orig=False,
                          from_mesh="mymugnotex",
                          to_mesh="targetmugnotex",
                          img_output=outp("mug_to_targetmug_notex_image_nosmooth.mp4"),
                          loss_output=outp("mug_to_targetmug_notex_image_nosmooth_loss.png"),
                          mesh_output="learned_mug_image_nosmooth"
                          )


def experiment_mug_image_smooth(args):
    from .experiments.learn_sine_mesh import image_and_smooth_loss
    image_and_smooth_loss(n_iterations = 500,
                          initial_lr=0.001,
                          lr_mult=0.2,
                          lr_epochs_for_mult=100,
                          n_levels=1,
                          pyramid_scale=1000.0,
                          pyramid_weight=0.6,
                          smooth_scale=6.0,
                          smooth_weight=0.4,
                          edge_length_scale=None,
                          scale_by_orig=False,
                          from_mesh="mymugnotex",
                          to_mesh="targetmugnotex",
                          img_output=outp("mug_to_targetmug_notex_image_nosmooth.mp4"),
                          loss_output=outp("mug_to_targetmug_notex_image_nosmooth_loss.png"),
                          mesh_output="learned_mug_image_nosmooth"
                          )

def experiment_pear_pyramid3_smooth(args):
    from .experiments.learn_sine_mesh import image_and_smooth_loss
    image_and_smooth_loss(n_iterations = 200,
                          initial_lr=0.005,
                          lr_mult=0.2,
                          lr_epochs_for_mult=30,
                          n_levels=3,
                          pyramid_scale=1000.0,
                          pyramid_weight=0.6,
                          smooth_scale=10.0,
                          smooth_weight=0.4,
                          edge_length_scale=None,
                          scale_by_orig=False,
                          from_mesh="sphere",
                          to_mesh="pear",
                          img_output=outp("sphere_to_pear_pyramid3_smooth_seq.mp4"),
                          loss_output=outp("sphere_to_pear_pyramid3_smooth_loss.png"),
                          mesh_output="sphere_to_pear_pyramid3_smooth_seq"
                          )

def parse_args():
    cmd_to_fn = {
        "test": run_tests,
        "tex": tex_test,
        "sine": sine_test,
        "exp1": experiment_sine_image_only,
        "exp2": experiment_sine_pyramid_3,
        "exp3": experiment_sine_pyramid_3_aggressive_lr,
        "exp4": experiment_sine_image_only_and_smooth,
        "exp5": experiment_mug_image_nosmooth,
        "exp6": experiment_mug_image_smooth,
        "exp7": experiment_pear_pyramid3_smooth
    }
    parser = argparse.ArgumentParser(description="Run various operations")
    parser.add_argument("cmd", choices=sorted(cmd_to_fn.keys()), help="Which command to run")
    args = parser.parse_args()
    args.fn = cmd_to_fn[args.cmd]
    return args

def main():
    args = parse_args()
    args.fn(args)

if __name__ == '__main__':
    main()

