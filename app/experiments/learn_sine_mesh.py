import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.nn as nn
import itertools
import app
import pyredner as pyr
from tqdm import tqdm

from ..util.tensops import gpu, gpui, cpu, cpui
from ..data import load_3d, load_settings, make_scenes
from ..loss.pyramid import ImagePyramidLoss
from ..loss.laplacian import LaplacianLoss
from ..loss.combined import CombinedLoss
from ..reconstruction.learner import Learner
from ..util.render import render, show_images, save_images, compare_images, extend_sequence, start_sequence, end_sequence, torch_to_np_image, show_np_image, plot_loss
from ..util.profiling import get_vram_usage
from ..util.imageops import compute_average_color
import cv2

#    pyr.set_device(torch.device('cpu'))
pyr.set_print_timing(False)

def image_loss_only(initial_lr = 0.01, 
                    lr_mult = 0.8,
                    lr_epochs_for_mult = 50,
                    lr_mult_until_epoch = -1,
                    n_iterations = 20,
                    n_levels = 1,
                    img_output = "flat_to_sine_pyramid_only_images.mp4",
                    mesh_output = "learned_sine",
                    loss_output = "flat_to_sine_pyramid_only_losses.png"):
    # Go from flat -> sine
    learner = Learner("flat")
    with torch.no_grad():
        _, orig_images = learner.generate()
    
    # 1 pyramid level = just the image
    loss_fn = (CombinedLoss("sine")
                .add_2d('pyramid', 1000.0, ImagePyramidLoss(n_levels))).to(pyr.get_device())
    gt_images = loss_fn.get_targets()
    
    # Learn the mesh
    learner.free_mesh()

    # Setup optimizer and scheduler
    parameters = learner.parameters()
    optimizer = torch.optim.Adam(parameters, lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lambda e_n: lr_mult if (e_n % lr_epochs_for_mult == 0 and e_n > 0 and e_n < lr_mult_until_epoch) else 1.0, last_epoch=-1)

    # Only 1 loss
    loss_fn.weight_2d('pyramid', 1.0)    

    loss_history = []
    image_hist = []

    img_seq = None

    for iternum in tqdm(range(n_iterations)):
        # Zero all gradients
        # Make the scenes
        optimizer.zero_grad()
        out_3d, out_2d = learner.generate()
        total_loss = loss_fn(out_3d, out_2d)
        loss_history.append(loss_fn.get_last_breakdown())
        total_loss.backward()
        learner.clean()
        optimizer.step()
        scheduler.step()
        if iternum % 5 == 0 or iternum == n_iterations - 1 or iternum == 0:
            # Draw it
            comparison = torch_to_np_image(compare_images(orig_images, gt_images, out_2d))
            if img_seq is None:
                img_seq = start_sequence(img_output, comparison)
            else:
                extend_sequence(img_seq, comparison)
        if iternum % 10 == 0 or iternum == n_iterations - 1 or iternum == 0:
            print(loss_history[-1])
    with torch.no_grad():
        _, out_2d = learner.generate()

    end_sequence(img_seq)
    show_np_image(comparison)
    learner.save(mesh_output, allow_overwrite=True)

    plot_loss(loss_history, loss_output)
    return loss_history

def image_and_smooth_loss(initial_lr = 0.01, 
                    lr_mult = 0.8,
                    lr_epochs_for_mult = 50,
                    lr_mult_until_epoch = -1,
                    n_iterations = 20,
                    n_levels = 1,
                    pyramid_scale = 1000.0,
                    pyramid_weight = 0.6,
                    smooth_scale = 6.0,
                    smooth_weight = 0.0,
                    edge_length_scale = None,
                    scale_by_orig = False,
                    from_mesh="flat",
                    to_mesh="sine",
                    img_output = "flat_to_sine_pyramid_and_smooth_images.mp4",
                    mesh_output = "learned_sine_smoothed",
                    loss_output = "flat_to_sine_pyramid_and_smooth_losses.png",
                    show_intermediates = True):
    # Go from flat -> sine
    learner = Learner(from_mesh)
    with torch.no_grad():
        _, orig_images = learner.generate()
    
    # 1 pyramid level = just the image
    loss_fn = CombinedLoss(to_mesh)
    if pyramid_weight > 0 and pyramid_scale > 0:
        loss_fn.add_2d('pyramid', pyramid_scale, ImagePyramidLoss(n_levels))
    if smooth_weight > 0 and smooth_scale > 0:
        loss_fn.add_3d('smoothness', smooth_scale, LaplacianLoss(edge_length_scale, scale_by_orig))
    loss_fn = loss_fn.to(pyr.get_device())
    gt_images = loss_fn.get_targets()
    
    # Learn the mesh
    learner.free_mesh()

    # Setup optimizer and scheduler
    parameters = learner.parameters()
    optimizer = torch.optim.Adam(parameters, lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lambda e_n: lr_mult if (e_n % lr_epochs_for_mult == 0 and e_n > 0 and e_n < lr_mult_until_epoch) else 1.0, last_epoch=-1)

    # Only 1 loss
    loss_fn.weight_2d('pyramid', pyramid_weight)
    loss_fn.weight_3d('smoothness', smooth_weight)

    loss_history = []
    image_hist = []

    img_seq = None

    for iternum in tqdm(range(n_iterations)):
        # Zero all gradients
        # Make the scenes
        optimizer.zero_grad()
        out_3d, out_2d = learner.generate()
        total_loss = loss_fn(out_3d, out_2d)
        loss_history.append(loss_fn.get_last_breakdown())
        total_loss.backward()
        learner.clean()
        optimizer.step()
        scheduler.step()
        if iternum % 5 == 0 or iternum == n_iterations - 1 or iternum == 0:
            # Draw it
            comparison = torch_to_np_image(compare_images(orig_images, gt_images, out_2d))
            if img_seq is None:
                img_seq = start_sequence(img_output, comparison)
            else:
                extend_sequence(img_seq, comparison)
            if show_intermediates:
                cv2.imshow('compare', comparison)
        if iternum % 10 == 0 or iternum == n_iterations - 1 or iternum == 0:
            print(loss_history[-1])
        
        if show_intermediates:
            if cv2.waitKey(16) & 0xFF == ord('q'):
                break

    with torch.no_grad():
        _, out_2d = learner.generate()

    end_sequence(img_seq)
    if show_intermediates:
        cv2.destroyAllWindows()
    show_np_image(comparison)
    learner.save(mesh_output, allow_overwrite=True)


    plot_loss(loss_history, loss_output)
    return loss_history