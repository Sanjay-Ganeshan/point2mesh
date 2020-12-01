import torch
import torch.nn as nn
import numpy as np
import pyredner as pyr
import matplotlib.pyplot as plt
import cv2
import ffmpeg

def render(scenes, settings, grad = None, alpha=True):
    cameras, lights = settings
    if grad is True:
        with torch.no_grad():
            imgs = pyr.render_deferred(scene=scenes, lights=lights, alpha=alpha)
    elif grad is False:
        with torch.enable_grad():
            imgs = pyr.render_deferred(scene=scenes, lights=lights, alpha=alpha)
    else:
        # Don't affect grad
        imgs = pyr.render_deferred(scene=scenes, lights=lights, alpha=alpha)
    return imgs

def show_images(imgs):
    if isinstance(imgs, list) or len(imgs.shape) == 4:
        # list of images
        for im in imgs:
            show_images(im)
    else:
        # 1 image
        # Gamma correct and convert to cpu tensor
        imgrgb = imgs[:,:,:3]
        imga = imgs[:,:,3:]
        
        # Linear->Gamma
        gammargb = torch.pow(imgrgb, 1.0 / 2.2)
        
        # cat RGB and A to make RGBA
        finalimg = torch.cat([gammargb, imga], dim=2)
        plt.imshow(finalimg.cpu().detach().numpy())
        plt.show()

def start_sequence(fn, image):
    if len(image.shape) == 2:
        h, w = image.shape
        c = 1
    else:
        h,w,c = image.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(fn, fourcc, 4, (w,h), c > 1)
    
    extend_sequence(writer, image)
    return writer

def extend_sequence(seq, image):
    if seq is not None:
        seq.write(image[:,:,:3])

def end_sequence(seq):
    if seq is not None:
        seq.release()

def torch_to_np_image(img):
    # Image is hwc, RGBA
    # We want it gamma encoded & composited on white
    imgrgb = img[:,:,:3]
    imga = img[:,:,3:]
    
    # Linear->Gamma
    gammargb = torch.pow(imgrgb, 1.0 / 2.2)
    
    # cat RGB and A to make RGBA
    finalimg = torch.cat([gammargb, imga], dim=2)
    np_rgba = finalimg.numpy()

    np_rgb = np_rgba[:,:,:3]
    np_a = np_rgba[:,:,3:]
    bg = np.ones_like(np_rgb)
    weighted_bg = bg * (1.0 - np_a)
    weighted_fg = np_rgb * np_a
    composited = weighted_bg + weighted_fg
    # This is between 0 and 1. Let's make it between 0..255
    quantized = np.clip(composited * 255, 0, 255).astype(np.uint8)
    return quantized

def show_np_image(img):
    plt.imshow(img)
    plt.show()

def compare_images(begin, gt, predicted):
    if not isinstance(begin, list) and len(begin.shape) == 3:
        # Single image. Unsqueeze it
        begin = [begin]
        gt = [gt]
        predicted = [predicted]

    if not isinstance(begin, list):
        assert len(begin.shape) == 4, "Expected BHWC"
        # Convert to 4d tensor
        begin = list(begin)
    
    if not isinstance(gt, list):
        assert len(gt.shape) == 4, "Expected BHWC"
        # Convert to 4d tensor
        gt = list(gt)
    
    if not isinstance(predicted, list):
        assert len(predicted.shape) == 4, "Expected BHWC"
        # Convert to 4d tensor
        predicted = list(predicted)
    
    # Cat different images vertically
    # [B]HWC ... should have same width, new height.
    begin_vcat = torch.cat(begin, dim=0)
    gt_vcat = torch.cat(gt, dim=0)
    predicted_vcat = torch.cat(predicted, dim=0)

    # Now, horizontally cat these three
    # They're all HWC...new width
    comparison = torch.cat([begin_vcat, gt_vcat, predicted_vcat], dim=1)

    # Now it's HWC!
    return comparison.detach().cpu()
        
def save_images(fns, imgs):
    if isinstance(imgs, list) or len(imgs.shape) == 4:
        # list of images
        for fn, im in zip(fns, imgs):
            save_images(fn, im)
    else:
        # 1 image
        # Gamma correct and convert to cpu tensor
        imgrgb = imgs[:,:,:3]
        imga = imgs[:,:,3:]
        
        # Linear->Gamma
        gammargb = torch.pow(imgrgb, 1.0 / 2.2)
        
        # cat RGB and A to make RGBA
        finalimg = torch.cat([gammargb, imga], dim=2)
        plt.imsave(fns, finalimg.cpu().detach().numpy())

def plot_loss(loss_history, save_to=None):
    # Loss history is a list of breakdowns
    x = list(range(len(loss_history)))
    keys = sorted(loss_history[0])
    y = [[loss_breakdown[k] for k in keys] for loss_breakdown in loss_history]
    legend = [k.replace("_total", "TOTAL") for k in keys]
    plt.plot(x, y)
    plt.title("Loss per epoch")
    plt.legend(legend)
    plt.show()
    if save_to is not None:
        plt.savefig(save_to)
    
