import os
import sys
import json

import torch
import numpy as np
import pyredner as pyr
from ..util.tensops import cpu, gpu, cpui, gpui
from ..util.meshops import recompute_normals
import matplotlib
import matplotlib.pyplot as plt

mydir = os.path.dirname(os.path.abspath(__file__))

def load_settings(mesh_name):
    '''
    Read, parse and return (cameras, lights) for the given mesh.
    '''
    dpath = os.path.join(mydir, mesh_name)
    fpath = os.path.join(dpath, "camera.json")
    with open(fpath) as f:
        camera_settings = json.load(f)
 
    cameras = [
        pyr.Camera(
            position=cpu(camera_pos),
            look_at=cpu(camera_settings['look_at']),
            up=cpu(camera_settings['up']),
            fov=cpu(camera_settings['fov']),
            resolution=camera_settings['resolution'],
            camera_type=pyr.camera_type.perspective
        )
        for camera_pos in camera_settings['positions']
    ]

    def parse_one_light(dict_light):
        if dict_light['type'] == 'directional':
            return pyr.DirectionalLight(
                direction = gpu(dict_light['direction']),
                intensity = gpu(dict_light['intensity'])
            )
        else:
            return None
    
    lights = [
        parse_one_light(d) for d in camera_settings['lights']
    ]
    lights = [l for l in lights if l is not None]

    return cameras, lights
    
def make_scenes(mesh, settings):
    '''
    Make redner scenes with the given mesh, and the (camera, lights) settings
    '''
    cameras, lights = settings
    if not isinstance(mesh, list):
        mesh = [mesh]
    scenes = [pyr.Scene(camera=c, objects=mesh) for c in cameras]
    return scenes

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


def load_3d(mesh_name):
    '''
    Loads a 3D model, computing vertex normals as needed
    '''
    dpath = os.path.join(mydir, mesh_name)
    fpath = os.path.join(dpath, "mesh.obj")
    if os.path.isfile(fpath):
        obj = pyr.load_obj(fpath, return_objects=True)[0]
        recompute_normals(obj)
        texpath = os.path.join(dpath, "texture.png")
        if os.path.isfile(texpath):
            tex_img = pyr.imread(texpath)
            obj.material.diffuse_reflectance = pyr.Texture(tex_img)
        return obj
    else:
        raise FileNotFoundError(f"Could not find {mesh_name}.obj")
    
def save_3d(mesh, mesh_name, allow_overwrite = True):
    '''
    Saves a 3D model
    '''
    dpath = os.path.join(mydir, mesh_name)
    fpath = os.path.join(dpath, "mesh.obj")
    os.makedirs(dpath, exist_ok=True)
    if os.path.isfile(fpath) and not allow_overwrite:
        raise FileExistsError(fpath)
    else:
        pyr.save_obj(mesh, fpath)

def get_known_meshes():
    return [
        dname for dname in os.listdir(mydir) 
        if (os.path.isdir(os.path.join(mydir, dname))
        and os.path.isfile(os.path.join(mydir, dname, "mesh.obj")))]

    