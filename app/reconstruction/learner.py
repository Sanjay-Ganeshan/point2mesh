import torch
import numpy as np
import math
import torch.nn as nn
import itertools
import pyredner as pyr

from ..data import load_settings, load_3d, make_scenes, save_3d
from ..util.render import render
from ..util.tensops import cpu, cpui, gpu, gpui
from ..util.meshops import recompute_normals

class Learner(object):
    def __init__(self, obj_name):
        # First, let's load the object
        self.mesh = load_3d(obj_name)
        self.settings = load_settings(obj_name)
        self.scenes = make_scenes(self.mesh, self.settings)
        self.params = []

        self.learning_vertices = False
        
        self.learning_texture = False
        self.learned_tex = None
    
    def generate(self):
        out_3d = self.mesh
        out_2d = render(self.scenes, self.settings, grad=None, alpha=True)
        return out_3d, out_2d
    
    def grad_image(self):
        grads = torch.clone(self.mesh.vertices.grad).detach()
        vcolormat = pyr.Material(use_vertex_color=True)
        grads_mag = torch.abs(grads)
        vcolors = (grads_mag - grads_mag.min()) / (grads_mag.max()-grads_mag.min())
        gradobj = pyr.Object(self.mesh.vertices, self.mesh.indices, material=vcolormat, normals=self.mesh.normals, colors=vcolors)
        cameras = self.settings[0]
        gradscenes = [pyr.Scene(c, objects=[gradobj]) for c in cameras]
        grads_rendered = pyr.render_albedo(gradscenes)
        return grads_rendered

    def clean(self):
        recompute_normals(self.mesh)
        if self.learning_texture:
            # Create a new texture with our texels
            self.mesh.material.diffuse_reflectance = pyr.Texture(self.learned_tex)
        

    def parameters(self):
        return self.params
    
    def free_mesh(self):
        self.mesh.vertices.requires_grad = True
        self.params.append(self.mesh.vertices)
        return self
    
    def free_tex(self):
        self.learning_texture = True
        self.learned_tex = self.mesh.material.diffuse_reflectance.texels.clone().detach()
        self.learned_tex.requires_grad = True
        self.params.append(self.learned_tex)
        self.mesh.material.diffuse_reflectance = pyr.Texture(self.learned_tex)
        return self

    def replace_tex_with_color(self, color):
        self.tex_backup = self.mesh.material.diffuse_reflectance.texels.clone().detach()
        new_tex_img = torch.ones(self.tex_backup.shape, device=pyr.get_device())
        new_tex_img[:,:,0] = color[0]
        new_tex_img[:,:,1] = color[1]
        new_tex_img[:,:,2] = color[2]
        self.mesh.material.diffuse_reflectance = pyr.Texture(new_tex_img)

    def restore_tex(self):
        '''
        Restore the texture from the backup. Only works once per backup!
        '''
        assert self.tex_backup is not None
        self.mesh.material.diffuse_reflectance = pyr.Texture(self.tex_backup)
        self.tex_backup = None


    def save(self, new_name, allow_overwrite=False):
        save_3d(self.mesh, new_name, allow_overwrite=allow_overwrite)