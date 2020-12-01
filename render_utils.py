import pyredner as pyr
import torch
import matplotlib.pyplot as plt
import matplotlib

    

def init_renderer(mesh):
    obj_fp = mesh.filename
    pyr.set_print_timing(False)
    objects = pyr.load_obj(obj_fp, return_objects=True)
    #camera = pyr.automatic_camera_placement(objects, (256, 256))
    camera = pyr.Camera(
        position=torch.tensor([1.2,0,0], dtype=torch.float32),
        look_at=torch.tensor([0,0,0], dtype=torch.float32),
        up=torch.tensor([0,1,0], dtype=torch.float32),
        fov = torch.tensor([60], dtype=torch.float32),
        resolution = (256, 256),
        camera_type=pyr.camera_type.perspective
    )
    lights = [
        pyr.DirectionalLight(
            direction=torch.tensor([-1, 0, 0], dtype=torch.float32, device=pyr.get_device()),
            intensity=torch.tensor([1, 1, 1], dtype=torch.float32, device=pyr.get_device())
        ),
        pyr.DirectionalLight(
            direction=torch.tensor([1, 0, 0], dtype=torch.float32, device=pyr.get_device()),
            intensity=torch.tensor([1, 1, 1], dtype=torch.float32, device=pyr.get_device())
        )
    ]
    return objects, camera, lights

def render_mesh(verts, render_settings):
    objects, camera, lights = render_settings
    objects[0].vertices = verts.float()
    scene = pyr.Scene(camera=camera, objects=objects)
    images = pyr.render_deferred(scene, lights, alpha=True)
    return images

def show_images(imgs):
    if isinstance(imgs, list) or len(imgs.shape) == 4:
        for im in imgs:
            show_images(im)
    else:
        imgrgb = imgs[:,:,:3]
        imga = imgs[:,:,3:]
        gammargb = torch.pow(imgrgb, 1.0 / 2.2)
        finalimg = torch.cat([gammargb, imga], dim=2)
        plt.imshow(finalimg.cpu().detach().numpy())
        plt.show()

