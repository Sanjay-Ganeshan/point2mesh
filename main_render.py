import torch
from models.layers.mesh import Mesh, PartMesh
from models.networks import init_net, sample_surface, local_nonuniform_penalty
import utils
import render_utils
import numpy as np
from models.losses import chamfer_distance
from options import Options
import time
import os

# Get command line args
options = Options()
opts = options.args

torch.manual_seed(opts.torch_seed)
device = torch.device('cuda:{}'.format(opts.gpu) if torch.cuda.is_available() else torch.device('cpu'))
print('device: {}'.format(device))

# initial mesh
mesh = Mesh(opts.initial_mesh, device=device, hold_history=True)
render_settings = render_utils.init_renderer(mesh)

# input point cloud
input_xyz, input_normals = utils.read_pts(opts.input_pc)
# normalize point cloud based on initial mesh
input_xyz /= mesh.scale
input_xyz += mesh.translations[None, :]
input_xyz = torch.Tensor(input_xyz).type(options.dtype()).to(device)[None, :, :]
input_normals = torch.Tensor(input_normals).type(options.dtype()).to(device)[None, :, :]

# Split the mesh into parts
part_mesh = PartMesh(mesh, num_parts=1, bfs_depth=opts.overlap)
print(f'number of parts {part_mesh.n_submeshes}')

# Initialize displacement network
net, optimizer, rand_verts, scheduler = init_net(mesh, part_mesh, device, opts)

# For each iteration
for i in range(opts.iterations):
    # Choose sampling density. Later = more samples (Coarse->Fine)
    num_samples = options.get_num_samples(i % opts.upsamp)
    
    # If using a global step, zero-out the gradient after doing all parts & back.
    # Otherwise, step and zero after each part
    if opts.global_step:
        optimizer.zero_grad()
    start_time = time.time()

    # net(rand_verts, part_mesh) calls forward on all the submeshes
    # yielding submesh index & estimated vertices
    # (Estimation is after autoencoder & edge2vert)
    for part_i, est_verts in enumerate(net(rand_verts, part_mesh)):
        if not opts.global_step:
            optimizer.zero_grad()

        # Update the mesh to use the estimated vertices, instead of the real ones
        # However, the mesh remembers the old ones as well?
        part_mesh.update_verts(est_verts[0], part_i)

        rendered_images = render_utils.render_mesh(est_verts[0], render_settings)

        # Repetitive? Choose sampling density
        num_samples = options.get_num_samples(i % opts.upsamp)

        # Sample the prediction
        recon_xyz, recon_normals = sample_surface(part_mesh.main_mesh.faces, part_mesh.main_mesh.vs.unsqueeze(0), num_samples)

        # The target is "input_xyz", and "input_normals"
        # Here, we calculate the difference between the sampled points, and their normals, and
        # The target via chamfer loss (find nearest x->y and nearest y->x, sum)
        # calc chamfer loss w/ normals
        recon_xyz, recon_normals = recon_xyz.type(options.dtype()), recon_normals.type(options.dtype())
        xyz_chamfer_loss, normals_chamfer_loss = chamfer_distance(recon_xyz, input_xyz, x_normals=recon_normals, y_normals=input_normals,
                                              unoriented=opts.unoriented)

        # Every once in a while, calculate the beamgap loss too. This is probably expensive.
        # It isn't used in the demo. We only use this at the beginning
        # It REPLACES chamfer loss.
        # Chamfer is testing both the normals & the positions
        # the normals have some weighting.
        loss = (xyz_chamfer_loss + (opts.ang_wt * normals_chamfer_loss))
        
        # Apply a smoothness loss. Normally has weight 0.1.
        # This is calculated as, the norm of for any given edge, the difference in area
        # between it's incident faces.
        if opts.local_non_uniform > 0:
            loss += opts.local_non_uniform * local_nonuniform_penalty(part_mesh.main_mesh).float()

        # Back propogate
        loss.backward()

        # Step for each sub-mesh
        if not opts.global_step:
            optimizer.step()
            scheduler.step()
        
        # Detach the vertices from the computation graph? Maybe for efficiency?
        part_mesh.main_mesh.vs.detach_()
    
    # IF using global steps, step after all the parts have accumulated gradients
    if opts.global_step:
        optimizer.step()
        scheduler.step()
    
    # Keep a timer
    end_time = time.time()

    # Save / print
    if i % 1 == 0:
        print(f'{os.path.basename(opts.input_pc)}; iter: {i} out of: {opts.iterations}; loss: {loss.item():.4f};'
              f' sample count: {num_samples}; time: {end_time - start_time:.2f}')
    if i % opts.export_interval == 0 and i > 0:
        print('exporting reconstruction... current LR: {}'.format(optimizer.param_groups[0]['lr']))
        render_utils.show_images(rendered_images)
        with torch.no_grad():
            part_mesh.export(os.path.join(opts.save_path, f'recon_iter_{i}.obj'))

    # Coarse -> Fine. Rarely, subdivide the mesh, and compute the manifold again. This keeps it watertight,
    # and more faces = more detail.
    if (i > 0 and (i + 1) % opts.upsamp == 0):
        # We're splitting the main mesh
        mesh = part_mesh.main_mesh
        # Multiply the # of faces by 1.5, but don't go above the given threshold
        num_faces = int(np.clip(len(mesh.faces) * 1.5, len(mesh.faces), opts.max_faces))

        # Only up-sample if the # of faces actually changed (if we were already at max thres
        # we wouldn't have to)
        # Though the # of faces may not change, manifolding may help keep the mesh waterproof
        if num_faces > len(mesh.faces) or opts.manifold_always:
            # up-sample mesh
            mesh = utils.manifold_upsample(mesh, opts.save_path, Mesh,
                                           num_faces=min(num_faces, opts.max_faces),
                                           res=opts.manifold_res, simplify=True)
            render_settings = render_utils.init_renderer(mesh)
            # Split it into parts again
            part_mesh = PartMesh(mesh, num_parts=1, bfs_depth=opts.overlap)
            print(f'upsampled to {len(mesh.faces)} faces; number of parts {part_mesh.n_submeshes}')
            
            # Now we're using a totally different mesh, throw away the old optimizer / weights
            # and start anew.
            net, optimizer, rand_verts, scheduler = init_net(mesh, part_mesh, device, opts)

# Finally, we have a reconstructed mesh, so just output that.
with torch.no_grad():
    mesh.export(os.path.join(opts.save_path, 'last_recon.obj'))