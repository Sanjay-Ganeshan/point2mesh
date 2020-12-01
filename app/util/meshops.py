import pyredner as pyr
import torch
import torch.cuda
import openmesh
import tempfile
import os

from typing import Union

# Functions to export
__all__ = ['decimate', 'subdivide', 'laplacian', 'recompute_normals']

def decimate(
    obj: pyr.Object,
    verbose = False):
    """
    Decimates a mesh, reducing the number of faces by 2.
    This is EXTREMELY inefficient, and not differentiable - use it sparingly!
    Modifies the input mesh.
    """
    # Let's make a temporary directory
    intermediate_dir = tempfile.mkdtemp()

    orig_out = os.path.join(intermediate_dir, "orig.obj")
    new_out = os.path.join(intermediate_dir, "decim.obj")

    if verbose:
        print("Made temp dir:")
        print(intermediate_dir)

    # First, let's save the redner
    pyr.save_obj(obj, orig_out)
    # Now, let's load in openmesh
    mesh = openmesh.read_trimesh(orig_out)
    # Now, decimate by half
    orig_nfaces = mesh.n_faces()
    
    if verbose:
        print("Original # of faces:", orig_nfaces)

    decimator = openmesh.TriMeshDecimater(mesh)
    algorithm = openmesh.TriMeshModQuadricHandle()

    decimator.add(algorithm)
    decimator.initialize()
    decimator.decimate_to_faces(n_faces = round(orig_nfaces / 2))

    mesh.garbage_collection()

    if verbose:
        print("New # of faces:", mesh.n_faces())

    openmesh.write_mesh(
        new_out,
        mesh)

    # Now, we have it. Load it back into redner
    decim_obj = pyr.load_obj(new_out, return_objects=True)[0]
    # And set the faces/indices
    obj.vertices = decim_obj.vertices
    obj.indices = decim_obj.indices
    
    # Recompute normals - the face normals have been broken
    recompute_normals(obj)

    # Finally, clean up the dir
    files_to_delete = os.listdir(intermediate_dir)
    for each_file in files_to_delete:
        apath = os.path.join(intermediate_dir, each_file)
        if verbose:
            print("Deleting",apath)
        os.remove(apath)
    if verbose:
        print("Deleting",intermediate_dir)
    os.rmdir(intermediate_dir)

def subdivide(
    vertices: Union[torch.FloatTensor, torch.cuda.FloatTensor],
    indices: Union[torch.IntTensor, torch.cuda.IntTensor]):
    """
    Subdivides a mesh, increasing the number of vertices and faces
    :param vertices: The vertices of the original mesh. Shape |V| x 3
    :param indices: The faces of the mesh. Shape |F| x 3
    :returns: (new_vertices, new_faces)
    """
    pass

def laplacian(
    vertices: Union[torch.FloatTensor, torch.cuda.FloatTensor],
    indices: Union[torch.IntTensor, torch.cuda.IntTensor]):
    '''
    Smooth a mesh using the Laplacian method. Each output vertex becomes
    the average of its neighbors

    :param vertices: float32 tensor of vertices. (shape |V| x 3)
    :param indices: 
    :returns: this is a description of what is returned
    :raises keyError: raises an exception
    '''
    nvertices = vertices.shape[0]
    nfaces = indices.shape[0]
    indices = indices.astype(torch.long)
    totals = torch.zeros_like(vertices, dtype=torch.float32, device=vertices.device)
    num_neighbors = torch.zeros((nvertices,), dtype=torch.float32, device=vertices.device)
    
    _face_add(vertices, indices, 1, 0, totals, num_neighbors)
    _face_add(vertices, indices, 2, 0, totals, num_neighbors)
    _face_add(vertices, indices, 0, 1, totals, num_neighbors)
    _face_add(vertices, indices, 2, 1, totals, num_neighbors)
    _face_add(vertices, indices, 0, 2, totals, num_neighbors)
    _face_add(vertices, indices, 1, 2, totals, num_neighbors)
    
    weighted_vertices = totals / num_neighbors.unsqueeze(1)
    return weighted_vertices

def recompute_normals(obj: pyr.Object):
    """
    Recomputes smooth shading vertex normals for obj, and sets them
    accordingly.

    :param obj: A PyRedner object
    """
    obj.normals = pyr.compute_vertex_normal(obj.vertices.detach(), obj.indices.detach(), 'cotangent')
    obj.normal_indices = None

# Helper functions to make these work
def _face_add(
    verts: Union[torch.FloatTensor, torch.cuda.FloatTensor],
    faces: Union[torch.LongTensor, torch.cuda.LongTensor],
    face_ix_from: int,
    face_ix_to: int, 
    output: Union[torch.FloatTensor, torch.cuda.FloatTensor],
    counts: Union[torch.FloatTensor, torch.cuda.FloatTensor]):
    """
    Helper function that adds the vertex values from the given face into
    an array, at its neighbors indices. Useful for Laplacians

    :param verts: The vertices of the mesh
    :param faces: The faces of the mesh
    :face_ix_from: Which index (0,1,2) of the faces to get from
    :face_ix_to: Which face index (0,1,2) to add at
    :output: Output for sum of vertices
    :counts: Output for # of elements added
    """
    ind_1d = faces[:,face_ix_to]
    one = torch.ones((1,), dtype=torch.float32, device=verts.device)
    data = verts[faces[:,face_ix_from]]
    ind_2d, _ = torch.broadcast_tensors(ind_1d.unsqueeze(1), data)
    one_1d, _ = torch.broadcast_tensors(one, ind_1d)
    output.scatter_add_(0, ind_2d, data)
    counts.scatter_add_(0, ind_1d, one_1d)
