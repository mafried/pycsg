import skimage.measure as sk
import numpy as np

def node_to_mesh(node, grid_min, grid_max, cell_size):
    nx = int((grid_max[0] - grid_min[0]) / cell_size)
    ny = int((grid_max[1] - grid_min[1]) / cell_size)
    nz = int((grid_max[2] - grid_min[2]) / cell_size)

    x_ = np.linspace(grid_min[0], grid_max[0], nx)
    y_ = np.linspace(grid_min[1], grid_max[1], ny)
    z_ = np.linspace(grid_min[2], grid_max[2], nz)

    x, y, z = np.meshgrid(x_, y_, z_)

    p = np.stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)), axis=1)

    d = node.signed_distance(p)

    volume = d.reshape((nx,ny,nz)).transpose()

    vertices, faces, normals, _ = sk.marching_cubes_lewiner(volume, level=0)

    return vertices, faces, normals