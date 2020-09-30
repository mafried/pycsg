import numpy as np


def volume_from_node(node, grid_min, grid_max, cell_size, max_dist):

    #TODO
    pass


def point_cloud_from_node(node, grid_min, grid_max, cell_size, max_dist):

    x_ = np.linspace(grid_min[0], grid_max[0], int((grid_max[0] - grid_min[0]) / cell_size))
    y_ = np.linspace(grid_min[1], grid_max[1], int((grid_max[1] - grid_min[1]) / cell_size))
    z_ = np.linspace(grid_min[2], grid_max[2], int((grid_max[2] - grid_min[2]) / cell_size))

    x, y, z = np.meshgrid(x_, y_, z_)

    p = np.stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)), axis=1)

    d = node.signed_distance(p)

    return p[np.where(abs(d[:]) < max_dist)]

def voxel_from_node(node, dim):
    x = y = z = np.linspace(0, dim, dim, endpoint=False)
    x, y, z = np.meshgrid(x, y, z)
    p = np.stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)), axis=1)
    d = node.signed_distance(p)
    v = d.reshape((dim, dim, dim)).transpose()
    v[v==0] = -1
    v[v>0] = 0
    v[v<0] = 1
    return np.array(v, dtype=bool)
