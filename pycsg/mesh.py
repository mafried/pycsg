import skimage.measure as sk

def node_to_mesh(node):

    volume = sampling

    vertices, faces, normals, _ = sk.marching_cubes_lewiner(pc.transpose(), level=0)

    return vertices, faces, normals