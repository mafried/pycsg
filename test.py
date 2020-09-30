import polyscope as ps

from generator.generator import *
from pycsg.mesh import node_to_mesh
from pycsg.sampling import point_cloud_from_node

tree = get_random_tree(n_primitives=6, configfile="generator/config.ini")

v, f, n = node_to_mesh(tree, [-32, -32, -32], [32, 32, 32], 0.5)

ps.init()
ps.register_surface_mesh("mesh 1", v, f)
ps.show()
