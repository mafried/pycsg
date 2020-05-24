import json

import polyscope as ps

from pycsg.csg_node import CSGNode
from pycsg.io import node_from_old_json_format, node_from_stack_format
from pycsg.mesh import node_to_mesh
from pycsg.operations import Union, Intersection, Difference
from pycsg.primitives import Sphere, Box, Cylinder
from pycsg.transforms import Pose
from pycsg.sampling import point_cloud_from_node

tree = Union('u2', [Cylinder(1.0,2.0,'s1'), Pose([1.0, 0.0, 0.0], [45.0,45.0,45.0], [Box([1.0,1.0,1.0],'b1')], 'p1')])

with open('data/bobbin.json') as json_file:
    tree = node_from_old_json_format(json.load(json_file), False)
    print('loaded tree: {}'.format(tree.to_dict()))


print(tree.to_dict())


tree2 = CSGNode.from_dict(tree.to_dict())

print(tree2.to_dict())

print(json.dumps(tree2.to_dict()))

# uncomment the following line for stack test
# tree = node_from_stack_format('data/stack.txt')

pc = point_cloud_from_node(tree, [-20,-20,-20], [20,20,20], 0.2, 0.2)

v,f,n = node_to_mesh(tree, [-20,-20,-20], [20,20,20], 0.8)

ps.init()
ps.register_surface_mesh("mesh 1", v, f)
ps.register_point_cloud("points 1", pc)
ps.show()
