import json

import polyscope as ps

from pycsg.csg_node import CSGNode
from pycsg.operations import Union, Intersection, Difference
from pycsg.primitives import Sphere, Box
from pycsg.transforms import Pose
from pycsg.sampling import point_cloud_from_node

tree = Union('u2', [Cylinder(1.0,2.0,'s1'), Pose([1.0, 0.0, 0.0], [45.0,45.0,45.0], [Box([1.0,1.0,1.0],'b1')], 'p1')])

print(tree.to_dict())


tree2 = CSGNode.from_dict(tree.to_dict())

print(tree2.to_dict())

print(json.dumps(tree2.to_dict()))

pc = point_cloud_from_node(tree2, [-4,-4,-4], [4,4,4], 0.05, 0.05)

# TODO
# v,f,n = pointcloud_to_mesh(pc)
# ps.register_surface_mesh("mesh 1", v, f)

ps.init()
ps.register_point_cloud("points 1", pc)
ps.show()
