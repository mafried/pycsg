from pycsg.csg_node import CSGNode, register_node_type
import numpy as np


class Sphere(CSGNode):
    def __init__(self, radius, name):
        super().__init__(name)

        self.radius = radius

    def signed_distance(self, p):
        return np.linalg.norm(p, axis=1) - self.radius

    def to_dict(self):
        d = super().to_dict().copy()
        d['radius'] = self.radius
        return d

    @staticmethod
    def from_dict(d, children):
        return Sphere(d['radius'],d['name'])

    @staticmethod
    def node_type():
        return 'sphere'


class Box(CSGNode):
    def __init__(self, size, name):
        super().__init__(name)

        self.size = np.array(size)

    def signed_distance(self, p):

       return np.max(np.abs(p) - self.size / 2.0, axis=1)

    def to_dict(self):
        d = super().to_dict().copy()
        d['size'] = list(self.size)
        return d

    @staticmethod
    def from_dict(d, children):
        return Box(d['size'],d['name'])

    @staticmethod
    def node_type():
        return 'box'


register_node_type(Sphere)
register_node_type(Box)
