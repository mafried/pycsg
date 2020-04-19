import numpy as np
from pycsg.csg_node import CSGNode, register_node_type


class Union(CSGNode):
    def __init__(self, name, children):
        super().__init__(name, children)

    def signed_distance(self, p):

        sd_per_child = [n.signed_distance(p) for n in self.children.values()]
        return np.amin(np.stack(sd_per_child), axis=0).transpose() if len(self) > 0 else 0.0

    @staticmethod
    def from_dict(d, children):
        return Union(d['name'], children)

    @staticmethod
    def node_type():
        return 'union'


class Intersection(CSGNode):
    def __init__(self, name, children):
        super().__init__(name, children)

    def signed_distance(self, p):

        sd_per_child = [n.signed_distance(p) for n in self.children.values()]
        return np.amax(np.stack(sd_per_child), axis=0).transpose() if len(self) > 0 else 0.0

    @staticmethod
    def from_dict(d, children):
        return Intersection(d['name'], children)

    @staticmethod
    def node_type():
        return 'inter'


class Complement(CSGNode):
    def __init__(self, name, children):
        super().__init__(name, children)

    def signed_distance(self, p):
        n = list(self.children.values())[0]
        return -n.signed_distance(p)

    @staticmethod
    def from_dict(d, children):
        return Complement(d['name'], children)

    @staticmethod
    def node_type():
        return 'complement'


class Difference(CSGNode):
    def __init__(self, name, children):
        super().__init__(name, children)

        n0 = list(self.children.values())[0]
        n1 = list(self.children.values())[1]

        self.diff = Intersection('', [n0, Complement('', [n1])])

    def signed_distance(self, p):

        return self.diff.signed_distance(p)

    @staticmethod
    def from_dict(d, children):
        return Difference(d['name'], children)

    @staticmethod
    def node_type():
        return 'minus'


register_node_type(Difference)
register_node_type(Complement)
register_node_type(Intersection)
register_node_type(Union)
