from abc import ABC, abstractmethod


class CSGNode(ABC):

    def __init__(self, name, children=[]):
        self.name = name
        self.children = {c.name: c for c in children}

    @abstractmethod
    def signed_distance(self, p):
        pass

    def gradient(self, p):
        pass

    def to_dict(self):
        return {
            'name': self.name,
            'children': [c.to_dict() for c in self.children.values()],
            'type': get_node_type(type(self))
        }

    @staticmethod
    def from_dict(d):
        return getattr(node_types[d['type']], 'from_dict')(d, [CSGNode.from_dict(c) for c in d['children']])

    def __getitem__(self, key):
        return self.children[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        if not isinstance(value, CSGNode):
            raise TypeError()

        self.children[self.__keytransform__(key)] = value

    def __delitem__(self, key):
        del self.children[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self.children)

    def __len__(self):
        return len(self.children)

    def __keytransform__(self, key):
        return key


node_types = {}


def get_node_type(type):
    return getattr(type, 'node_type')()


def register_node_type(node_type):
    node_types[get_node_type(node_type)] = node_type


def create_node(node_type_name, dictionary):
    return node_types[node_type_name].from_dict(dictionary)
