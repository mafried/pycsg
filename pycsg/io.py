from pycsg.operations import Union, Intersection, Complement, Difference
from pycsg.primitives import Cylinder, Sphere, Box
from pycsg.transforms import Pose
from pycsg.csg_node import CSGNode

import uuid


def node_from_old_json_format(json, degrees=True):
    children = [node_from_old_json_format(c, degrees) for c in json['childs']] if 'childs' in json else []

    def pose(j, n):
        return Pose(j['params']['center'], j['params']['rotation'] if 'rotation' in j['params'] else [0,0,0], [n],
                    str(uuid.uuid1()), degrees)

    def name_or_cnt(j):
        return j['name'] if 'name' in j else str(uuid.uuid1())

    ops = {
        'union': lambda n: Union(name_or_cnt(n), children),
        'intersect': lambda n: Intersection(name_or_cnt(n), children),
        'subtract': lambda n: Difference(name_or_cnt(n), children)
    }

    geo = {
        'cylinder': lambda n: pose(json, Cylinder(n['params']['radius'], n['params']['height'], name_or_cnt(n))),
        'sphere': lambda n: pose(json, Sphere(n['params']['radius'], name_or_cnt(n))),
        'cube': lambda n: pose(json, Box([c * 2.0 for c in n['params']['radius']], name_or_cnt(n)))
    }

    return ops[json['op']](json) if 'op' in json else geo[json['geo']](json)

def node_from_stack(stack, degrees=True):
    def name():
        return str(uuid.uuid1())

    ops = {
        "+": lambda c: Union(name(), c),
        "*": lambda c: Intersection(name(), c),
        "-": lambda c: Difference(name(), c)
    }

    geo = {
        "b": lambda s: Pose(s[1:4], s[4:7], [Box(s[7:], name())], name(), degrees),
        "c": lambda s: Pose(s[1:4], s[4:7], [Cylinder(s[7], s[8], name())], name(), degrees),
        "s": lambda s: Pose(s[1:4], s[4:7], [Sphere(s[7], name())], name(), degrees)
    }

    stack = np.array(stack.split(",")).reshape(-1, 10)
    stack = [[item[0]] + list(map(float, item[1:])) for item in stack]
    labels = [line[0] for line in stack]
    labels = [geo[label](stack[i]) if label in geo else label for i, label in enumerate(labels)]
    restart = True
    while restart:
        restart = False
        for i, label in enumerate(labels):
            if label in ops:
                labels[i] = ops[label]([labels[i-2], labels[i-1]])
                labels.pop(i-1)
                labels.pop(i-2)
                restart = True
                break

    return labels[-1]
