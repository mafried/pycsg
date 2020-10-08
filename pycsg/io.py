from pycsg.operations import Union, Intersection, Complement, Difference
from pycsg.primitives import Cylinder, Sphere, Box
from pycsg.transforms import Pose
from pycsg.csg_node import CSGNode, get_node_type

import uuid
import sys
import numpy as np


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

def node_to_stack(node):

    ops = {
        "minus": "-",
        "union": "+",
        "inter": "*"
    }

    geo = {
        "box": "b",
        "cylinder": "c",
        "sphere": "s"
    }

    def get_op(node_type):
        op = [ops[node_type]]
        for i in range(9):
            op.append("0")
        return op

    def get_params(p):

        def get_pos():
            pose = p.pose
            x, y, z = 0, 0, 0
            v = np.array([x, y, z, 1])
            x, y, z, _ = np.matmul(pose, v)
            for coord in [x, y, z]:
                assert coord.is_integer()
            x, y, z = int(x), int(y), int(z)
            size = [x, y, z]
            return [str(i+32) for i in size] # remove +32 for range -32 <= x <= 32 instead of 0 <= x <= 64

        def get_rot():
            rot = [int(round(r)) for r in p.r.as_euler("xyz", degrees=True)]
            return [str(i) for i in rot]

        def get_size():
            children = p.children
            for key, prim in children.items():
                node_type = get_node_type(prim)
                if node_type == "box":
                    w, d, h = prim.size
                    size = [w, d, h]
                elif node_type == "cylinder":
                    r = prim.radius
                    h = prim.height
                    size = [2*r, h, 0]
                elif node_type == "sphere":
                    r = prim.radius
                    size = [2*r, 0, 0]
                else:
                    sys.exit("unknown node type in get_size()")
            return [str(i) for i in size]

        def get_label():
            children = p.children
            for key, prim in children.items():
                node_type = get_node_type(prim)
            return [geo[node_type]]

        assert len(p.children) == 1
        pos = get_pos()
        rot = get_rot()
        size = get_size()
        label = get_label()
        params = label + pos + rot + size
        return params

    def get_subtree(p):
        arr = []
        node_type = get_node_type(p)
        if node_type in ops:
            children = p.children
            for k, obj in children.items():
                sub = get_subtree(obj)
                for item in sub:
                    arr.append(item)
            op = get_op(node_type)
            arr.append(op)
        else:
            assert node_type == "pose"
            params = get_params(p)
            arr.append(params)
        return arr

    arr = get_subtree(node)
    stack = ",".join([",".join(i) for i in arr])

    return stack
