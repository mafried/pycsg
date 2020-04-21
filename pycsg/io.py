from pycsg.operations import Union, Intersection, Complement, Difference
from pycsg.primitives import Cylinder, Sphere, Box
from pycsg.transforms import Pose

import uuid


def node_from_old_json_format(json, degrees=True):
    children = [node_from_old_json_format(c, degrees) for c in json['childs']] if 'childs' in json else []

    def pose(j, n):
        return Pose(j['params']['center'], j['params']['rotation'], [n], str(uuid.uuid1()), degrees)

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
