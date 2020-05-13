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

def node_from_stack_format(stack, degrees=True):
	label_to_type = {
		'001': 'box',
		'010': 'cylinder',
		'011': 'sphere',
		'100': 'union',
		'101': 'intersect',
		'110': 'minus'
	}
	
	def name():
		return str(uuid.uuid1())

	def deg(rot):
		return list(map(lambda r: round(r * 365), rot))

	ops = {
		'union': lambda c: Union(name(), c),
		'intersect': lambda c: Intersection(name(), c),
		'minus': lambda c: Difference(name(), c)
	}

	geo = {
		'box': lambda s: Pose(s[1:4], deg(s[4:7]), [Box(s[7:], name())], name(), degrees),
		'cylinder': lambda s: Pose(s[1:4], deg(s[4:7]), [Cylinder(s[7], s[8], name())], name(), degrees),
		'sphere': lambda s: Pose(s[1:4], deg(s[4:7]), [Sphere(s[7], name())], name(), degrees)
	}

	stack = [line.strip('\n').split(',') for line in open(stack)]
	stack = [[item[0]] + list(map(float, item[1:])) for item in stack]
	labels = [label_to_type[item[0]] for item in stack] if len(stack) > 1 else [geo[label_to_type[item[0]]](item) for item in stack]

	while not isinstance(labels[0], CSGNode):
		for i, label in enumerate(labels[:-2]):
			if label in ops.keys() and (labels[i+1] in geo.keys() or isinstance(labels[i+1], CSGNode)) and (labels[i+2] in geo.keys() or isinstance(labels[i+2], CSGNode)):
				labels[i] =  ops[label]([geo[labels[i+2]](stack[i+2]) if labels[i+2] in geo.keys() else stack[i+2], geo[labels[i+1]](stack[i+1]) if labels[i+1] in geo.keys() else stack[i+1]])
				stack[i] = labels[i]
				labels = labels[:i+1] + labels[i+3:]
				stack = stack[:i+1] + stack[i+3:]
				
	return labels[0]
