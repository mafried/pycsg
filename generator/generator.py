from configparser import ConfigParser
import random
import sys

from pycsg.primitives import *
from generator.utils import *
from generator.patterns import *

def get_random_tree(n_primitives, configfile="generator/config.ini"):
    assert n_primitives >= 1

    config = ConfigParser()
    config.read(configfile)

    primitives = []
    if config.getboolean("primitives", "box"):
        primitives.append("box")
    if config.getboolean("primitives", "cylinder"):
        primitives.append("cylinder")
    if config.getboolean("primitives", "sphere"):
        primitives.append("sphere")

    patterns = []
    if config.getboolean("patterns", "cube"):
        patterns.append("cube")
    if config.getboolean("patterns", "halfcylinder"):
        patterns.append("halfcylinder")
    if config.getboolean("patterns", "halfsphere"):
        patterns.append("halfsphere")

    operations = []
    if config.getboolean("operations", "union"):
        operations.append("union")
    if config.getboolean("operations", "intersection"):
        operations.append("intersection")
    if config.getboolean("operations", "difference"):
        operations.append("difference")

    shapes = [primitive for primitive in primitives]
    for pattern in patterns:
        if patterns_dict[pattern]["box"] and "box" not in shapes:
            continue
        if patterns_dict[pattern]["cylinder"] and "cylinder" not in shapes:
            continue
        if patterns_dict[pattern]["sphere"] and "sphere" not in shapes:
            continue
        if patterns_dict[pattern]["operation"] not in operations and patterns_dict[pattern]["operation"] is not None:
            continue
        if patterns_dict[pattern]["n_primitives"] > n_primitives:
            continue
        shapes.append(pattern)

    size_min = config.getint("size", "min")
    size_max = config.getint("size", "max")
    size_step = config.getint("size", "step")

    rot_min = config.getint("rotation", "min")
    rot_max = config.getint("rotation", "max")
    rot_step = config.getint("rotation", "step")

    mirror_x = config.getboolean("mirror", "x")
    mirror_y = config.getboolean("mirror", "y")
    mirror_z = config.getboolean("mirror", "z")

    joints = []
    tree = None

    while n_primitives > 0:
        joint, base, direction, shape, op = get_random_joint(joints, shapes, operations)
        size, pos, rot = get_random_params(tree, joint, base, direction, shape, op, size_min, size_max, size_step, rot_min, rot_max, rot_step)
        tree, subtree = update_tree(tree, op, shape, size, pos, rot)
        joints = update_joints(joints, joint, tree, subtree, pos, op, direction, operations, size_min)
        n_primitives = update_n_primitives(n_primitives, shape)

    return tree

