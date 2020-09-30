from configparser import ConfigParser
import random
import sys

from pycsg.primitives import *
from generator.utils import *

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

    operations = []
    if config.getboolean("operations", "union"):
        operations.append("union")
    if config.getboolean("operations", "intersection"):
        operations.append("intersection")
    if config.getboolean("operations", "difference"):
        operations.append("difference")

    size_min = config.getint("size", "min")
    size_max = config.getint("size", "max")
    size_step = config.getint("size", "step")
    
    for val in [size_min, size_max, size_step]:
        assert val % 2 == 0

    assert size_min > 0
    assert size_max < 64

    joints = []
    tree = None

    while n_primitives > 0:
        joint, base, direction, shape, op = get_random_joint(joints, primitives, operations)
        size, pos, rot = get_random_params(tree, joint, base, direction, shape, op, size_min, size_max, size_step)
        tree, subtree = update_tree(tree, op, shape, size, pos, rot)
        joints = update_joints(joints, joint, tree, subtree, pos, op, direction, operations, size_min)
        n_primitives = update_n_primitives(n_primitives, shape)

    return tree
