import sys
import random
import numpy as np
import uuid

from pycsg.csg_node import CSGNode
from pycsg.primitives import Sphere, Box, Cylinder
from pycsg.operations import Union, Difference, Intersection
from pycsg.transforms import Pose

def directions(param=None):
    if param is None:
        return ["left", "right", "front", "back", "bottom", "top"]
    elif param == "neg":
        return ["left", "front", "back"]
    elif param == "pos":
        return ["right", "back", "top"]
    elif param == "x":
        return ["left", "right"]
    elif param == "y":
        return ["front", "back"]
    elif param == "z":
        return ["bottom", "top"]
    else:
        sys.exit("unknown param in directions()")

def rotations(param=None):
    if param is None:
        return ((90, 0, 0), (0, 0, 0), (0, 0, 90))
    elif param == "x":
        return (90, 0, 0)
    elif param == "y":
        return (0, 0, 0)
    elif param == "z":
        return (0, 0, 90)
    elif param == (90, 0, 0):
        return "x"
    elif param == (0, 0, 0):
        return "y"
    elif param == (0, 0, 90):
        return "z"
    else:
        sys.exit("unknown param in rotations")

def operations():
    return ["union", "difference", "intersection"]

def get_distances(tree, x, y, z):
    def convert(dd):
        d = np.array(dd)
        d[d>0] = 1
        d[d<0] = -1
        return d

    xx, yy, zz = [], [], []

    for i in range(-32, 32+1):
        px = np.array([[i, y, z]])
        py = np.array([[x, i, z]])
        pz = np.array([[x, y, i]])

        dx = tree.signed_distance(px)
        dy = tree.signed_distance(py)
        dz = tree.signed_distance(pz)

        xx.append(float(dx))
        yy.append(float(dy))
        zz.append(float(dz))

    xx = convert(xx)
    yy = convert(yy)
    zz = convert(zz)

    return xx, yy, zz

def get_random_joint(joints, shapes, ops):
    if len(joints) == 0:
        joint = None
        base = None
        direction = None
        op = None
    else:
        joint = random.choice(joints)
        base = joint.get("base")
        directions = [k for k in joint.keys() if k != "base"]
        direction = random.choice(directions)
        ops = []
        if joint[direction][0]:
            ops.append("union")
        if joint[direction][1]:
            ops.append("difference")
        if joint[direction][2]:
            ops.append("intersection")
        op = random.choice(ops)
    shape = random.choice(shapes)
    return joint, base, direction, shape, op

def get_random_params(tree, joint, base, direction, shape, op, size_min, size_max, size_step):

    def get_dim_base():
        key = list(base.children.keys()).pop()
        primitive = base.children[key]
        shape = primitive.node_type()
        if shape == "box":
            width, depth, height = primitive.size
        elif shape == "cylinder":
            rot = tuple([int(round(r)) for r in base.r.as_euler("xyz", degrees=True)])
            if rotations(rot) == "x":
                width, depth, height = primitive.radius*2, primitive.radius*2, primitive.height
            elif rotations(rot) == "y":
                width, depth, height = primitive.radius*2, primitive.height, primitive.radius*2
            elif rotations(rot) == "z":
                width, depth, height = primitive.height, primitive.radius*2, primitive.radius*2
            else:
                sys.exit("unknown rotation in get_dim_base()")
        elif shape == "sphere":
            width, depth, height = primitive.radius*2, primitive.radius*2, primitive.radius*2
        else:
            sys.exit("error: unknown shape")
        return width, depth, height

    def get_pos_base():
        pose = base.pose
        x, y, z = 0, 0, 0
        v = np.array([x, y, z, 1])
        x, y, z, _ = np.matmul(pose, v)
        for coord in [x, y, z]:
            assert coord.is_integer()
        x, y, z = int(x), int(y), int(z)
        return x, y, z

    def get_pos_base_surface():
        x, y, z = get_pos_base()
        w, d, h = get_dim_base()
        for size in [w, d, h]:
            assert size % 2 == 0
        if direction == "left":
            x -= int(w/2)
        elif direction == "right":
            x += int(w/2)
        elif direction == "front":
            y -= int(d/2)
        elif direction == "back":
            y += int(d/2)
        elif direction == "bottom":
            z -= int(h/2)
        else:
            z += int(h/2)
        return x, y, z

    def get_pos_new_outside(w, d, h):
        x, y, z = get_pos_base_surface()
        if direction == "left":
            x -= int(w/2)
        elif direction == "right":
            x += int(w/2)
        elif direction == "front":
            y -= int(d/2)
        elif direction == "back":
            y += int(d/2)
        elif direction == "bottom":
            z -= int(h/2)
        else:
            z += int(h/2)
        return x, y, z

    def get_pos_new_inside(w, d, h):
        x, y, z = get_pos_base_surface()
        if direction == "left":
            x += int(w/2)
        elif direction == "right":
            x -= int(w/2)
        elif direction == "front":
            y += int(d/2)
        elif direction == "back":
            y -= int(d/2)
        elif direction == "bottom":
            z += int(h/2)
        else:
            z -= int(h/2)
        return x, y, z

    def get_inside():
        w, d, h = get_dim_base()
        for d in [w, d, h]:
            assert d % 2 == 0
        if direction in directions("x"):
            w = int(w/2)
        elif direction in directions("y"):
            d = int(d/2)
        else:
            h = int(h/2)
        if w % 2 != 0:
            w -= 1
        if d % 2 != 0:
            d -= 1
        if h % 2 != 0:
            h -= 1

        w = size_min if w < size_min else w
        d = size_min if d < size_min else d
        h = size_min if h < size_min else h

        return w, d, h

    def get_outside():

        def get_range(dd, c, axis):

            def is_perpendicular(axis):
                if axis == "x" and direction in directions("x"):
                    return True
                elif axis == "y" and direction in directions("y"):
                    return True
                elif axis == "z" and direction in directions("z"):
                    return True
                return False

            if is_perpendicular(axis):
                rr = 0
                c += 32
                if direction in directions("pos"):
                    while c <= 64 and dd[c] >= 0:
                        rr += 1
                        c += 1
                else:
                    while c >= 0 and dd[c] >= 0:
                        rr += 1
                        c -= 1
                rr -= 1
            else:
                c += 32
                neg, pos = 0, 0
                for i in range(c, -1, -1):
                    if dd[i] >= 0:
                        neg += 1
                    else:
                        break
                for i in range(c, 65):
                    if dd[i] >= 0:
                        pos += 1
                    else:
                        break
                rr = (min(neg, pos) - 1) * 2

            return rr

        x, y, z = get_pos_base_surface()

        xx1, yy1, zz1 = get_distances(tree, x, y, z)
        rx1 = get_range(xx1, x, "x")
        ry1 = get_range(yy1, y, "y")
        rz1 = get_range(zz1, z, "z")

        if direction in directions("x"):
            l_max = rx1 if rx1 < size_max else size_max
            x = x - int(l_max/2) if direction == "left" else x + int(l_max/2)
        elif direction in directions("y"):
            l_max = ry1 if ry1 < size_max else size_max
            y = y - int(l_max/2) if direction == "front" else y + int(l_max/2)
        else:
            l_max = rz1 if rz1 < size_max else size_max
            z = z - int(l_max/2) if direction == "bottom" else z + int(l_max/2)

        xx2, yy2, zz2 = get_distances(tree, x, y, z)
        rx2 = get_range(xx2, x, "x")
        ry2 = get_range(yy2, y, "y")
        rz2 = get_range(zz2, z, "z")

        rx = min(rx1, rx2) if min(rx1, rx2) >= size_min else max(rx1, rx2)
        ry = min(ry1, ry2) if min(ry1, ry2) >= size_min else max(ry1, ry2)
        rz = min(rz1, rz2) if min(rz1, rz2) >= size_min else max(rz1, rz2)

        if rx % 2 != 0:
            rx -= 1
        if ry % 2 != 0:
            ry -= 1
        if rz % 2 != 0:
            rz -= 1

        rx = size_min if rx < size_min else rx
        ry = size_min if ry < size_min else ry
        rz = size_min if rz < size_min else rz

        w_max = rx if rx < size_max else size_max
        d_max = ry if ry < size_max else size_max
        h_max = rz if rz < size_max else size_max

        return w_max, d_max, h_max

    if direction is None:
        if shape == "cylinder":
            rot = random.choice(rotations())
        else:
            rot = (0, 0, 0)
        w, d, h = [random.randrange(2*size_min, size_max+size_step, size_step) for _ in ["w", "d", "h"]]
        if shape == "cylinder":
            if rotations(rot) == "x":
                dd = min(w, d)
                w, d, h = dd, dd, h
            elif rotations(rot) == "y":
                dd = min(w, h)
                w, d, h = dd, d, dd
            else:
                dd = min(d, h)
                w, d, h = w, dd, dd
        size = (w, d, h)
        pos = (0, 0, 0)
    else:
        dim_inside = get_inside()
        dim_outside = get_outside()
        base_shape = base.children[list(base.children.keys()).pop()].node_type()
        base_rot = tuple([int(round(r)) for r in base.r.as_euler("xyz", degrees=True)])
        rot = random.choice(rotations())

        if op == "intersection":
            sys.exit("get_random_params() not implemented for intersection")

        if (base_shape == "box" and (shape == "box" or (shape == "cylinder" and (rotations(rot) == "x" and direction in directions("z")) or (rotations(rot) == "y" and direction in directions("y")) or (rotations(rot) == "z" and direction in directions("x"))))) or (base_shape == "cylinder" and (rotations(base_rot) == "x" and direction in directions("z") and (shape == "box" or (shape == "cylinder" and rotations(rot) == "x"))) or (rotations(base_rot) == "y" and direction in directions("y") and (shape == "box" or (shape == "cylinder" and rotations(rot) == "y"))) or (rotations(base_rot) == "z" and direction in directions("x") and (shape == "box" or (shape == "cylinder" and rotations(rot) == "x")))):

            if op == "union" or op is None:
                w_max, d_max, h_max = dim_outside
            else:
                w_max, d_max, h_max = dim_inside

            w = random.randrange(size_min, w_max+size_step, size_step)
            d = random.randrange(size_min, d_max+size_step, size_step)
            h = random.randrange(size_min, h_max+size_step, size_step)

            if shape == "box":
                rot = (0, 0, 0)
            else:
                if rotations(rot) == "x":
                    dd = min(w, d)
                    w, d, h = dd, dd, h
                elif rotations(rot) == "y":
                    dd = min(w, h)
                    w, d, h = dd, d, dd
                else:
                    dd = min(d, h)
                    w, d, h = w, dd, dd

            size = (w, d, h)

            if op == "union" or op is None:
                pos = get_pos_new_outside(w, d, h)
            else:
                pos = get_pos_new_inside(w, d, h)

        else:

            w_base, d_base, h_base = get_dim_base()
            w_out, d_out, h_out = dim_outside
            w_in, d_in, h_in = dim_inside
            w_max, d_max, h_max = w_out, d_out, h_out

            if direction in directions("x"):
                w_max += w_in
            elif direction in directions("y"):
                d_max += d_in
            else:
                h_max += h_in

            if shape == "box" or shape == "cylinder":
                w = random.randrange(size_min, w_max+size_step, size_step)
                d = random.randrange(size_min, d_max+size_step, size_step)
                h = random.randrange(size_min, h_max+size_step, size_step)
                if shape == "cylinder":
                    if rotations(rot) == "x":
                        dd = min(w, d)
                        w, d, h = dd, dd, h
                    elif rotations(rot) == "y":
                        dd = min(w, h)
                        w, d, h = dd, d, dd
                    else:
                        dd = min(d, h)
                        w, d, h = w, dd, dd
            else:
                l_max = min(w_max, d_max, h_max)
                l = random.randrange(size_min, l_max+size_step, size_step)
                w = l
                d = l
                h = l

            lx, ly, lz = get_pos_base()
            ux, uy, uz = get_pos_base_surface()

            if direction in directions("x"):
                assert w % 2 == 0
                if w_base > w:
                    lx = ux - int(w/2) if direction == "right" else ux + int(w/2)
                if w > w_out:
                    ux = ux - (w - w_out) if direction == "right" else ux + (w - w_out)
                try:
                    if op == "union" or op is None:
                        x = random.randrange(lx, ux+1) if direction == "right" else random.randrange(ux, lx+1)
                    else:
                        x = random.randrange(lx, ux) if direction == "right" else random.randrange(ux+1, lx+1)
                except ValueError:
                    x = lx if direction in directions("pos") else ux
                x = x + int(w/2) if direction == "right" else x - int(w/2)
                y = ly
                z = lz
            elif direction in directions("y"):
                assert d % 2 == 0
                if d_base > d:
                    ly = uy - int(d/2) if direction == "back" else uy + int(d/2)
                if d > d_out:
                    uy = uy - (d - d_out) if direction == "back" else uy + (d - d_out)
                x = lx
                try:
                    if op == "union" or op is None:
                        y = random.randrange(ly, uy+1) if direction == "back" else random.randrange(uy, ly+1)
                    else:
                        y = random.randrange(ly, uy) if direction == "back" else random.randrange(uy+1, ly+1)
                except ValueError:
                    y = ly if direction in directions("pos") else uy
                y = y + int(d/2) if direction == "back" else y - int(d/2)
                z = lz
            else:
                assert h % 2 == 0
                if h_base > h:
                    lz = uz - int(h/2) if direction == "top" else uz + int(h/2)
                if h > h_out:
                    uz = uz - (h - h_out) if direction == "top" else uz + (h - h_out)
                x = lx
                y = ly
                try:
                    if op == "union" or op is None:
                        z = random.randrange(lz, uz+1) if direction == "top" else random.randrange(uz, lz+1)
                    else:
                        z = random.randrange(lz, uz) if direction == "top" else random.randrange(uz+1, lz+1)
                except ValueError:
                    z = lz if direction in directions("pos") else uz
                z = z + int(h/2) if direction == "top" else z - int(h/2)

            if shape == "box" or shape == "sphere":
                rot = (0, 0, 0)

            size = (w, d, h)
            pos = (x, y, z)

    return size, pos, rot

def update_tree(tree, op, shape, size, pos, rot):
    def get_name():
        return str(uuid.uuid4())

    width, depth, height = size

    if shape == "box":
        child = Box(size=size, name=get_name())
    elif shape == "cylinder":
        if rot == rotations("x"):
            assert width == depth
            radius = int(width/2)
            height = height
        elif rot == rotations("y"):
            assert width == height
            radius = int(width/2)
            height = depth
        else:
            assert depth == height
            radius = int(depth/2)
            height = width
        child = Cylinder(radius=radius, height=height, name=get_name())
    elif shape == "sphere":
        radius = int(min(width, depth, height)/2)
        child = Sphere(radius=radius, name=get_name())
    else:
        sys.exit("shape '{}' not implemented in update_tree()".format(shape))

    subtree = Pose(pos=pos, rot=rot, children=[child], name=get_name())

    if tree is None:
        tree = subtree
    else:
        if op == "union":
            tree = Union(name=get_name(), children=[tree, subtree])
        elif op == "intersection":
            tree = Intersection(name=get_name(), children=[tree, subtree])
        elif op == "difference":
            tree = Difference(name=get_name(), children=[tree, subtree])
        else:
            sys.exit("unknown operation")

    return tree, subtree

def update_joints(joints, old_joint, tree, subtree, pos, op, direction, ops, size_min):

    def get_axis(d, xx, yy, zz):
        x, y, z = pos
        if d in directions("x"):
            ax, p = xx, x
        elif d in directions("y"):
            ax, p = yy, y
        else:
            ax, p = zz, z
        return p, ax

    def union_possible(d, xx, yy, zz):
        if "union" not in ops:
            return False

        p, ax = get_axis(d, xx, yy, zz)
        p += 32
        count = 0
        if d in directions("neg"):
            while p >= 0 and ax[p] <= 0:
                p -= 1
            while p >= 0 and ax[p] > 0:
                p -= 1
                count += 1
        else:
            while p <= 64 and ax[p] <= 0:
                p += 1
            while p <= 64 and ax[p] > 0:
                p += 1
                count += 1

        if count >= size_min:
            return True

        return False

    def diff_possible(d, xx, yy, zz):
        def get_dim(base):
            key = list(base.children.keys()).pop()
            primitive = base.children[key]
            shape = primitive.node_type()
            if shape == "box":
                width, depth, height = primitive.size
            elif shape == "cylinder":
                rot = tuple([int(round(r)) for r in base.r.as_euler("xyz", degrees=True)])
                if rotations(rot) == "x":
                    width, depth, height = primitive.radius*2, primitive.radius*2, primitive.height
                elif rotations(rot) == "y":
                    width, depth, height = primitive.radius*2, primitive.height, primitive.radius*2
                elif rotations(rot) == "z":
                    width, depth, height = primitive.height, primitive.radius*2, primitive.radius*2
                else:
                    sys.exit("unknown rotation in get_dim_base()")
            elif shape == "sphere":
                width, depth, height = primitive.radius*2, primitive.radius*2, primitive.radius*2
            else:
                sys.exit("error: unknown shape")
            return width, depth, height
        if "difference" not in ops:
            return False

        if d != direction:
            if old_joint is None:
                width, depth, height = get_dim(tree)
            else:
                width, depth, height = get_dim(old_joint["base"])

            if width < 2 * size_min or depth < 2 * size_min or height < 2 * size_min:
                return False

            if d in directions("x"):
                if width >= 2 * size_min:
                    return True
            if d in directions("y"):
                if depth >= 2 * size_min:
                    return True
            if d in directions("z"):
                if height >= 2 * size_min:
                    return True
            return False

        return False

    def inter_possible(d, xx, yy, zz):
        if "intersection" in ops:
            sys.exit("update_joints() not implemented for intersection")
        return False

    if old_joint is not None:
        del old_joint[direction]
        if len(list(old_joint.keys())) <= 1:
            joints.remove(old_joint)

    if op == "union" or op is None:
        new_joint = {"base": subtree}

        xx, yy, zz = get_distances(tree, *pos)

        for d in directions():
            union = union_possible(d, xx, yy, zz)
            diff = diff_possible(d, xx, yy, zz)
            inter = inter_possible(d, xx, yy, zz)
            if union or diff or inter:
                new_joint[d] = (union, diff, inter)

        if len(list(new_joint.keys())) > 1:
            joints.append(new_joint)

    return joints

def update_n_primitives(n_primitives, shape):
    n_primitives -= 1
    return n_primitives
