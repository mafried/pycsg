import numpy as np
from scipy.spatial.transform import Rotation

from pycsg.csg_node import CSGNode, register_node_type


class Pose(CSGNode):
    def __init__(self, pos=[0.0, 0.0, 0.0], rot=[0.0, 0.0, 0.0], children=[], name='', degrees=True):
        super().__init__(name, children)

        self.degrees = degrees

        # translate
        self.t = np.array([
            [1.0, 0.0, 0.0, pos[0]],
            [0.0, 1.0, 0.0, pos[1]],
            [0.0, 0.0, 1.0, pos[2]],
            [0.0, 0.0, 0.0, 1.0]
        ])

        # rotate
        self.r = Rotation.from_euler('xyz', rot, degrees=self.degrees)
        r = self.r.as_matrix()
        r = np.append(r, [[0.0, 0.0, 0.0]], axis=0)
        r = np.append(r, [[0.0], [0.0], [0.0], [1.0]], axis=1)

        # get pose from translate and rotate
        self.pose = np.matmul(self.t, r)
        self.inv_pose = np.linalg.inv(self.pose)

    def signed_distance(self, p):

        # is only applied to first child
        n = list(self.children.values())[0]

        # fill to have 4d vectors.
        if p.shape[1] < 4:
            p = np.concatenate((p, np.ones((p.shape[0], 4 - p.shape[1]))), axis=1)

        return n.signed_distance(np.matmul(self.inv_pose, p.transpose()).transpose()[:,:3]) if len(self.children) > 0 else 0.0

    def to_dict(self):
        d = super().to_dict().copy()
        d['pos'] = [self.t[0, 3], self.t[1, 3], self.t[2, 3]]
        d['rot'] = list(self.r.as_euler('xyz', degrees=self.degrees))
        return d

    @staticmethod
    def from_dict(d, children):
        return Pose(d['pos'], d['rot'], children, d['name'])

    @staticmethod
    def node_type():
        return 'pose'


register_node_type(Pose)
