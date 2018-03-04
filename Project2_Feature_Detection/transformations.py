import math
import numpy as np


def get_rot_mx(angle):
    '''
    Input:
        angle_x -- Rotation around the x axis in radians
        angle_y -- Rotation around the y axis in radians
        angle_z -- Rotation around the z axis in radians
    Output:
        A 4x4 numpy array representing 3D rotations. The order of the rotation
        axes from first to last is x, y, z, if you multiply with the resulting
        rotation matrix from left.
    '''
    # Note: For MOPS, you need to use angle_z only, since we are in 2D

    rot_mx = np.array([[math.cos(angle), -math.sin(angle), 0],
                         [math.sin(angle), math.cos(angle), 0],
                         [0, 0, 1]])

    return rot_mx


def get_trans_mx(trans_vec):
    '''
    Input:
        trans_vec -- Translation vector represented by an 1D numpy array with 2
        elements
    Output:
        A 4x4 numpy array representing 3D translation.
    '''
    assert trans_vec.ndim == 1
    assert trans_vec.shape[0] == 2

    trans_mx = np.eye(3)
    trans_mx[:2, 2] = trans_vec

    return trans_mx


def get_scale_mx(s):
    '''
    Input:
        s_x -- Scaling along the x axis
        s_y -- Scaling along the y axis
        s_z -- Scaling along the z axis
    Output:
        A 4x4 numpy array representing 3D scaling.
    '''
    # Note: For MOPS, you need to use s_x and s_y only, since we are in 2D

    scale_mx = np.eye(3)

    for i, s in range(2):
        scale_mx[i, i] = s

    return scale_mx

