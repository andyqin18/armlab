"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

import numpy as np
# expm is a matrix exponential function
from scipy.linalg import expm

"""
dh_params = np.array([[0, 0, 0, 0],
                      [0, -np.arctan(0.25), 103.91, 0],
                      [0, -np.arctan(4), 205.73, 0],
                      [0, 0, 200, 0],
                      [0, 0, 131, 0]
                      ])
"""
                      
def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle

def translate(axis, length):
    mat = np.eye(4)
    d = {"x": 0,
         "y": 1,
         "z": 2}

    mat[d[axis], 3] = length
    return mat

# print(translate("x",2))

def rotate(axis, angle):
    mat = np.eye(4)
    if axis == "x":
        mat[1:3, 1:3] = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    elif axis == "y":
        mat[:3, :3] = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    elif axis == "z":
        mat[:2, :2] = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return mat

# print(rotate("x", np.pi/6))


def FK_dh(dh_params, joint_angles, link):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

                TODO: implement this function

                Calculate forward kinematics for rexarm using DH convention

                return a transformation matrix representing the pose of the desired link

                note: phi is the euler angle about the y-axis in the base frame

    @param      dh_params     The dh parameters as a 2D list each row represents a link and has the format [a, alpha, d,
                              theta]
    @param      joint_angles  The joint angles of the links
    @param      link          The link to transform from

    @return     a transformation matrix representing the pose of the desired link
    """
    mat = np.eye(4)

    dh_params[0][3] += joint_angles[0]
    dh_params[1][1] -= joint_angles[1]
    dh_params[2][1] -= joint_angles[2]
    dh_params[3][1] -= joint_angles[3]
    dh_params[4][3] -= joint_angles[4]

    for i in range(link, 5):
        mat = mat @ get_transform_from_dh(dh_params[i][0], dh_params[i][1], dh_params[i][2], dh_params[i][3])

    return mat


def get_transform_from_dh(a, alpha, d, theta):
    """!
    @brief      Gets the transformation matrix T from dh parameters.

    TODO: Find the T matrix from a row of a DH table

    @param      a      a meters
    @param      alpha  alpha radians
    @param      d      d meters
    @param      theta  theta radians

    @return     The 4x4 transformation matrix.
    """

    return rotate("z", theta) @ translate("z", d) @ translate("x", a) @ rotate("x", alpha)

# EE = FK_dh(dh_params, [0,0,0,0,0], 0)
# print(EE)

def get_euler_angles_from_T(T):
    """!
    @brief      Gets the euler angles from a transformation matrix.

                TODO: Implement this function return the 3 Euler angles from a 4x4 transformation matrix T
                If you like, add an argument to specify the Euler angles used (xyx, zyz, etc.)

    @param      T     transformation matrix

    @return     The euler angles from T.
    """

    # This function is for ZYZ
    r13, r31 = T[0,2], T[2,0]
    r23, r32= T[1,2], T[2,1]
    r33 = T[2,2]
    angles = np.zeros((2,3))
    if r13 != 0 or r23 != 0:
        theta = np.arctan2(np.sqrt(1 - r33 ** 2), r33)
        phi = np.arctan2(r23, r13)
        psi = np.arctan2(r32, -r31)
        angles[0,:] = np.array([phi, theta, psi])
        theta = np.arctan2(-np.sqrt(1 - r33 ** 2), r33)
        phi = np.arctan2(-r23, -r13)
        psi = np.arctan2(-r32, r31)
        angles[1,:] = np.array([phi, theta, psi])
    return angles

# print(get_euler_angles_from_T(EE))

def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the 6DOF pose vector from a 4x4 transformation matrix T

    @param      T     transformation matrix

    @return     The pose vector from T.
    """
    pose = np.zeros((2,6))
    angles = get_euler_angles_from_T(T)
    if np.any(angles):
        pose[:,3:6] = angles
        pose[:,:3] = np.tile(T[:3,3], (2,1))
    return pose

# print(get_pose_from_T(EE))

def FK_pox(joint_angles, m_mat, s_lst):
    """!
    @brief      Get a  representing the pose of the desired link

                TODO: implement this function, Calculate forward kinematics for rexarm using product of exponential
                formulation return a 4x4 homogeneous matrix representing the pose of the desired link

    @param      joint_angles  The joint angles
                m_mat         The M matrix
                s_lst         List of screw vectors

    @return     a 4x4 homogeneous matrix representing the pose of the desired link
    """
    pass

def vec_to_skew(p):
    """
    transforms a 3D vector into a skew symmetric matrix
    """
    return np.array([[0., -p[2], p[1]],
                     [p[2], 0., -p[0]],
                     [-p[1], p[0], 0]])

def to_s_matrix(w, v):
    """!
    @brief      Convert to s matrix.

    TODO: implement this function
    Find the [s] matrix for the POX method e^([s]*theta)

    @param      w     { parameter_description }
    @param      v     { parameter_description }

    @return     { description_of_the_return_value }
    """
    mat = np.eye(4)
    mat[:3, :3] = vec_to_skew(w)
    mat[:3, 3] = v
    return mat

# print(to_s_matrix(np.array([0,1,2]), np.array([2,1,0])))

def IK_geometric(dh_params, pose):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose vector as np.array to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose vector as np.array 

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    pass