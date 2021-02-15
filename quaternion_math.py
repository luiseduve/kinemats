#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Luis Quintero | luiseduve@hotmail.com
# Created Date: 2021/01/08
# =============================================================================
"""
Functions to do Quaternion Analysis

These methods assume that the quaternion representation is arranged as:
    [qw, qi, qj, qk]

In addition, the euler representation assumes that the coordinate system is:
    front=j, left=j, up=k.
To change this coordinate system, use the parameters `axis_front`,`axis_left`,
and `axis_up`.

"""
# =============================================================================
# Imports
# =============================================================================

import numpy as np
import math

# =============================================================================
# Main
# =============================================================================


def slerp(v0, v1, t_array):
    """Spherical linear interpolation.
    From Wikipedia: https://en.wikipedia.org/wiki/Slerp"""

    # >>> slerp([1,0,0,0], [0,0,0,1], np.arange(0, 1, 0.001))

    t_array = np.array(t_array)
    v0 = np.array(v0)
    v1 = np.array(v1)

    # Compute the cosine of the angle between the two vectors.
    dot = np.sum(v0 * v1)
    
    # If the dot product is negative, slerp won't take
    # the shorter path. Note that v1 and -v1 are equivalent when
    # the negation is applied to all four components. Fix by 
    # reversing one quaternion

    if dot < 0.0:
        v1 = -v1
        dot = -dot
    
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        # If the inputs are too close for comfort, linearly interpolate
        # and normalize the result.
        result = v0[np.newaxis,:] + t_array[:,np.newaxis] * (v1 - v0)[np.newaxis,:]
        return (result.T / np.linalg.norm(result, axis=1)).T
    
    # Since dot is in range [0, DOT_THRESHOLD], acos is safe
    theta_0 = np.arccos(dot)        # theta_0 = angle between input vectors
    sin_theta_0 = np.sin(theta_0)   # compute this value only once

    theta = theta_0 * t_array       # theta = angle between v0 and result
    sin_theta = np.sin(theta)       # compute this value only once
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0  # == sin(theta_0 - theta) / sin(theta_0)
    s1 = sin_theta / sin_theta_0

    result = (s0[:,np.newaxis] * v0[np.newaxis,:]) + (s1[:,np.newaxis] * v1[np.newaxis,:])
    return result


# METHODS WITH PYTHON LISTS

def point_rotation_by_quaternion(point:list, q:list):
    """
    Rotate the point `p` (List with 3 or 4 elements)
    with the rotation specified by quaternion `q` (list
    with 4 elements)
    """
    r = point
    if(len(point)==3):
        r = [0]+point
    q_conj = [q[0],-1*q[1],-1*q[2],-1*q[3]]

    result = quaternion_mult(quaternion_mult(q,r),q_conj)
    if(len(point)==3):
        result = result[1:]
    return result

def quaternion_to_euler_list(quaternion:list, degrees = False, axis_qw=0, axis_front=1, axis_left=2, axis_up=3):
    """
    Returns the euler representation of a quaternion [qw, qi, qj, qk] into
        yaw     = [-pi, pi]
        pitch   = [-pi/2, pi/2]
        roll    = [-pi, pi]

    If the quaternion is not in the order [qw, qi, qj, qk], you can specify these variables to set the index of the dimension:
        - `axis_qw`     > Contains the scalar factor of the quaternion
        - `axis_front`  > Contains the axis indicating the POSITIVE-FRONT in the coordinate reference system
        - `axis_left`   > Contains the axis indicating the POSITIVE-LEFT in the coordinate reference system
        - `axis_up`     > Contains the axis indicating the POSITIVE-UP in the coordinate reference system

    Based on Eq 2.9 of the Technical report in: 
    https://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=08A583E84796E221D446200475B7841A?doi=10.1.1.468.5407&rep=rep1&type=pdf

    :param quaternion: Unit quaternion representation
    :type quaternion: list of length 4
    :return: [yaw, pitch, roll] in radians, or degrees if `degrees=True`
    :rtype: list of length 3
    """

    qr = quaternion[axis_qw]
    qi= quaternion[axis_front]
    qj = quaternion[axis_left]
    qk = quaternion[axis_up]

    # Squares of vector components to speed-up calculations
    sqi = qi*qi
    sqj = qj*qj
    sqk = qk*qk

    ### Euler angles
    # roll (x-axis rotation)
    sinr_cosp = 2 * (qr * qi + qj * qk)
    cosr_cosp = 1 - 2 * (sqi + sqj)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (qr * qj - qk * qi)
    # Evaluates singularity at +/- 90 degrees to prevent GIMBAL LOCK. Happening at north/south pole.
    pitch = 0
    if (abs(sinp) >= 1):
        print("GIMBAL LOCK!! with quaternion", quaternion)
        pitch = math.copysign(math.pi / 2, sinp) # use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (qr * qk + qi * qj)
    cosy_cosp = 1 - 2 * (sqj + sqk)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    # Transform from radians to degrees
    factor = 1
    if(degrees): ## CONVERT FROM RADIANS TO DEGREES
        factor = 180.0 / math.pi

    return [yaw*factor, pitch*factor, roll*factor] # In rotation = [phi, theta, psi]



### QUATERNION WITH NDARRAYS

def quaternion_conjugate(quaternion:np.ndarray):
    """
    Given an array with 4 elements (quaternions) in the last dimension,
    returns the complex conjugate, inverting the sign of the last
    three components.
    E.g. q = [1, 0.3, -0.3, 0.4], quaternion_conjugate(q)
         returns array([1, -0.3, 0.3, -0.4])
    """
    if quaternion.shape[quaternion.ndim-1] != 4:
        print('quaternion needs to have 4 elements (unit quaternions) in the last dimension')
        return None
    # Complex conjugate (inverting non-real dimensions)
    q_conjugate = quaternion.copy()
    q_conjugate[...,1:4] = -1 * q_conjugate[...,1:4]
    return q_conjugate

######### ROTATION
def quaternion_mult(p, q):
    """
    Hamilton Product of two quaternions or arrays with quaternions
    Multiplication of the quaternions `p` and `q`, to calculate the 
    composite rotation, when `p` is applied first and then `q`.
    Equation 7.1 in book 1999, Kuipers, Quaternions and Rotation Sequences.
    """

    if(not isinstance(p,np.ndarray)):
        p = np.array(p)
    if(not isinstance(q,np.ndarray)):
        q = np.array(q)

    result = None
    if(p.shape == q.shape):
        # Both arrays are the same because is the second stage of multiplication
        result = np.zeros(p.shape)
    elif (p.ndim == 1 and p.size == 4) and (q.shape[q.ndim]==4):
        # The 1D quaternion `p` is rotated by the sequence of quaternions in `q`
        result = np.zeros(q.shape)
    elif (q.ndim == 1 and q.size == 4) and (p.shape[p.ndim]==4):
        # The 1D quaternion `q` is rotated by the sequence of quaternions in `p`
        result = np.zeros(p.shape)
    else:
        print('Both arrays should be equal length or one of them shape (4,) and the other (...,4')
        return None

    # Quaternion multiplication
    result[...,0] = p[...,0]*q[...,0] - p[...,1]*q[...,1] - p[...,2]*q[...,2] - p[...,3]*q[...,3]
    result[...,1] = p[...,0]*q[...,1] + p[...,1]*q[...,0] + p[...,2]*q[...,3] - p[...,3]*q[...,2]
    result[...,2] = p[...,0]*q[...,2] - p[...,1]*q[...,3] + p[...,2]*q[...,0] + p[...,3]*q[...,1]
    result[...,3] = p[...,0]*q[...,3] + p[...,1]*q[...,2] - p[...,2]*q[...,1] + p[...,3]*q[...,0]

    return result

def quaternion_mult_reverse(q,p):
    """
    Used to pass the parameters in the order q,p; but
    perform the multiplication in the order p,q.
    Might be utilized when broadcasting np.apply_along_axis(),
    requiring specific order of args.
    """
    return quaternion_mult(p,q)


def apply_quaternion_sequence(reference_point:np.ndarray, quaternions_array:np.ndarray):
    """
    Apply a sequence of rotations defined in `quaternions_array` to the 
    1D quaternion in `reference_point`.
    :param reference_point: 1D array with 3 or 4 elements containing the reference point.
        If there are 3 elements, it will be expanded as a quaternion as [0, px, py, pz]
    :type reference_point: numpy.ndarray
    :param quaternions_array: Multi-dimensional array which number of elements in the last dimension is 4, 
    corresponding to the set of unit quaternions that transform the `reference_point`
    :type quaternion: numpy.ndarray
    :return: An array of the same dimension than `quaternions_array` representing the rotated version
                of `reference_point` for each position.
    :rtype: numpy.ndarray
    """

    reference_point = np.array(reference_point)
    quaternion_to_rotate = reference_point

    reference_is_3D = False    # Whether the reference point is a 3D point, not quaternion
    # SETUP CONDITIONS
    if quaternion_to_rotate.ndim != 1:
        print('quaternion_to_rotate should be 1D.')
        return None
    elif quaternion_to_rotate.size == 3:
        reference_is_3D = True
        quaternion_to_rotate = np.zeros((4))
        quaternion_to_rotate[1:4] = reference_point

    if quaternions_array.shape[quaternions_array.ndim-1] != 4:
        print('quaternions_array needs to have 4 elements (unit quaternions) in the last dimension')
        return None

    # PROCESS
    n_dim = quaternions_array.ndim

    # Store result of multiplication q * v * q_conjugate
    # Performs a half-rotation indicated by the quaternion over the last (n_dim-1) dimension
    half_rotation = np.apply_along_axis(quaternion_mult, n_dim-1,\
                                        quaternions_array, quaternion_to_rotate)

    # Take the half rotations and multiply for the conjugate of the rotation sequence
    # Complex conjugate (inverting non-real dimensions)
    q_conjugate = quaternion_conjugate(quaternions_array)
    result = quaternion_mult(half_rotation, q_conjugate)

    # Delete the first dimension of quaternion to return an array in 3D
    if(reference_is_3D):
        result = result[...,1:].copy()

    return result

### CONVERT REPRESENTATIONS

def quaternion_to_euler(quaternions:np.ndarray, degrees = False, axis_qw=0, axis_front=1, axis_left=2, axis_up=3):
    """
    Returns the euler representation of a quaternion [qw, qi, qj, qk] into
        yaw     = [-pi, pi]
        pitch   = [-pi/2, pi/2]
        roll    = [-pi, pi]

    If the quaternion is not in the order [qw, qi, qj, qk], you can specify these variables to set the index of the dimension:
        - `axis_qw`     > Contains the scalar factor of the quaternion
        - `axis_front`  > Contains the axis indicating the POSITIVE-FRONT in the coordinate reference system
        - `axis_left`   > Contains the axis indicating the POSITIVE-LEFT in the coordinate reference system
        - `axis_up`     > Contains the axis indicating the POSITIVE-UP in the coordinate reference system

    Based on Eq 2.9 of the Technical report in: 
    https://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=08A583E84796E221D446200475B7841A?doi=10.1.1.468.5407&rep=rep1&type=pdf

    :param quaternion: Array with quaternions in the last dimension
    :type quaternion: numpy.ndarray
    :return: Same array replacing the quaternions with Euler degrees [yaw, pitch, roll] in radians, or degrees if `degrees=True`
    :rtype: numpy.ndarray
    """

    if type(quaternions) in [list]:    
        quaternion_to_euler_list(quaternions, degrees=degrees, axis_front=axis_front, axis_left=axis_left, axis_up=axis_up, axis_qw=axis_qw)

    print(quaternions.shape)

    qr = quaternions[...,axis_qw]
    qi = quaternions[...,axis_front]
    qj = quaternions[...,axis_left]
    qk = quaternions[...,axis_up]

    # Where to store the result
    result = np.empty(quaternions.shape)
    # Delete one column from the last axis (from 4dim to 3dim)
    result = np.delete(result, 3, result.ndim-1) 

    # Squares of vector components to speed-up calculations
    sqi = qi*qi
    sqj = qj*qj
    sqk = qk*qk

    ### Euler angles
    # roll (front-axis rotation)
    sinr_cosp = 2 * (qr * qi + qj * qk)
    cosr_cosp = 1 - 2 * (sqi + sqj)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (left-axis rotation)
    sinp = 2 * (qr * qj - qk * qi)
    pitch = np.arcsin(sinp)
    # Find and correct singularity at +/- 90 degrees to prevent GIMBAL LOCK. Happening at north/south pole.
    gimbal_lock_idx = np.argwhere(np.isnan(pitch))
    for idx in gimbal_lock_idx:
        sliced_idx = tuple(slice(x,x+1) for x in idx)
        pitch[sliced_idx] = np.copysign(np.pi / 2, sinp[sliced_idx]) # use 90 degrees if out of range
        print("GIMBAL LOCK!! pos:", idx, " with quaternion", quaternions[sliced_idx], "replaced by", pitch[sliced_idx])

    # yaw (up-axis rotation)
    siny_cosp = 2 * (qr * qk + qi * qj)
    cosy_cosp = 1 - 2 * (sqj + sqk)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    # Results
    result[...,0] = yaw
    result[...,1] = pitch
    result[...,2] = roll

    # Transform from radians to degrees
    if(degrees): ## CONVERT FROM RADIANS TO DEGREES
        result = np.rad2deg(result)

    return result # In rotation = [phi, theta, psi]


### DISTANCES


def geodesic_distance(quaternion_reference:np.ndarray, quaternions_array:np.ndarray):
    """
    Compares the quaternion in `quaternion_reference` with all the quaternions
    in the array `quaternions_array`, which last dimension should have 4
    elements.

    Returns an array of the same shape than `quaternions_array`, with the geodesic distance
    to each variable
    """
    quaternion = np.array(quaternion_reference)

    reference_is_3D = False    # Whether the reference point is a 3D point, not quaternion
    # SETUP CONDITIONS
    if quaternion.ndim != 1:
        print('quaternion should be 1D.')
        return None
    elif quaternion.size == 3:
        reference_is_3D = True
        quaternion = np.zeros((4))
        quaternion[1:4] = quaternion_reference

    if quaternions_array.shape[quaternions_array.ndim-1] != 4:
        print('quaternions_array needs to have 4 elements (unit quaternions) in the last dimension')
        return None

    result = np.zeros(quaternions_array.shape[0:quaternions_array.ndim-2])

    return result.shape
    """
    p = np.abs(1 - np.sum())
    
    gdistance <- function(x, y) {
    x <- abs(1 - sum((x - y)^2) / 2)
    if (x > 1) x <- 1
    2 * acos(x)
    }

        #' Unit Quaternion Geodesic Distance
    #'
    #' This function computes the geodesic distance between two unit quaternions.
    #'
    #' @param x A length-4 numeric vector of unit norm representing the first
    #'   quaternion.
    #' @param y A length-4 numeric vector of unit norm representing the second
    #'   quaternion.
    #'
    #' @return A positive scalar providing a measure of distance between the two
    #'   input quaternions.
    #'
    #' @export
    
    """



