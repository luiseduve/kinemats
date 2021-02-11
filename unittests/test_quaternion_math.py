#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Luis Quintero | luiseduve@hotmail.com
# Created Date: 2021/01/08
# =============================================================================
"""
Unit tests for quaternion_math.py
"""
# =============================================================================
# Imports
# =============================================================================

import unittest
import numpy as np

from ..quaternion_math import *

# =============================================================================
# Main
# =============================================================================

class TestQuaternionMath(unittest.TestCase):
    def test_quaternion_to_euler(self):
        dataset = np.zeros((1,1,4))
        dataset[0,0,:] = np.array([ 0.07402912, -0.04800714,  0.06036643, -0.99426903])

        # Proof of functions that convert quaternion to euler
        tseries = 0
        tstamp = 0
        
        np.testing.assert_array_almost_equal(quaternion_to_euler(dataset[tseries,tstamp,:]),
                                            np.array([-2.98740064, -0.08663461, -0.1279765]),
                                            decimal=5)

        # quat = np.array([0.86169383, 0.02081877, -0.5058515, 0.03412598])
        # np.testing.assert_array_almost_equal(quaternion_to_euler(quat, degrees=True),
        #                                        [0.15911653941132517, -60.832556785346696, -9.335093630495875])

        # Basic rotations 90 degrees around Z,Y,X
        print("\n::PROOF BASIC ROTATIONS::")
        quat = np.array([np.cos(np.pi/4) * 1, 0, 0, np.sin(np.pi/4) * 1])
        np.testing.assert_array_almost_equal(quaternion_to_euler(quat, degrees=True),
                                            np.array([90,0,0]) ) # Rotate pi/4 around z axis
        quat = np.array([np.cos(np.pi/4) * 1, 0, np.sin(np.pi/4) * 1, 0])
        # Generate GIMBAL LOCK
        np.testing.assert_array_almost_equal(quaternion_to_euler(quat, degrees=True),
                                           np.array([180,90,180])) # Rotate pi/4 around y axis # CAUSES GIMBAL LOCK!
        quat = np.array([np.cos(np.pi/4) * 1, np.sin(np.pi/4) * 1, 0, 0])
        np.testing.assert_array_almost_equal(quaternion_to_euler(quat, degrees=True),
                                           np.array([0,0,90])) # Rotate pi/4 around x axis