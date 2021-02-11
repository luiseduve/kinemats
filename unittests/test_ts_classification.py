#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Luis Quintero | luiseduve@hotmail.com
# Created Date: 2021/01/08
# =============================================================================
"""
Unit tests for ts_classification.py
"""
# =============================================================================
# Imports
# =============================================================================

import unittest
import numpy as np

from ..ts_classification import *

# =============================================================================
# Main
# =============================================================================

class Test_TsClassification(unittest.TestCase):
    def test_train_test_split_multidim(self):
        
        # Random array to do tests
        array = np.random.rand(30,20,5)
        train_data, test_data, trn_idx, tst_idx = train_test_split(array, axis=0, training_split=0.7)

        # Split when multidimensional data
        self.assertEqual(train_data.shape, (21,20,5))
        self.assertEqual(test_data.shape, (9,20,5))
        self.assertEqual(trn_idx.size, 21)
        self.assertEqual(tst_idx.size, 9)

    def test_train_test_split_1D(self):
        
        # Random array to do tests 1-dimension
        array = np.random.rand(100)
        train_data, test_data, trn_idx, tst_idx = train_test_split(array, axis=0, training_split=0.8)

        # Split when multidimensional data
        self.assertEqual(train_data.shape, (80,))
        self.assertEqual(test_data.shape, (20,))
        self.assertEqual(trn_idx.size, 80)
        self.assertEqual(tst_idx.size, 20)
    
    def test_train_test_split_labels(self):
        
        # Random array to do tests
        array = np.random.rand(30,20,5)
        labels = np.array([0]*15 + [1]*15)
        train_data, test_data, trn_idx, tst_idx = train_test_split(array, labels=labels, axis=0, training_split=0.7)

        # Split when multidimensional data
        self.assertEqual(train_data.shape, (21,20,5))
        self.assertEqual(test_data.shape, (9,20,5))
        np.testing.assert_array_equal(set(trn_idx), set([0,1]))
        self.assertEqual(tst_idx.size, 9)

    def test_labels_size(self):
        array = np.random.rand(30,20,5)
        # Wrong number of elements in the elements
        labels = np.array([0]*12 + [1]*15) # 12 should be 15 if is correct
        self.assertRaises(ValueError, train_test_split, array, labels=labels)


    def test_distance_measure_euclidean(self):
        array = np.random.rand(30,200,4) # Simulate time-series [idx,sample,quaternion]
        # Distance measure between all TS of the array should create a diagonal-zero
        diff = distance_time_series(array, array, axis=0, distance_metric='euclidean')
        self.assertEqual(np.sum(np.diag(diff)), 0)

if __name__ == '__main__':
    unittest.main()