#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Luis Quintero | luiseduve@hotmail.com
# Created Date: 2021/01/08
# =============================================================================
"""
Functions to do basic time series classification.
"""
# =============================================================================
# Imports
# =============================================================================

import numpy as np
import pandas as pd

from enum import Enum, IntEnum, unique


# =============================================================================
# Enum
# =============================================================================


@unique
class EnumDistMetrics(IntEnum):
    Euclidean                   = 0     # L2-norm
    EuclideanClamped_EulerAngle = 1     # Clamped Euclidean for Euler angles
    EuclideanClamped_Quat       = 2     # Clamped Euclidean for Quaternions
    EuclideanLB_paa             = 3     # Lower Bound for PAA
    EuclideanLB_paa_Euler       = 4     # Lower Bound for PAA + clamped euclidean for Euler
    EuclideanLB_paa_Quat        = 5     # Lower Bound for PAA + clamped eucl
    DtwSumMultiDim              = 6     # DTW over each dimension, and sum error together.

# =============================================================================
# Main
# =============================================================================

def train_test_split(array:np.ndarray, labels:np.ndarray = None, axis=0, training_split=0.7, stratified=False):
    """
    Takes a multidimensional numpy array and returns a 
    random split based on the specified axis.
    
    :param array: Numpy array with [axis] dimension equal to number of
                elements in the labels.
    :type array: numpy.ndarray
    :param labels: 1D numpy array with labels of each time series. Optional
    :type labels: numpy.ndarray
    :param axis: Which axis should array be analyzed
    :type axis: int
    :param training_split: Percentage of the training set [0,1]
    :type training_split: float
    :return: Four arrays with array of time series and indices of split 
            (train, test, idx_train, idx_test). If labels are passed it
            returns the classes instead of the index: (train, test, y_train, y_test)
    :rtype: Tuple of numpy.ndarrays
    """

    # Create array with incremental indices
    NUM_SAMPLES = array.shape[axis]

    if((labels is not None) and (array.shape[axis] != labels.size)):
        raise ValueError('Error: array.shape[axis] should be equal to number of elements in `labels`')

    # Calculate samples samples
    train_size = int(NUM_SAMPLES*training_split)

    # Non-stratified sampling
    indices_array = np.arange(NUM_SAMPLES)
    idx_train = np.random.choice(indices_array, train_size, replace=False)
    # Stratified sampling
    if(stratified and (labels is not None)):
        idx_train = []
        classes = np.unique(labels)
        for c in classes:
            # Do a `training_split` over the samples that belong to class c.
            subset = np.where(labels==c)[0] # Extract positions of class==c
            subset_size = subset.size
            train_size_class = int(subset_size*training_split)
            if(train_size_class < 1): raise ValueError(f"Not enough samples in class {c} to do a split of {training_split}")
            # Do a sampling from the indices that belong to the class
            indices_array_class = np.arange(subset_size)
            idx_train_class = np.random.choice(subset, train_size_class, replace=False)
            # Append to general indices
            idx_train = idx_train + list(idx_train_class)
        # To numpy
        idx_train = np.array(idx_train)
                
    # Substract the sets of original values and training ts
    idx_test = list(set(indices_array) - set(idx_train))
    idx_test = np.array(idx_test)

    # Subset in the specific axis
    idx = [slice(None)]*array.ndim
    idx[axis] = idx_train # Select the original N values in the axis
    train = array[tuple(idx)].copy()

    idx = [slice(None)]*array.ndim
    idx[axis] = idx_test # Select the original N values in the axis
    test = array[tuple(idx)].copy()

    if (labels is not None):
        try:
            return train, test, labels[idx_train], labels[idx_test]
        except Exception as e:
            print(f"Exception while returning labels, returned indices instead. Exception: {e}")
            return train, test, idx_train, idx_test        

    # Return the indices used for train-test split
    return train, test, idx_train, idx_test

def distance_time_series(array, training_array=None, axis=0, axis_dim=2, distance_metric:EnumDistMetrics=EnumDistMetrics.Euclidean):
    """
    Calculates cross-similarity matrix, i.e. distance between each entry of the `array`, comparing it with
    each element from the `training_array` along the specified `axis`.
    
    Both `array` and `training_array` should be equal shape in all dimensions besides
    `axis`. Common convention shape = [idx_ts, time, dimensions], and the axis to iterate
    is `axis=0`

    `axis_dim`, represents the dimension in which the 'channels' of the ts are present, in case of
    multidimensional time series. If the array is 2D, a 3D dimension will be added with shape 1, in the axis=2.

    If training_array=None, it is assumed that all similarity should be calculated in the ts in `array`

    distance_metric:
        - 'euclidean' # Norm of the point-wise difference
        - 'euclidean_eulerangle' # Norm of the point-wise difference clipped between [0,pi]
        - 'euclidean_quat_norm'  # Norm of the point-wise norm of quaternions
        - 'euclidean_quat_dotp'  # Norm of the point-wise dot-product of quaternions
        - 'dtw_sum_multidim'     # DTW on each dimension, then adds together the error per dimension

    Returns a nparray with size (NUM_TST, NUM_TRN), where every ts in the test (row), is compared to
    every ts in the training (column), for the given axis.
    """

    def _eucl(array):
        """
        Receives an array and calculates norm distance.
        """
        return np.linalg.norm(array)

    # Match the time_series_array to 3D even if it has lower dimensions
    dimensions = array.ndim
    if(dimensions==3):
        pass
    elif(dimensions == 2):
        array = np.expand_dims(array, axis=axis_dim)
        dimensions = array.ndim
    else:
        raise ValueError('Invalid number of dimensions. Either 2 or 3 dimensions in the array')

    calculate_crossdist_mat = False
    if(training_array is None):
        training_array = array
        calculate_crossdist_mat = True

    # Check whether it is a square matrix
    NUM_TST = array.shape[axis]
    NUM_TRN = training_array.shape[axis]
    NUM_DIMS = array.shape[axis_dim]        # How many dimensions the TS has
    distance_matrix = np.zeros((NUM_TST,NUM_TRN))
    
    # Extract length of a single time series
    idx = [slice(None)]*dimensions
    idx[axis] = 0
    idx[axis_dim] = 0
    single_ts = array[tuple(idx)]
    TS_LENGTH = single_ts.size

    # Conduct distance calculation
    if(distance_metric == EnumDistMetrics.Euclidean):  ## Pointwise is {a-b}, still multidimensional, then Euclidean Norm
        # Euclidean metric
        for i in range(NUM_TST):
            # Select the original TS in the specified `axis`
            idx = [slice(None)]*dimensions
            idx[axis] = i 

            # Reference TS - Each of the TS in training data
            diff = array[tuple(idx)] - training_array

            ## Concatenate all the dimensions in one single axis (UNIDIMENSIONAL) Because does not summarize.
            diff = diff.reshape(diff.shape[axis], -1)
            # Unbounded interval of distance. Uncomment below to check!!
            # print(f"min:{np.min(diff)}\tto\tmax:{np.max(diff)}|shape{diff.shape}")

            # Apply distance to all TS in training_array # Axis=1 because is the axis to disappear
            distance_matrix[i,:] = np.linalg.norm(diff, axis=1) # After the reshape above, it's always 2D, then it is ok to have axis=1 hardcoded.
    
    elif(distance_metric == EnumDistMetrics.EuclideanClamped_EulerAngle): ## Pointwise is MIN { |a-b|, 2\pi - |a-b| }, then Euclidean Norm
        # Euclidean metric - Euler angle
        for i in range(NUM_TST):
            # Select the original TS in the specified `axis`
            idx = [slice(None)]*dimensions
            idx[axis] = i 

            # Reference TS - Each of the TS in training data
            diff = array[tuple(idx)] - training_array

            # Apply euclidean norm along the euler angles (Axis=2), axis to disappear
            if(diff.ndim == 3):
                diff = np.linalg.norm(diff, axis=2)

            ## MIN between { |a-b|, 2\pi - |a-b| }
            diff = np.abs(diff)
            min_diff = np.minimum(diff, 2*np.pi - diff)
            # Min_diff contains values in the interval [0,pi], which corresponds to
            #    maximum distance between two euler angles. Uncomment below!!
            # print(f"min:{np.min(min_diff)}\tto\tmax:{np.max(min_diff)}|shape{min_diff.shape}")

            # Apply distance to all TS in training_array # Axis=1 because is the axis to disappear
            distance_matrix[i,:] = np.linalg.norm(min_diff, axis=1) #np.apply_along_axis(_eucl, 1, min_diff)
    
    elif(distance_metric == EnumDistMetrics.EuclideanClamped_Quat): ## Pointwise is MIN{ ||p-q||, ||p+q|| }, then Euclidean Norm
        # Euclidean metric - Quaternion
        for i in range(NUM_TST):
            # Select the original TS in the specified `axis`
            idx = [slice(None)]*dimensions
            idx[axis] = i 

            # ||p-q||
            # Reference TS - Each of the TS in training data
            diff = array[tuple(idx)] - training_array
            # Apply euclidean norm along the quaternion (Axis=2) axis to disappear
            if(diff.ndim == 3):
                difference = np.linalg.norm(diff, axis=2) #np.apply_along_axis(_eucl, 2, diff)
            else:
                difference = diff

            # ||p+q||
            # Reference TS - Each of the TS in training data
            add = array[tuple(idx)] + training_array
            # Apply euclidean norm along the quaternion (Axis=2) axis to disappear
            if(diff.ndim == 3):
                addition = np.linalg.norm(add, axis=2)#np.apply_along_axis(_eucl, 2, add)
            else:
                addition = add

            ## Pointwise: MIN { ||p-q||, ||p+q|| }
            min_diff = np.minimum(difference, addition)
            # min_diff contains values in the interval [0,sqrt(2)], or [0,1.4142]. Uncomment below!!
            # print(f"min:{np.min(min_diff)}\tto\tmax:{np.max(min_diff)}|shape{min_diff.shape}")

            # Apply distance to all TS in training_array # Axis=1 because is the axis to disappear
            distance_matrix[i,:] = np.linalg.norm(min_diff, axis=1) #np.apply_along_axis(_eucl, 1, min_diff)
            
    elif (distance_metric == "euclidean_quat_dotp"): ## Pointwise is DOTPRODUCT{p,q}, then Euclidean Norm
        raise ValueError(f"distance_metric={distance_metric} TO BE IMPLEMENTED")
        ### TODO
        # Euclidean metric - Quaternion
        for i in range(NUM_TST):
            # Select the original TS in the specified `axis`
            idx = [slice(None)]*dimensions
            idx[axis] = i 

            # ||p-q||
            # Reference TS - Each of the TS in training data
            diff = array[tuple(idx)] - training_array
            # Apply euclidean norm along the quaternion (Axis=2) axis to disappear
            if(diff.ndim > 1):
                difference = np.linalg.norm(diff, axis=2) #np.apply_along_axis(_eucl, 2, diff)

            # ||p+q||
            # Reference TS - Each of the TS in training data
            add = array[tuple(idx)] + training_array
            # Apply euclidean norm along the quaternion (Axis=2) axis to disappear
            if(diff.ndim > 1):
                addition = np.linalg.norm(add, axis=2)#np.apply_along_axis(_eucl, 2, add)

            ## Pointwise: MIN { ||p-q||, ||p+q|| }
            min_diff = np.minimum(difference, addition)
            # min_diff contains values in the interval [0,sqrt(2)], or [0,1.4142]. Uncomment below!!
            print(f"min:{np.min(min_diff)}\tto\tmax:{np.max(min_diff)}|shape{min_diff.shape}")

            # Apply distance to all TS in training_array # Axis=1 because is the axis to disappear
            distance_matrix[i,:] = np.linalg.norm(min_diff, axis=1) #np.apply_along_axis(_eucl, 1, min_diff)

    elif (distance_metric == EnumDistMetrics.DtwSumMultiDim):

        #from dtw import dtw
        import pyts.metrics

        # DTW setup
        WINDOW_TYPE = "sakoechiba"
        WINDOW_SIZE = int(np.ceil(0.1 * TS_LENGTH))   # Percentage from ts length

        # DTW metric
        # For each TS in the test set
        for i in range(NUM_TST):
            # Compare with each TS in train set

            # Calculating SYMMETRIC cross-dist matrix? Otherwise always compare from 0 to NUM_TRN
            symmetry_idx = i if(calculate_crossdist_mat) else 0

            for j in range(symmetry_idx, NUM_TRN):
                # Iterate per TS dimension
                dist_per_dim = np.zeros( NUM_DIMS ) 
                for k in range(NUM_DIMS):
                    # Select the original TS in the specified `axis`
                    idx_tst = [slice(None)]*dimensions
                    idx_tst[axis] = i 
                    idx_tst[axis_dim] = k
                    test_ts = array[tuple(idx_tst)]

                    # Select the training ts in the specific dim
                    idx_trn = [slice(None)]*dimensions
                    idx_trn[axis] = j
                    idx_trn[axis_dim] = k
                    train_ts = training_array[tuple(idx_trn)]

                    ### CALCULATE DTW IN 1D TIME SERIES
                    """USES PACKAGE: dtw-python"""
                    #alignment = dtw(test_ts, train_ts, keep_internals=False, \
                    #                distance_only=True, window_type="itakura",
                    #                window_args={})
                    #dist_per_dim[k] = alignment.distance
                    """USES PACKAGE: pyts"""
                    alignment = pyts.metrics.dtw(test_ts,train_ts, dist="square", method=WINDOW_TYPE, options={"window_size":WINDOW_SIZE})
                    dist_per_dim[k] = alignment

                ### JOIN DIMENSIONS BY ADDING DISTANCE FROM EACH CHANNEL.
                # Future: Another option is keep the channels and do majority voting.
                distance_matrix[i,j] = np.sum(dist_per_dim)
                if (calculate_crossdist_mat):
                    distance_matrix[j,i] = distance_matrix[i,j]

                # Print progress
                if (j%WINDOW_SIZE==0): print(f"DTW >> TRN:{i+1}/{NUM_TST} vs TST:{j+1}/{NUM_TRN} ")

    else:
        raise ValueError(f"distance_metric={distance_metric} does not exist... Check documentation!")

    return distance_matrix


def classification_metrics_from_conf_matrix(confusion_matrix, suffix_cols=None):
    """ Receives a pandas df with confusion matrix (e.g. resulting from pandas.crosstab),
    and returns a pandas df with relevant metrics
    `suffix_cols` is used to append the text to column names
    """

    # Guarantee that Confusion Matrix is square
    CM = confusion_matrix.copy()
    if(CM.shape[0] != CM.shape[1]):
        # Find the columns that were never predicted and add a zero-valued column
        for c in CM.index.values:
            if c not in CM.columns:
                CM[c] = [0] * CM.index.values.size
        CM = CM[CM.index.values] # Reorder column according to rows

    CM = CM.to_numpy()
    num_classes = len(confusion_matrix.index)

    # Metrics
    acc=np.sum(np.diag(CM))/np.sum(CM)
    accuracy = np.repeat(acc,num_classes)
    precision = np.diag(CM)/np.sum(CM,axis=0)
    recall = np.diag(CM)/np.sum(CM,axis=1)
    
    colnames = ["class_label", "accuracy", "precision", "recall"]
    data_results = [confusion_matrix.index.values, accuracy, precision, recall]
    
    # Change column names
    if suffix_cols is not None:
        colnames = [str(col+"_"+suffix_cols) for col in colnames]
    
    data = {c:v for c,v in zip(colnames,data_results)}

    results = pd.DataFrame(data = data, columns = data.keys())
    return results


def predict_knn(distances_array, train_labels, num_neighbors=1, more_nn=False, returnDataFrame=False):
    """
    Receives `distances_array` as a 2D numpy array with test set elements as rows and elements from training set as cols,
    which can be generated from the function `distance_time_series()`.
    `train_labels` is a numpy array of one dimensions with labels from the training set, equals to distance_array.shape[1]
    Returns an array with shape [num_test_elements, len(num_neighbors)] containing the predicted labels KNN classifiers.

    - `more_nn`: Includes in the calculation a classifier that performs KNN classification with the closest next odd number
        from the number of labels. For example: If the number of labels is 8, will look for 1-nn and 9-nn; because 
        applying 3-nn or 5-nn would not guarantee that there is majority. If the number of labels is 5, then will look for 7-nn.
    - `returnDataFrame`: Returns a formatted pandas df instead of a numpy array.
    """

    if (distances_array.shape[1] != train_labels.shape[0]):
        print("Arrays do not match. train_labels should have the number of columns existing in distances_array")
        return

    if type(num_neighbors) is int:
        num_neighbors = [num_neighbors]

    def _mode1D(a):
        vals, cnts = np.unique(a, return_counts=True)
        return vals[cnts.argmax()], cnts.max()

    # Main analysis
    test_samples = distances_array.shape[0]
    num_classes = np.unique(train_labels).size

    # Row 0 has 1-nn. Row 1 has other-NN
    if (more_nn):
        n_next_odd_from_labels = num_classes + 1 if num_classes%2==0 else num_classes + 2
        num_neighbors = num_neighbors + [n_next_odd_from_labels]
        print(f'Finding more_nn with num_neigh = {n_next_odd_from_labels}, labels = {num_classes}')

    # Placeholder for results
    predicted_label = np.zeros( (test_samples, len(num_neighbors)) )

    for i in range(test_samples):
        # Distance from a time series in the test set to all in the training set
        ts_dist = distances_array[i,:]

        for j,neigh in enumerate(num_neighbors):
            if(neigh == 1):
                ## 1-NN
                # Argmin over all ts. 1-NN
                index_min_dist = ts_dist.argmin(axis=0)
                predicted_label[i,j] = train_labels[index_min_dist]
            else:    
                ## Other-NN
                # Find the indices of the lowest `neigh` distances in the training set
                indices_lowest_dist = np.argpartition(ts_dist,neigh)[0:neigh]
                # Classes of the respective lowest indices
                classes_knn = train_labels[indices_lowest_dist]
                # Mode value among array of labels
                predicted_label[i,j] = _mode1D(classes_knn)[0] # [0] returns val, [1] count

    if(returnDataFrame):
        # Format result as dataframe
        dict_data = { str(f"{neigh}-NN"): predicted_label[:,i] for i,neigh in enumerate(num_neighbors) }
        predicted_label_df = pd.DataFrame(dict_data, dtype=int)
        return predicted_label_df
    else:
        # Bring back 1-D in case there is only one dimension to be calculated.
        if(len(num_neighbors) == 1):
            predicted_label = predicted_label.reshape(predicted_label.size)
        return predicted_label


if __name__ == "__main__":
    print(f"Functions for {__file__}")
    print(f"Usage: Help")