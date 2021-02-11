import numpy as np

import string
from scipy.stats import norm

"""
Some time-series preprocessing
NOTE: THIS FUNCTIONS ASSUME TIME SERIES OF EQUAL LENGTH
"""

def z_normalization(array, axis=0):
    """ Applies z-normalization to input array as (array - mean)/std
    """
    array = np.array(array)
    m = np.mean(array, axis = axis, keepdims = True)
    sd = np.std(array, axis = axis, keepdims = True)

    # Replace nan with zero, occurs when sd is zero.
    result = np.nan_to_num((array-m)/sd)
    return result

def pad_to_power_of_2(array, axis=1, pad_value=0):
    """ Returns an array with the length in the specified axis
    divisible by 2.
    E.g. Used for transform
    """
    if(np.log2(array.shape[axis])%2 == 0): # If div by 2, no padding needed
        return array
    
    # If array is different, pad it with `pad_value`
    new_size = list(array.shape)
    new_size[axis] = int(np.power(2,np.ceil(np.log2(array.shape[axis]))))
    new_array = pad_value*np.ones(new_size)
    # Copy to the big array
    new_array[tuple([slice(0, n) for n in array.shape])] = array
    return new_array

def truncate_array(array, length, axis=1):
    """
    Reduces the size of a multidimensional numpy array
    to the specific length in the specific axis.
    E.g. 
    Used in the Haar transform for APCA transformation, to 
    truncate an array that has been padded with zeros to
    a size 2^n, to return it to the original length after.
    """

    # If length is already as specified
    if(array.shape[axis] == length):
        return array

    idx = [slice(None)]*array.ndim
    idx[axis] = range(0,length) # Select the original N values in the axis
    result = array[idx].copy()
    return result


## PAA
def transform_PAA(time_series, chunk_size = 4):
    """
    Returns the Piecewise Aggregate Approximation of a Time series in 2-D numpy array, with the indexes of the left-limit of each chunk, for plotting.
    chunk_size defines the number of data points to be aggregated by the mean, if the number is not multiple of ts length, last chunk is until remaining points.
    Aggregation is performed with non-overlapping windows
    time_series.shape should be [ts_idx, ts_values]
    """
    shape = time_series.shape

    num_ts = 0
    N = 0
    if (len(shape) == 1):
        num_ts = 1
        N = shape[0]
        time_series = time_series.T.copy() # Convert into Dataframe with one row
    elif (len(shape) == 2):
        num_ts = shape[0]
        N = shape[1]
        time_series = time_series
    else:
        print("Error: ts has more than 2 dimensions")
        return
    
    # Number of splits to perform
    M = int(np.ceil(N/chunk_size))
    
    # Returned index is in the middle of each interval
    indexes = np.array([i*chunk_size + (chunk_size/2) for i in range(M)])
    if indexes[-1]>N: indexes[-1]=N #Correct overflow
    vlines = np.array([i*chunk_size for i in range(M+1)])
    if vlines[-1]>N: vlines[-1]=N   #Correct overflow
    
    paa_ts = np.zeros( (num_ts,M) )
    # Process pandas DataFrame in chunks
    #for ts in range(num_ts):
    for i in range(M):
        low_idx = chunk_size*i
        high_idx= chunk_size*(i+1) if (chunk_size*(i+1) < N) else N # Condition for last chunk that might be different length
        paa_ts[:,i] = np.mean(time_series[:, low_idx:high_idx], axis = 1)
        #print(ts,i,'|', low_idx, high_idx, ' = ', mean_value_segment)

    return paa_ts, indexes, vlines


def transform_PAA_multivariate(time_series_array:np.ndarray, chunk_size=4, axis_time=2, indices_dataviz = True):
    """
        Applies PAA (Piecewise Aggregation Approximation) for multivariate TS. Default axis_time is defined 
            to guarantee compatibility with previous experiments.
        :param time_series_array: 3D array which 
        :param chunk_size: Length of each segment.
        :param axis: Value corresponding to the axis containing TIME. 
        :param indices_dataviz: If True, returns a tuple of three elements, otherwise just the data with PAA
        :return: three elements:
            - data_PAA: ndarray with the aggregation over the `axis` each `chunk_size` values.
            if indices_dataviz = True, also returns:
            - centers:  (For plotting) Indices of the center of each interval w.r.t original TS length.
            - borders:  (For plotting) Indices of the borders of all intervals w.r.t original TS length.
    """

    # PAA TS length
    num_ts = None
    N = None

    # Match the time_series_array to 3D even if it has lower dimensions
    dimensions = time_series_array.ndim
    if(dimensions==3):
        pass
    elif(dimensions == 2):
        time_series_array = np.expand_dims(time_series_array, axis=2)
        dimensions = time_series_array.ndim
    elif(dimensions == 1):
        time_series_array = np.expand_dims(time_series_array, axis=1)
        time_series_array = np.expand_dims(time_series_array, axis=2)
        dimensions = time_series_array.ndim
    else:
        raise ValueError('Invalid number of dimensions. Max 3 dimensions in the array')

    # Calculate number of intervals to create
    N = time_series_array.shape[axis_time]
    M = int(np.ceil(N/chunk_size))
    print("Number of segments to create with PAA:", M)

    # Returned index is in the middle of each interval
    centers = np.array([i*chunk_size + (chunk_size/2) for i in range(M)])
    if centers[-1]>N: centers[-1]=N #Correct overflow
    borders = np.array([i*chunk_size for i in range(M+1)])
    if borders[-1]>N: borders[-1]=N   #Correct overflow

    # Allocate memory for new array
    new_size = list(time_series_array.shape)
    new_size[axis_time] = M # Aggregation from N values to M.
    data_PAA = np.zeros( tuple(new_size) )
    for i in range(M):
        # Find index of chunk that needs to be aggregated
        low_idx = chunk_size*i
        high_idx= chunk_size*(i+1) if (chunk_size*(i+1) < N) else N # Condition for last chunk that might be different length
        # Subselect the values in the specific axis
        idx       = [slice(None)]*dimensions
        idx[axis_time] = range(low_idx, high_idx)
        idx = tuple(idx)
        subarray = np.mean(time_series_array[idx], axis=axis_time)

        # Put back to the aggregated version of the array
        idx       = [slice(None)]*dimensions
        idx[axis_time] = i
        idx = tuple(idx)
        data_PAA[idx] = subarray.copy()

    if indices_dataviz:
        return data_PAA, centers, borders
    else:
        return data_PAA


def distance_PAA(ts_, train_array):
    pass
    

### APCA
def truncate_haar_values(array_haar, M=3, axis=2):
    """ Sets to zero the haar coefficients lower than the `M` highest.
    """
    def below_top_M_values(row, num=M):
            """ Function to create a boolean ndarray indicating is lower
            than the `M` highest values in the row
            """
            threshold_in_ts = np.sort(row)[-M]
            return row < threshold_in_ts

    # Find if a value is lower than M highest coefficients, per time-series
    mask = np.apply_along_axis(below_top_M_values, axis, np.abs(array_haar))

    # Truncate values to zero, to keep coefficients with high values
    array_haar_filtered = array_haar.copy()
    array_haar_filtered[mask] = 0
    return array_haar_filtered

def DWT_haar(array, axis = 1):
    """ Returns the approximation and detail coefficients from the Haar Wavelet 
    transform over a 3D array of time series. `axis` defines the dimension where
    the time series values are located
    """

    # Constant normalization factor to preserve signal's energy
    K = np.sqrt(2.)

    # Allocate space for result
    result_coeff = np.zeros(array.shape)

    N = array.shape[axis] # Number  of samples (divisible by 2)

    # Iteration for haar coefficient calculations
    iterations = int(np.log2(N))-1
    for i in range(iterations,-1,-1): # reverse iterations to 0 e.g. 2,1,0
        # Initialization
        if (i==iterations):
            A = array # Approximation Coefficients (Averages in Haar)
            # Indexes to know where to put the coefficients in the result
            hl = int(N)
        else:
            # After second decomposition, the indexes are halved
            hl = ll
            # New shape of the averaged array
            N = A.shape[axis]
        
        ll = int(hl/2)

        # Select specific indexes in the specified axis
        idx_odd = [slice(None)]*A.ndim
        idx_odd[axis] = range(0,N,2)
        A_odd = A[idx_odd]
        #print(A_odd)
        idx_even = [slice(None)]*A.ndim
        idx_even[axis] = range(1,N,2)
        A_even = A[idx_even]
        #print(A_even)

        # Average over odd and even indexes of the time series
        A = (A_odd+A_even)/2
        #print("A",A)
        
        # Detail coeffcients
        detailed_coeffs = (A - A_even)/(np.power(K,i))

        # Results of the Wavelet transform over the original ndarray
        idx_results = [slice(None)]*A.ndim
        idx_results[axis] = range(ll,hl)
        result_coeff[idx_results] = detailed_coeffs

        ## Termination
        if (i==0):
            idx_results = [slice(None)]*A.ndim
            idx_results[axis] = range(0,1) ## The first index includes the approx. coeff.
            result_coeff[idx_results] = A
    
    return result_coeff

def inv_DWT_haar(haar_coeff, axis=1, stop_iterations=None):
    """ Reconstructs a signal based on a ndarray containing the 
    haar coefficients that represent the time series.
    `stop_iterations=None` reconstructs the whole time series, indicate a number
    to stop the reconstruction in the specified step
    """

    K=np.sqrt(2.)
    
    array = haar_coeff

    # Allocate space for result
    result_coeff = np.zeros(array.shape)

    N = array.shape[axis] # Number  of samples (divisible by 2)

    # Iteration for reconstruction from haar coefficients
    iterations = int(np.log2(N))
    # Rename variable
    A = array.copy()
    for i in range(iterations):
        
        # Early stop for partial reconstruction
        if(stop_iterations is not None and i >= stop_iterations):
            idx_results = [slice(None)]*A.ndim
            idx_results[axis] = range(0,N) ## The first index includes the approx. coeff.
            A = A[idx_results].copy()
            break

        if (i==0):
            # Initialization
            gap = 1 # Distance between detail coefficients for calculations
            N = 2 # Number of objects to process
            # Indexes to know where to put the coefficients in the result
        else:
            # Increase the number of reconstructed samples per iteration
            gap = gap*2
            N = N*2

        # Divide the two halves of the arrays
        threshold = int(N/2)
        
        # Select approximation and detail coefficients for calculations
        idx_left = [slice(None)]*A.ndim
        idx_left[axis] = range(0,threshold)
        coeff_left = A[idx_left]
        #print(coeff_left)
        idx_right = [slice(None)]*A.ndim
        idx_right[axis] = range(threshold,N)
        coeff_right = A[idx_right] * (np.power(K,i)) # De-scale for energy conservation
        #print(coeff_right)

        # Select odd and even positions to store the new values
        # The algorithm of inverse transform can be analyzed step by step from:
        #       https://youtu.be/6EyJ70u7IK4?t=1157
        idx_odd = [slice(None)]*A.ndim
        idx_odd[axis] = range(0,N,2)
        idx_even = [slice(None)]*A.ndim
        idx_even[axis] = range(1,N,2)

        # Average over odd and even indexes of the time series
        A[idx_odd] = (coeff_left+coeff_right)
        A[idx_even] = (coeff_left-coeff_right)
    return A

def merge_haar_segments(haar_approximation, original_values, M = 3, truncate_length_result=False, verbose=False):
    """ Replaces the approximated segments from haar reconstruction
    with the real mean values from another array.
    Set `M=None` to return the haar coefficients without merging segments. Returns the
    haar reconstruction with real means and the indices where the value changes
    NOTE: Works in only 1-D array. Used with `apply_along_axis()` over ndarrays
    """

    def calculate_reconstruction_cost(from_index, to_index, original_ts, haar_reconstruction):
        original_signal = original_ts[from_index:to_index]
        haar_before_merging = haar_reconstruction[from_index:to_index]
        #
        before = np.linalg.norm(original_signal-haar_before_merging)

        # TS if both segments were compressed to the mean value
        mean_value = np.mean(haar_reconstruction[ from_index:to_index ])
        modified_signal = [mean_value]*(to_index-from_index)
        #
        after = np.linalg.norm(original_signal-modified_signal)

        # Euclidean distance between before and after reconstruction
        # minimize (error after merging - error before merging)
        return after-before

    # Replace the haar reconstruction for real mean values among
    #   intervals with same reconstruction value
    haar_corrected = haar_approximation.copy()
    
    indices = []   # Right indices that limit the intervals
    costs = [] # Cost of merging two subsequent intervals (to reach M segments)
    unique_vals = [] # Contains the unique values in the array

    # Left index that limits the start of a segment with similar values
    left_lim = 0
    
    # Comparison of similar values
    prev_value = haar_approximation[0]
    for i in range(1,len(haar_approximation)):
        # Flag to know if a change in value has been detected
        flag_changed = False
        segment_mean = 0
        # End of scanning
        if i == (len(haar_approximation)-1):
            flag_changed = True
            # Replace the values for the corresponding mean
            segment_mean = np.mean(original_values[left_lim:])
            haar_corrected[left_lim:] = segment_mean
            indices.append(i) # Add the last index as end of segment
            unique_vals.append(segment_mean)

        # When a different value is found, replace all previous values with their mean
        #  and calculate the cost of subsequent interval for prunning
        elif haar_approximation[i] != prev_value:
            flag_changed = True
            # Replace the values for the corresponding mean
            segment_mean = np.mean(original_values[left_lim:i])
            haar_corrected[left_lim:i] = segment_mean
            indices.append(i-1) # Add the previous value as end of segment
            unique_vals.append(prev_value)

        if flag_changed:
            # Calculate costs of merging between segments
            if left_lim != 0: # More than one segment has been already found
                # L2-Norm between values since the previous segment until end of current segment
                cost = calculate_reconstruction_cost(indices[-2], i, original_values, haar_corrected)
                costs.append(cost)
            # Update values for search in future indices
            prev_value = haar_approximation[i]
            left_lim = i
    if(verbose):
        print('Haar2:',haar_corrected)
        print('Idces:',indices)
        print('UVals:',unique_vals)
        print('Costs:',costs)

    # The signal is a constant and does not fluctuate. Caused error in some TS
    if (len(indices)==1):
        indices = np.floor(np.linspace(0,indices[0],M)).tolist()
        unique_vals = unique_vals[0]*M

    # Merge the subsequent intervals until reaching only `M` segments
    while (len(indices) > M):
        pos_min_cost = np.argmin(costs)
        if(verbose): print('pos_min_cost',pos_min_cost)
        
        # Merge subsequent segments with mean value
        left_limit = indices[pos_min_cost-1]+1 if pos_min_cost !=0 else 0      # Beginning of first segment
        right_limit = indices[pos_min_cost+1]+1
        # MERGE SEGMENTS WITH MEAN
        mean_after_merged = np.mean(haar_corrected[left_limit:right_limit] )
        haar_corrected[ left_limit:right_limit ] = mean_after_merged

        # Update neighboring costs
        #  Right unique value
        unique_vals[pos_min_cost+1] = mean_after_merged
        #  Left neighbouring cost
        if(pos_min_cost != 0):
            left_limit = indices[pos_min_cost-2]+1 if pos_min_cost !=1 else 0
            right_limit = indices[pos_min_cost+1]
            if(verbose): print('upd_left_nb:',left_limit,right_limit)
            costs[pos_min_cost-1] = calculate_reconstruction_cost(left_limit, right_limit, original_values, haar_corrected)
        # Right neighbouring cost    
        if(pos_min_cost != (len(costs)-1)):
            left_limit = indices[pos_min_cost-1]+1 if pos_min_cost !=0 else 0
            right_limit = indices[pos_min_cost+2] if pos_min_cost != (len(costs)-2) else indices[pos_min_cost+1]+1
            if(verbose): print('upd_right_nb:',left_limit,right_limit)
            costs[pos_min_cost+1] = calculate_reconstruction_cost(left_limit, right_limit, original_values, haar_corrected)

        # Remove value from costs and indices
        costs.pop(pos_min_cost)
        indices.pop(pos_min_cost)
        unique_vals.pop(pos_min_cost)

        if(verbose):
            print('---')
            print('Haar2:',haar_corrected)
            print('Idces:',indices)
            print('UVals:',unique_vals)
            print('Costs:',costs)

    # In the case where the reconstruction was lower than M, fill in with existing values
    while (len(indices) < M):
        # Choose two random indices, if they are more than 1 sample away, fill in a sample between them
        idx = np.random.randint(1,len(indices)) # Offset from zero
        #print(idx)
        try:
            if (indices[idx] - indices[idx-1] > 1):
                previous_idx = indices[idx]
                #print(previous_idx)
                previous_value = unique_vals[idx]
                #print(previous_value)
                indices.insert(idx, previous_idx-1)
                unique_vals.insert(idx, previous_value)
        except IndexError:
            print("IndicesShape:",indices.shape)
            print("idx",idx)

    if(truncate_length_result is True):
        if(verbose): print("Returning unique values len=", len(unique_values))
        return unique_vals, indices
    else:
        # Returns the same size than the time series, but with M segments
        if(verbose): print("Returning haar corrected len=", len(haar_corrected))
        return haar_corrected,indices

def merge_haar_segments_multidim(haar_approximation_multidim, original_values_multidim, axis=2, M=3, truncate_length_result=False, verbose=False):
    """ Merges the haar coefficients over the AXIS 2 of a multidimensional array, and merges the segments
    until getting `M` segments. Merging is based on minimization of reconstruction error based on the real values in
    `original_values_multidim` which has to be the same shape than haar_approximation_multidim.
    Set `truncate_length_result=False` to avoid merging process, and return the values and indices after haar reconstruction.
    """

    dims, num_ts, N = haar_approximation_multidim.shape
    if(axis != 2 or dims != 3):
        print("axis!=2. So far this function only works for 3D ndarrays with the values of the time-series in the second dimension.")
        return
    
    if(haar_approximation_multidim.shape != original_values_multidim.shape):
        print("First two parameters should be the same shape")


    # Avoid merging process, returns the original signal
    haar_values = None
    if(truncate_length_result is False):
        # Return the whole vector after merging M segments
        haar_values = np.zeros((dims,num_ts,N))
    else:
        # Return the M merged segments
        haar_values = np.zeros((dims,num_ts,M))

    indices = np.zeros((dims,num_ts,M)) # Store the indices

    for i in range(dims):
        for j in range(num_ts):
            #print("index:",i,j)
            # 1-D iterative merging of haar coefficients
            haar_values[i,j,:], indices[i,j,:] = merge_haar_segments(haar_approximation_multidim[i,j,:], original_values_multidim[i,j,:],
                                                                     M=M, truncate_length_result=truncate_length_result, verbose=verbose)

    return np.array(haar_values), np.array(indices)

def transform_APCA(time_series_array, M=3, truncate_length_result=True, verbose=False):
    """ Calculates the APCA representation of a 1-D time series.
    Returns two arrays trucated to the length M. First, the values of a segment,
    and second array with indexes of the right limits for each segment.

    If `truncate_length_result=False` returns an array of the same shape than the input containing,
    the reconstruction of the time series from the M highest haar coefficients (very likely contain 
    more than M segments), and the indices with right limit of the segments.

    E.g.
    C = np.array([7,5,5,3,7,3,4,6])# ,8,10,2,5,4,8,3,4,5, 2,1,4,5,3,5,9, 1,3,4,5,6,7,8,3,4,7,6,4])
    M = 3 # Coefficients to take from Haar transform

    haar_approx =  transform_APCA(C, M=M, truncate_length_result=False)
    plt.plot(C, 'r', haar_approx, 'b')
    plt.show()

    # Truncating the final result
    unique_vals, indices = transform_APCA(C, M=M, truncate_length_result=True)
    plt.plot(C, 'r', [0]+indices, [unique_vals[0]]+unique_vals, 'b')
    plt.show()
    """
    axis = 0
    # Pad to length power of 2
    padded = pad_to_power_of_2(time_series_array, axis=axis)
    # Apply DWT Haar Transform
    haar_coeffs = DWT_haar(padded,axis=axis)
    # Keep the M highest 
    haar_filtered = truncate_haar_values(haar_coeffs,M=M,axis=axis)

    # Reconstruct time series
    # Apply the inverse Discreete Wavelet Haar Transformation with few values
    reconstructed = inv_DWT_haar(haar_filtered, axis=axis)
    # Truncate (remove padded zeros) to original size if the input time series
    reconstructed = truncate_array(reconstructed, length=time_series_array.shape[axis],axis=axis)

    # Merge segments until get real 
    # Replace intervals with real means and iterate the combination of segments until getting M
    result_APCA, indices= merge_haar_segments(reconstructed, time_series_array, M = M, truncate_length_result=truncate_length_result, verbose=verbose)
    return result_APCA, indices
    
def transform_APCA_multivariate(time_series_array, M=3, axis=2, truncate_length_result=False, verbose=False):
    """ Calculates the APCA representation of a set of multivariate time series stored in a 3D array.
    Returns two multidimensional arrays trucated in the time series to the length M. First, 
    the values of a segment, and second array with indexes of the right limits for each segment.

    If `truncate_length_result=False` returns an array of the same shape than the input containing,
    the reconstruction of the time series from the M highest haar coefficients (very likely contain 
    more than M segments), and the indices with right limit of the segments.

    From paper:
    Chakrabarti et al. (2002) 'Locally Adaptive Dimensionality Reduction for Indexing
    Large Time Series Databases'

    Summary: 
        ### Algorithm Compute_APCA(C,M)
    1. if length(C) is not a power of two, pad it with zeros to make it so.
    2. Perform the Haar Discrete Wavelet Transform on C.
    3. Sort coefficients in order of decreasing normalized magnitude, truncate after M.
    4. Reconstruct approximation (APCA representation) of C from retained coeffs.
    5. If C was padded with zeros, truncate it to the original length.
    6. Replace approximate segment mean values with exact mean values.
    7. `while` the number of segments is greater than M
    8. |||Merge the pair of segments that can be merged with least rise in error
    9. `endwhile`

    Use Example:
    # Multidimensional APCA
    data = np.floor(20*np.random.rand(3,200,100)) # Random 200, 3D time series with 100 samples each
    num_segments_APCA = 30
    result_APCA_md, indices_md = transform_APCA_multivariate(data, M=num_segments_APCA, axis=2, truncate_length_result=True)
    # Visualize one time series
    idx = [0,0,slice(None)]
    plt.plot(data[idx], 'r', [0]+indices_md[idx].tolist(), [result_APCA_md[idx][0]]+result_APCA_md[idx].tolist(), 'b')
    plt.show()
    """
    if(axis!=2 and truncate_length_result is True):
        print("By default, the reconstruction from haar coeffs only work if axis=2. \
        Set truncate_length_result=False to be able to process the array in another axis, and \
        get time series with the same length than input array.")
    
    # Pad to length power of 2
    padded = pad_to_power_of_2(time_series_array, axis=axis)
    # Apply DWT Haar Transform
    haar_coeffs = DWT_haar(padded,axis=axis)
    # Keep the M highest 
    haar_filtered = truncate_haar_values(haar_coeffs,M=M,axis=axis)

    # Reconstruct time series
    # Apply the inverse Discreete Wavelet Haar Transformation with few values
    reconstructed = inv_DWT_haar(haar_filtered, axis=axis)
    # Truncate (remove padded zeros) to original size if the input time series
    reconstructed = truncate_array(reconstructed, length=time_series_array.shape[axis],axis=axis)
    
    # Merge segments until get real 
    # Replace intervals with real means and iterate the combination of segments until getting M
    result_APCA, indices= merge_haar_segments_multidim(reconstructed, time_series_array, axis=2, M=M, truncate_length_result=truncate_length_result, verbose=verbose)
    return result_APCA, indices
    

### SAX
def breakpoints_SAX_normal(n_breakpoints = 4):
    """
    Returns a sorted increasing list ordering the x-axis to create `n_breakpoints` 
    that separate a Gaussian curve in equal-sized areas.
    Assumes that the first breakpoint is in x=-Inf, and the last in x=Inf, only returns

    """
    if(n_breakpoints <=2):
        print('Two breakpoints are located by default in [-Inf, Inf] to create 1 bin. Please put a n_breakpoints > 2')
        return [-np.infty, np.infty]
    breakpoints = norm.ppf(np.linspace(0, 1, n_breakpoints)[1:-1])
    return breakpoints

def transform_SAX(time_series_array, n_bins=4, chunk_size_PAA = 4, process_PAA = True):
    """
    Applies SAX for univariate or multivariate TS. First dimension determines number of dimensions,
    second and third dimension are used to calculate the time series
    By default PAA is applied to the array with `chunk_size_PAA` as non-overlapped window_size, `process_PAA = False` to cancel

    time_series_array.shape should be [dim, ts_idx, ts_values]

    Returns the modified array and the alphabet to map the indexes to a latin string alphabet (up to 26)
    returns index, and vlines positions for plotting
    """
    if(process_PAA and len(time_series_array.shape)!=3):
        print('Error, transform_SAX can only process arrays in 3dim if PAA is going to be applied')
        return

    # Create list of alphabet letters to label the bins
    alphabet = list(string.ascii_lowercase)[0:n_bins]

    # Calculate the breakpoints according to a normal distribution
    bins_SAX = breakpoints_SAX_normal(n_bins + 1)

    data_SAX = None
    if(process_PAA):
        # Calculate PAA from the array
        data_PAA, index_plot, vlines_plot = transform_PAA_multivariate(time_series_array, chunk_size=chunk_size_PAA)
        # Digitize allocates each value according to the given bins
        data_SAX = np.digitize(data_PAA, bins_SAX, right = True)
    else:
        data_SAX = np.digitize(time_series_array, bins_SAX, right = True)

    # Transform bins with plottable values, i.e. mean values between subsequent breakpoints, extremes are moved a percentage
    offset_extremes = 1.4     # How far to plot the extreme values that are in the area between [-inf,b1],[b_t,inf]
    center_bins = [min(bins_SAX)*offset_extremes] + ( [(a + b) / 2 for a, b in zip(bins_SAX[1:], bins_SAX[:-1])] ) + [max(bins_SAX)*offset_extremes]
    
    breakpoints = bins_SAX # Rename
    
    if(process_PAA):
        return data_SAX, alphabet, breakpoints, center_bins, index_plot, vlines_plot
    else:
        return data_SAX, alphabet, breakpoints, center_bins


def distance_lu_table_SAX(alphabet_size, pandas_df = False):
    """
    Creates a look-up table with the distances between alphabet letters 
    from SAX. `pandas_df=True` returns a pandas DataFrame
    with index and columns corresponding labeled with letters from the alphabet.
    If `False`, returns a numpy array.
    """
    bp = breakpoints_SAX_normal(alphabet_size+1) # Breakpoints (beta) from normal dist.

    # Allocate distance table
    distance_table = np.zeros((alphabet_size,alphabet_size))

    # Fill in distances based on MINDIST function from SAX paper
    for i in range(alphabet_size):
        for j in range(i+2,alphabet_size): # Start in +2 because all other cells are already 0
            max_index = np.max([i,j])
            min_index = np.min([i,j])

            distance_table[i,j] = bp[max_index-1] - bp[min_index] # Piecewise
            distance_table[j,i] = distance_table[i,j] # Symmetric matrix

    # Conver to Pandas
    if(pandas_df):
        # Create sequence of letters from a-z according to number of breakpoints
        axis_labels = list(string.ascii_lowercase)[0:alphabet_size]
        distance_table = pd.DataFrame(data=distance_table, index = axis_labels, columns=axis_labels)

    return distance_table