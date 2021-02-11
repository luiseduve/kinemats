import numpy as np
import pandas as pd

# FUNCTION TO LOAD DATASET FILES
def load_files_into_dict(ds_name, sets = ["TRAIN","TEST"], suffix_folders = ["X","Y","Z"], root_dataset_folder = "datasets/", file_format = ".tsv", missing_val = 0):
    """
    Returns a `dict` for each group. Another `dict` contains the data from the suffix folders.
    Access the data from each file as: result[group][suffix] containing the numpy array read from the file.
    Missing values are replaced to 0
    """

    result_dictionary = {}
    for group in sets:
        group_dictionary = {}
        for suffix in suffix_folders:
            # File to open
            filename = root_dataset_folder + ds_name + suffix + "/" + ds_name + suffix + "_" + group + file_format
            # Load file
            array = pd.read_csv(filename, delimiter="\t")
            array.fillna(missing_val,inplace=True)
            array = array.to_numpy()
            print("Loading {0} size {1}".format(filename,array.shape))

            # Create dictionary keys
            # The first column of each file contains the class, these are stored in a different key called suffix_"CLASS"
            group_dictionary[str("class"+suffix)] = pd.Series(array[:,0], name=str("class"+suffix), dtype='Int32')

            # Normalize time series per row, if specified in parameter
            time_series_array = array[:,1:]
            
            # Create dataframe with axis info
            group_dictionary[suffix] = pd.DataFrame(data = time_series_array)

        # Final dictionary key
        result_dictionary[group] = group_dictionary
    print(">>Done!")
    return result_dictionary

def load_ucr_dataset(dataset_name, keep_sets = True, root_folder = "datasets/", suffix_folders = ["X","Y","Z"], sets = ["TRAIN","TEST"], file_format = ".tsv", missing_val = 0):
    """
    Loads files from UCI repository: 
    - dataset_name: "UWaveGestureLibrary", "Cricket", "AllGestureWiimote"
    - combine_sets: Defines whether TRAIN/TEST split should be maintained, or merged.
    - missing_val = For which value should NA be replaced? Default: 0

    The location of the files is constructed as:
        root_folder + dataset_name + suffix_folders + "/" + dataset_name + suffix_folders + "_" + sets + file_format
        e.g. dataset/UCR/ + UWaveGestureLibrary + Z + "/" + UWaveGestureLibrary + Z + "_" + TRAIN + .tsv

    If keep_sets=True, returns (train, test, labels_train, labels_test), where the train, test data are numpy arrays
        with dimensions = len(suffix_folders), and with shape [idx_time_series, time, dimension]
    If keep_sets=False, both sets are merged and returns (data, labels)
    """
    
    data_dictionary = load_files_into_dict(dataset_name, suffix_folders = suffix_folders,\
                                         sets = sets, root_dataset_folder = root_folder, 
                                         file_format = file_format, missing_val = missing_val)

    data_loaded = data_dictionary

    # Extract labels for each set
    labels_trn = data_loaded[sets[0]][str("class"+suffix_folders[0])]     # data_loaded['TRAIN']['classX']
    labels_tst = data_loaded[sets[1]][str("class"+suffix_folders[0])]     # data_loaded['TEST']['classX']


    ### DATA 'TRAIN' SET
    num_ts_train, N = data_loaded[sets[0]][suffix_folders[0]].shape       # data_loaded['TRAIN']['X'].shape
    dims = len(suffix_folders)      # Dimensions per folder

    # Create training data
    train = np.zeros( (num_ts_train, N, dims) )
    for i in range(dims):
        train[:,:,i] = data_loaded[sets[0]][suffix_folders[i]].to_numpy()

    ### DATA 'TEST' SET
    num_ts_test, N = data_loaded[sets[1]][suffix_folders[0]].shape       # data_loaded['TEST']['X'].shape

    # Create training data
    test = np.zeros( (num_ts_test, N, dims) )
    for i in range(dims):
        test[:,:,i] = data_loaded[sets[1]][suffix_folders[i]].to_numpy()

    # Combine in one single set if neede, e.g. for custom cross-validation:
    if(not keep_sets):
        return np.concatenate((train,test),axis=0), np.concatenate((labels_trn,labels_tst),axis=0), 
    return train, test, labels_trn, labels_tst