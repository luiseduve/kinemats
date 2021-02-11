import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import data manipulation libraries
from io import BytesIO
import tarfile
from enum import Enum

# Import scientific 
import numpy as np
import pandas as pd

import utils
from quaternion_math import *


# %%
def is_subfolder(path, parent):
    """
    Checks whether one file or folder path is an child of the parent.
    :param str path: Path to compare as a subchild
    :param str parent: Path to validate parenthood
    :returns: True or False
    """
    return (parent in path)

# %%
def is_immediate_child_path(path, parent):
    """
    Checks whether one file or folder path is an IMMEDIATE child of the parent.
    :param str path: Path to compare as a subchild
    :param str parent: Path to validate parenthood
    :returns: True or False
    """
    # By default the parenthood is false
    result = False
    
    # String of the parent needs to be contained in the path of comparison
    is_child_path = is_subfolder(path, parent)
    
    if is_child_path:
        # Consider the part of the string that does not include the parent path
        substring = path[len(parent)+1:]
        #print("sub: " + substring)
        
        # It is an immediate child it the remaining path does not contain "/" symbol
        if("/" not in substring):
            result = True
    return result



class VideoList(Enum):
    """
    Enum to access the dictionary with the data per video
    """
    Diving = "Diving"
    Paris = "Paris"
    Rollercoaster = "Rollercoaster"
    Timelapse = "Timelapse"
    Venice = "Venice"
    Elephant = "Elephant-train"
    Rhino = "Rhino-train"

    def __str__(self):
        return super().value.__str__()

class DatasetHeadMovIMT():
    """
    This class contains the loads the original .tar.gz file for 
    *360-Degree Videos Head Movements Dataset* developed by Corbillo et al.[CORBI2017]_
    at the Institute Mines-Telecom (IMT).

    The output of this class is a structured dataframe containing users 
    demographics and general information about loaded dataset. 
    Moreover, a dictionary contains the head movement data per user in a numpy
    array for easy access. 

    Download link for original dataset: http://dash.ipv6.enstb.fr/headMovements/

    .. [CORBI2017]
        MMSys'17: Proceedings of the 8th ACM on Multimedia Systems Conference
        June 2017 Pages 199–204 https://doi.org/10.1145/3083187.3083215
    """

    
    # Final structure of the filepaths per user
    filenames_dict = {
        "folder_id": "",
        "demographics": "",
        "main_data": {}
    }

    # Initial columns of the dataframe containing demographics info.
    # The columns dataframe are expanded by script to store values
    # about the videos (e.g. number of samples recorded per video)
    general_dict = {'user': '',
                'folder_path': '',
                'gender': '',
                'age': 0,
                'impairment': '',
                'hoursHMD': -1,
                'appsUsed': '',
                'devicesUsed':'',
                }
    
    # Structure of the dataset containing the data.
    # The values of the dict correspond to folder name where data is stored,
    # these values are used to find valid movement files per user
    movement_dict = {"Diving": "Diving-2OzlksZBTiA",
              "Paris":"Paris-sJxiPiAaB4k",
              "Rollercoaster":"Rollercoaster-8lsB-P8nGSM",
              "Timelapse":"Timelapse-CIw8R8thnm8",
              "Venice":"Venise-s-AJRFQuAtE",
              "Elephant-train": "Elephant-training-2bpICIClAIg",
              "Rhino-train":"Rhino-training-7IWp875pCxQ",
             }

    # Labels of the six variables included in the Head Movements per video
    movement_headings = ['timestamp', 'frameId', 'q0', 'q1i', 'q2j', 'q3k']


    ### STRUCTURE OF THE COMPRESSED .tar.gz FILE
    root_folder = "results/"
    data_files_extension = ".txt"
    user_demographics_filename = "formAnswers.txt"

    # OUTPUT VALUES
    general = pd.DataFrame()
    movement = []
    processed = []
    
    def __init__(self, tar_path, dictionary):
        """
        Initializes object that analyzes dataset

        :param tar_path: Path with the file `tar.gz`
        :type tar_path: str
        :param json_path: Path where the JSON file with the directory will be saved
        :type json_path: str
        """
        self.tar_path = tar_path
        self.dict = dictionary

    def get_movement(self, user, video):
        """
        Returns the nparray of the specified user and video
        :param user: User ID
        :type user: int
        :param video: Video key
        :type video: `VideoList`

        Example: data.get_movement(1,`VideoList.Paris`)
        """
        return self.movement[user][video.value]

    def get_movement_filtered(self, user, video, column_to_filter=0, min_value=0, max_value=1):
        user_mov = self.movement[user][video.value].copy()

        # Find the filtered data
        rows_to_keep = np.where( (user_mov[:,column_to_filter]>=min_value) \
                                    & (user_mov[:,column_to_filter]<=max_value))

        return user_mov[rows_to_keep]

    ##### CREATE DICTIONARY FROM COMPRESSED FILE
    def generate_file_index(self):
        """
        Generates a dictionary with the filepaths of a structured dataset per user
        :return: 0 if the JSON file with pathfiles of was created
        :rtype: int
        """
        return self.__create_dict_from_tar(self.tar_path, self.dict)

    # %%
    def __create_dict_from_tar(self, tar_path = 'dataset.tar.gz', json_path = "folder_tree.json"):
        """
        Generates a dictionary with the filepaths of a structured dataset per user

        :param tar_path: Path with the file `tar.gz`
        :param json_path: Path where the JSON file with the directory will be saved
        :type tar_path: str
        :type json_path: str
        :return: Dictionary of file paths per user and video
        :rtype: dict
        """
        
        filenames_dict_keys = list(self.filenames_dict.keys())

        try:
            # Dictionary to store files
            files_index = {}
            
            # Process compressed dataset
            with tarfile.open(tar_path, "r:*") as tar:
                user_count = 0
                for member in tar.getmembers():

                    # Find the folders that below the root folder, to consider each folder as user
                    if (member.isdir() and is_immediate_child_path(member.name, self.root_folder)):
                        # Create keys for every different user
                        user_count += 1
                        user_id = "user"+str(user_count)

                        # Create main dictionary with folder tree
                        files_index[user_id] = self.filenames_dict.copy()
                        files_index[user_id][filenames_dict_keys[0]] = member.name
                        files_index[user_id][filenames_dict_keys[2]] = self.movement_dict.copy()
                        #print("Updating dict", user_id)

                    # Filter files that finish in the extension of the main data (e.g. ".txt")
                    if (member.isfile()) and (self.data_files_extension in member.name[-4:]):
                        # Associate the main data file with its respective user
                        for user in files_index.keys():
                            # If the path contains a part of the users that was already found
                            if (is_subfolder(member.name, files_index[user][filenames_dict_keys[0]])):
                                filename = member.name  # Holds the text path of the file  with the data
                                # The file is either demographics information
                                if filename.find(self.user_demographics_filename) != -1:
                                    # print(filename, "is demographics for", user)
                                    files_index[user][filenames_dict_keys[1]] = filename
                                # Or data from a video folder, matched from the initial dict of videos
                                else:
                                    # Compare if the path includes a substring similar to the videos
                                    for key, value in files_index[user][filenames_dict_keys[2]].items():
                                        if filename.find(value) != -1:
                                            #print(filename, "belongs to video", key, "in", user)
                                            files_index[user][filenames_dict_keys[2]][key] = filename

                # Store the files in a JSON
                utils.create_json(files_index, json_path)
                return 0
                
        except SystemExit:
            print("No users where found in the provided dataset")

    ### CREATE STRUCTURED DATA FROM DICTIONARY
    def uncompress_data(self, files_index_path, debug_users = None, list_unprocessed_users = None):
        """
        Loads in memory a structured object of the dataset
        The dictionary contains the file structure with data from each users.
        
        :param tar_path: Path with the file `tar.gz`
        :type tar_path: str
        :param files_index_path: Path of JSON file with the dictionary
        :type files_index: dict
        :param debug_users: Load a subset of users
        :type debug_users: int
        :param list_unprocessed_users: Does not process the users with the indexes defined in the list
        :type list_unprocessed_users: list[int]
        :return: 0 if data was loaded properly
        :rtype: int
        """
        return self.__load_structured_data(self.tar_path, files_index_path, debug_users, list_unprocessed_users)
        

    def __load_structured_data(self, tar_path, files_index, debug_users = None, list_unprocessed_users = None):        
        """
        Loads in memory a structured object of the dataset
        The dictionary contains the file structure with data from each users.
        
        :param tar_path: Path with the file `tar.gz`
        :type tar_path: str
        :param files_index: Path of JSON file with the dictionary
        :type files_index: dict
        :param debug_users: Load a subset of users, to maximize data loaded
        :type debug_users: int
        :param list_unprocessed_users: Does not process the users with the indexes defined in the list
        :type list_unprocessed_users: list[int]
        :return: 0 if data was loaded properly
        :rtype: int
        """

        user_dict_keys = list(self.general_dict.keys())
        videos_dict_keys = list(self.movement_dict.keys())
        
        # Load compressed dataset
        tar = None
        try:  
            tar = tarfile.open(tar_path, "r:*")  
        except FileNotFoundError as e:
            print(e)
            print("No file found. Please check that the dataset and json files exist.\n" +
                "Remember that to create the json file you need to execute the code setting CREATE_FILE_INDEX_JSON to \'True\'")
            
        #Iterate over users
        internal_user_idx = 0
        for user in files_index.keys():
            internal_user_idx += 1
            # Debug some of the users in the file
            if debug_users != None and type(debug_users) == type(int()):
                #Run for a subset of users
                if (internal_user_idx > debug_users): break

            if list_unprocessed_users is not None:
                # Skip users proven to have EMPTY values, after first preprocessing stage
                if internal_user_idx in list_unprocessed_users:
                    print("Skipping entry with EMPTY data, userID:",internal_user_idx)
                    continue
            
            print("Loading...", user)

            # Create a new dictionary reading the demographic information
            user_general_data = self.general_dict.copy()
            user_general_data[user_dict_keys[0]] = user
            # Create a new dictionary to contain movement data per video
            user_movement_data = self.movement_dict.copy()

            # Analyze all the folders per user and load files
            for key,value in files_index[user].items():
                # Create the respective dataframes
                
                    # Check if the path exists in the compressed dataset and it is a file
                    if (key == "folder_id"):
                        # Fill out the folder
                        user_general_data[user_dict_keys[1]] = value

                    elif(key == "demographics"):
                        member = None
                        try:
                            member = tar.getmember(value)
                        except KeyError:
                            # The string in the dict does not correspond to a path to a file in the .tar.gz
                            pass

                        file_object = BytesIO(tar.extractfile(member).read())
                        read_file = pd.read_csv(file_object, header=None, sep = ";", skiprows=[0])
                        # Read lines of file demographics.csv in each users folder
                        if(read_file.shape[0] >= 6 and read_file.shape[1] >= 2):
                            # Fill out demographics in the dictionary
                            compressed = dict(zip(user_dict_keys[2:8], read_file.iloc[1:7,1].tolist()))
                            for k,v in compressed.items(): 
                                user_general_data[k] = v

                    # DATA CONTAINING THE ROTATIONAL VALUES FROM THE VIDEOS
                    elif(key == "main_data"):
                        for video,path in files_index[user][key].items():
                            
                            data_video = None
                            try:
                                data_video = tar.getmember(path)

                                # Load file with main rotational data
                                if(data_video.isfile()):
                                    # Read the file
                                    file_object = BytesIO(tar.extractfile(data_video).read())
                                    #read_file = pd.read_csv(file_object, header=None, sep = " ")
                                    movement_ndarray = np.loadtxt(file_object)
                                    
                                    # General information about the video
                                    user_general_data[video] = [movement_ndarray.shape[0]]  # Value to show per video in demographics df (Number of elements)
                                    # Movement data of that video
                                    user_movement_data[video] = movement_ndarray.copy()

                            except KeyError:
                                # If it is a non-existing file from the main_data key, containing the videos,
                                # Add a column to general df whether video exists or not
                                user_general_data[video] = [0]
                                # Add movement data with 6 zeros
                                user_movement_data[video] = np.zeros(6).reshape((1,6))
                                #print("key,value=",key,value," is not found in dataset")

            ## All the files of ONE user are loaded
            user_general_df = pd.DataFrame.from_dict(user_general_data)
            #print("Final_DF\n",user_general_df)
            self.general = self.general.append(user_general_df, ignore_index = True)
            self.movement.append(user_movement_data.copy())

            #print("user", user, "||\tdict Keys_size",len(movement_data.keys()), movement_data["Elephant-train"].shape, movement_data["Rhino-train"].shape)
        # Close dataset compressed file
        tar.close()
        return 0

    ### PREPROCESSING

    def create_original_sampling_summary(self):
        """
        Create summary of statistics of the original data sampling. Useful to understand
        the nature of the original head-movement dataset and define a proper Sampling Frequency
        and Window to make the time series comparable.

        :return: Dataframe with sample of statistics per user, per video
        :rtype: `pandas.DataFrame`
        """

        sampling_stats_headers = ["user","video","startingTime","endTime","N","firstFrame","lastFrame","avTsampling","avFsampling",]

        sampling_stats_table = []
        num_users = len(self.movement)
        for user in range(num_users):
            for video, rot_array in self.movement[user].items():
                start_timestamp = rot_array[0,0]
                end_timestamp = rot_array[rot_array.shape[0]-1,0]

                samples = rot_array.shape[0]
                firstframe = rot_array[0,1]
                lastframe = rot_array[rot_array.shape[0]-1,1]

                sampling_time = (end_timestamp-start_timestamp)/samples
                sampling_freq = 1 / sampling_time

                stats_per_video = [user, str(video), start_timestamp, end_timestamp, samples, firstframe, lastframe, sampling_time, sampling_freq]
                sampling_stats_table.append(stats_per_video)

        return pd.DataFrame(sampling_stats_table, columns=sampling_stats_headers)
        

    def delete_data_from_videos(self, video_keys_to_delete):
        """
        Deletes the head-movement data for the specified video keys in all users.
        
        :param video_keys_to_delete: List of keys in `VideoList` that will be deleted for all users
        :type video_keys_to_delete: list of elements in the Enum `VideoList`
        :return: 0 if data was loaded properly deleted in the dataframe
        :rtype: int
        """
        videos_to_delete = [str(video) for video in video_keys_to_delete]
        num_users = len(self.movement)
        for user in range(num_users):
            for video in videos_to_delete:
                r = self.movement[user].pop(video, None)    # Returns None if video is not found
                if r is None: print("Key",video,"was not found in user index", user)
        return 0


    def resample_movement(self, sampling_frequency = 30, starting_time = None, end_time = None):
        
        """
        Performs SLERP (Spherical Linear Interpolation) in every quaternion of the head movements,
        to the specified sampling frequency between the defined time window. 
        The window duration must exist in all videos.

        :param sampling_frequency: Frequency to which the data will be interpolated
        :type sampling_frequency: float
        :param starting_time: Start of timestamps window that will be analyzed.
        :type starting_time: float
        :param end_time: End of timestamps window that will be analyzed.
        :type end_time: float
        :return: 0 if data was loaded properly processed
        :rtype: int
        """

        self.processed = self.movement.copy()

        RESAMPLING_FREQUENCY = sampling_frequency # Hz
        SAMPLING_TIME = 1/RESAMPLING_FREQUENCY

        # Chunks of videos to process
        STARTING_TIME = starting_time
        END_TIME = end_time

        NUM_SAMPLES = round((END_TIME-STARTING_TIME)*RESAMPLING_FREQUENCY + 1) # +1 because the starting time includes a sample
        print("Each numpy array will be resampled to", NUM_SAMPLES, "samples. Timestamps from",STARTING_TIME, "to", END_TIME, "seconds")

        num_user = num_users = len(self.processed)
        for user in range(num_users): #[1,2]:
            for video in self.processed[user].keys():
                original_movement = self.processed[user][video]
                rows, cols = original_movement.shape

                # nparray containing the resampled head movement data
                resampled_movement = np.empty((NUM_SAMPLES,5))
                idx_resampled_mov = 0        # Access numpy array to populate with resampled data
                
                # Variables to keep track of the previous values to do the SLERP
                prev_timestamp = None
                prev_quaternion = None

                # Explore all rows of the original data
                for i in range(rows): #range(300):
                    # Cols contain (timestamp, frameId, q0, q1, q2, q3)
                    timestamp = original_movement[i,0]
                    quaternion = original_movement[i,2:6]
                    
                    # PROCESS SLERP
                    next_resampled_timestamp = STARTING_TIME + (SAMPLING_TIME * idx_resampled_mov) # Steadily increases timestamp according to SAMPLING_TIME


                    #if(user == 22 and video==str(VideoList.Rollercoaster) and idx_resampled_mov==898):
                    #    print("CHECK!")


                    # Find the sample slightly below and above desired resampled timestamp to perform interpolation
                    if (timestamp <= next_resampled_timestamp):
                        prev_timestamp = timestamp
                        prev_quaternion = quaternion
                    
                    if (next_resampled_timestamp <= timestamp):
                        # Multiple resampled values might be within the same two points in the original data, loop all of them.
                        while(next_resampled_timestamp <= timestamp):
                            #print("next timestamp sample:", next_resampled_timestamp, "found between", prev_timestamp, "and",timestamp)

                            # Execute SLERP. t remap the timestamps to move from [0-1], slerp calculates proportional quaternion.
                            t = (next_resampled_timestamp - prev_timestamp)/(timestamp - prev_timestamp) if (timestamp - prev_timestamp) !=0 else 0 # prevent divide by zero.
                            slerp_result = slerp(prev_quaternion,quaternion,[t])

                            ### SAVE THE RESAMPLED TIMESTAMP AND SLERP VALUE
                            resampled_movement[idx_resampled_mov,0:6] = [next_resampled_timestamp] + slerp_result[0].tolist()
                            # Find next resampled timestamp
                            idx_resampled_mov += 1
                            next_resampled_timestamp = STARTING_TIME + (SAMPLING_TIME * idx_resampled_mov)

                            if (idx_resampled_mov >= NUM_SAMPLES and timestamp>=END_TIME):
                                break 
                        
                        # Update previous values with latest processed timestamp for next comparison
                        prev_timestamp = timestamp
                        prev_quaternion = quaternion

                    # Only process until one sample over the ending time, then stop reading the file.
                    if (timestamp>=END_TIME):
                        break 

                # The new array with proper sampling frequency has been created
                self.processed[user][video] = resampled_movement.copy()
                print("USER:", user, "VIDEO:", str(video), "| NEW SHAPE:", self.processed[user][video].shape, "idx_resampled", idx_resampled_mov)
        return 0

    def filter_subset_movement_data(self, column_to_filter=0, min_value=0, max_value=1):
        """
        Replace the data for the filtered version according to a column an range 
        (including both extreme values)
        """
        num_user = num_users = len(self.movement)
        for user in range(num_users): #[1,2]:
            for video in self.movement[user].keys():
                original_movement = self.movement[user][video]
                rows, cols = original_movement.shape

                # Find the filtered data
                rows_to_keep = np.where( (original_movement[:,column_to_filter]>=min_value) & \
                                            (original_movement[:,column_to_filter]<=max_value))
                filtered_data = original_movement[rows_to_keep]

                # The new array with filtered data
                self.movement[user][video] = filtered_data.copy()
                print("USER:", user, "VIDEO:", str(video), "| NEW SHAPE:", self.movement[user][video].shape)
        return 0

def load_dataset_IMT(labels_filename, timestamps_filename, dataset_filename):
    # Load or create dataframe with statistics of initial dataset (58 users, 5 videos)
    labels = None
    timestamps = None
    dataset = None

    ### INPUTS / OUTPUTS
    """EDIT CUSTOM FILENAMES"""
    input_files = [labels_filename, timestamps_filename, dataset_filename]

    RELOAD_TRIES = 2
    # Try to load files maximum two times
    for tries in range(RELOAD_TRIES):
        try:
            ### LOAD FILE
            print(f"Trying {tries+1}/{RELOAD_TRIES} to load files: {input_files}")
            
            ### CUSTOM SECTION TO READ FILES
            """EDIT CUSTOM READ"""
            labels = pd.read_csv(input_files[0])
            print(f"File {input_files[0]} was successfully loaded")
            timestamps = np.loadtxt(input_files[1])
            print(f"File {input_files[1]} was successfully loaded")
            dataset = utils.load_binaryfile_npy(input_files[2])
            print(f"File {input_files[2]} was successfully loaded")


        except Exception as e:
            ### CREATE FILE
            print(f"File not found. Creating again! {e}")

            ### CUSTOM SECTION TO CREATE FILES 
            """EDIT CUSTOM WRITE"""

            print(">>> Run the notebook `preprocess_datasets.ipynb` to create the necessary files!!!")

            ### ---- CONTROL RETRIES
            if tries+1 < RELOAD_TRIES:
                continue
            else:
                raise FileNotFoundError('File not found. Run first the notebook `preprocess_datasets.ipynb` to create the necessary files!')
        
        # Finish iteration
        break
    return labels, timestamps, dataset







if __name__ == "__main__":
    import sys
    print('\t==== Class to load dataset from http://dash.ipv6.enstb.fr/headMovements/!!!!')
    print('\tPathfile',sys.argv[0])
    if(len(sys.argv) != 1):
        print('\tFirst parameter',sys.argv[1])
