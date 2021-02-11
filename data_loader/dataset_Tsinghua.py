import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import data manipulation libraries
from io import BytesIO
import tarfile
from enum import Enum


from zipfile import ZipFile

# Import scientific 
import numpy as np
import pandas as pd

#from . import utils
import utils
from quaternion_math import *

"""
The dataset contains two experiments:
- Experiment 1: Users are free to move around without focusing on the content.
- Experiment 2: Users are explicitly told to focus on the contant.

"""


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
        
        if(len(substring)>0 and substring[len(substring)-1] == "/"):
            substring = substring[:-1]
        
        # It is an immediate child it the remaining path does not contain "/" symbol
        if("/" not in substring):
            result = True
    return result

class DatasetHeadMovTsinghua():
    """
    This class contains the loads the original .zip file for 
    *A Dataset for Exploring User Behaviors in VR Spherical Video Streaming*
    developed by Wi et al.[Wu2017]_
    at Tsinghua University (IMT).

    The output of this class is a structured dataframe containing users 
    demographics and general information about loaded dataset. 
    Moreover, a dictionary contains the head movement data per user in a numpy
    array for easy access. 

    Download link for original dataset: https://wuchlei-thu.github.io/

    .. [Wu2017]
        Chenglei Wu, Zhihao Tan, Zhi Wang, Shiqiang Yang. 2017. A Dataset for
        Exploring User Behaviors in VR Spherical Video Streaming. In Proceedings
        of MMSys’17, Taipei, Taiwan, June 20-23, 2017, 6 pages.
        DOI: http://dx.doi.org/10.1145/3083187.3083210
    """

    
    # Final structure of the filepaths per user
    filenames_dict = {
        "demographics": "",
        "metadata_videos": {},
        "main_data": {}
    }

    subdirs_dict = {
        "Experiment_1": "",
        "Experiment_2": ""
        }

    # # Initial columns of the dataframe containing demographics info.
    # # The columns dataframe are expanded by script to store values
    # # about the videos (e.g. number of samples recorded per video)
    # general_dict = {'user': '',
    #             'folder_path': '',
    #             'gender': '',
    #             'age': 0,
    #             'impairment': '',
    #             'hoursHMD': -1,
    #             'appsUsed': '',
    #             'devicesUsed':'',
    #             }

    # Labels of the six variables included in the Head Movements per video
    dataframe_headings = ['playbackTime','q0', 'q1i', 'q2j', 'q3k','posx','posy','posz']


    ### STRUCTURE OF THE COMPRESSED .tar.gz FILE
    ROOT_FOLDER = "Formated_Data/"
    DATA_FILES_EXTENSION = "csv"
    USER_DEMOG_FILENAME = "userDemo.csv"
    METADATA_VIDEO_FILENAME = "videoMeta.csv"

    # OUTPUT VALUES
    general = pd.DataFrame()
    metadata_videos = [pd.DataFrame(), pd.DataFrame()] # Metadata for each experiment
    original_data = {}
    processed = {}
    
    def __init__(self, zip_path, json_dict_path):
        """
        Initializes object that analyzes dataset

        :param zip_path: Path with the file `tar.gz`
        :type zip_path: str
        """
        self.zip_path = zip_path
        self.dict_path = json_dict_path

    ##### CREATE DICTIONARY FROM COMPRESSED FILE
    def generate_file_index(self):
        """
        Generates a dictionary with the filepaths of a structured dataset per user
        :return: 0 if the JSON file with pathfiles of was created
        :rtype: int
        """
        return self.__create_dict_from_zip(self.zip_path, self.dict_path)

    # %%
    def __create_dict_from_zip(self, zip_path = 'dataset.zip', json_path = "folder_tree.json"):
        """
        Generates a dictionary with the filepaths of a structured dataset per user

        :param zip_path: Path with the file `tar.gz`
        :param json_path: Path where the JSON file with the directory will be saved
        :type zip_path: str
        :type json_path: str
        :return: Dictionary of file paths per user and video
        :rtype: dict
        """
        
        try:

            # Process compressed dataset
            myzip = ZipFile(zip_path)

            for f in myzip.infolist():
                ## ITERATE ALL OBJECTS IN ZIP

                # print(f.filename)

                if(f.is_dir() and is_immediate_child_path(f.filename, self.ROOT_FOLDER)):
                    # Experiment subfolder
                    parts = f.filename.split("/")
                    experiment = parts[1]       # Extract experiment key
                    if experiment is not "":
                        self.filenames_dict["main_data"][experiment] = {}
                    
                elif(f.is_dir()):
                    # User folder
                    parts = f.filename.split("/")
                    experiment = parts[1]       # Extract experiment key
                    userId = int(parts[2])           # Extract userId

                    self.filenames_dict["main_data"][experiment][userId] = {}

                elif( not f.is_dir() and ( self.USER_DEMOG_FILENAME in f.filename) ):
                    # User demographics
                    self.filenames_dict["demographics"] = f.filename
                
                elif( not f.is_dir() and ( self.METADATA_VIDEO_FILENAME in f.filename) ):
                    # Metadata Video
                    parts = f.filename.split("/")
                    experiment = parts[1]       # Extract experiment key
                    self.filenames_dict["metadata_videos"][experiment] = f.filename

                elif( not f.is_dir() and ( f.filename.split(".").pop(-1) == self.DATA_FILES_EXTENSION ) ):
                    # Final data
                    parts = f.filename.split("/")
                    experiment = parts[1]       # Extract experiment key
                    userId = int(parts[2])           # Extract userId
                    videoFile = parts[3]
                    videoId = videoFile[ parts[3].find("_")+1:parts[3].find(self.DATA_FILES_EXTENSION)-1 ] # Extract number from "video_X.csv"
                    # print(f"data[{experiment}][{userId}][{videoId}]={f.filename}")
                    
                    # Create key if needed
                    # if (not isinstance(self.filenames_dict["main_data"][userId][experiment], dict)):
                    #     self.filenames_dict["main_data"][experiment][userId] = {}

                    # Store filepath
                    self.filenames_dict["main_data"][experiment][userId][videoId] = f.filename
                            
            myzip.close()

            # Store the files in a JSON
            utils.create_json(self.filenames_dict, json_path)
            return 0
            
        except SystemExit:
            print("No users where found in the provided dataset")

    ### CREATE STRUCTURED DATA FROM DICTIONARY
    def uncompress_data(self, files_index_path=None, debug_users = None, list_unprocessed_users = None):
        """
        Loads in memory a structured object of the dataset
        After returning from this function dictionary contains the file to access:
            data.demographics: Dataframe with demographics of users
            data.metadata_videos[0] >   Dataframe with metadata from Experiment_1
            data.metadata_videos[1] >   Dataframe with metadata from Experiment_2
            data.original_data[0] > Data from Experiment_1, use [1] for Experiment_2
                .original_data[0][userId][video] = Numpy array with data
        
        :param zip_path: Path with the file `tar.gz`
        :type zip_path: str
        :param debug_users: Load a subset of users
        :type debug_users: int
        :param list_unprocessed_users: Does not process the users with the indexes defined in the list
        :type list_unprocessed_users: list[int]
        :return: 0 if data was loaded properly
        :rtype: int
        """
        return self.__load_structured_data(self.zip_path, files_index_path, debug_users, list_unprocessed_users)
        

    def __load_structured_data(self, zip_path, files_index, debug_users = None, list_unprocessed_users = None):        
        """
        Wrapper for uncompress_data
        """
        
        # Load compressed dataset
        myzip = None
        try:
            # Process compressed dataset
            myzip = ZipFile(zip_path)
            
        except FileNotFoundError as e:
            print(e)
            print("No file found. Please check that the dataset and json files exist.")
            
        # Files
        if files_index is None:
            files_index = utils.load_json(self.dict_path)
        
        # Load demographics
        self.demographics = pd.read_csv( BytesIO(myzip.read(files_index["demographics"])) )

        # Experiments
        for i,exp in enumerate(files_index["metadata_videos"].keys()):
            self.metadata_videos.append( pd.read_csv( BytesIO(myzip.read( files_index["metadata_videos"][exp] ) ) ) )
            self.original_data[i] = {} # Create dict placeholder for data

        # Iterate over Experiments
        for expIdx, (expKey, expData) in enumerate(files_index["main_data"].items()):
            
            self.original_data[expIdx] = {}

            # Iterate over users
            internal_user_idx = -1
            for userIdx, videoData in expData.items():
                userIdx = int(userIdx)

                internal_user_idx = internal_user_idx + 1
                # Debug some of the users in the file
                if debug_users != None and isinstance(debug_users,int):
                    #Run for a subset of users
                    if (internal_user_idx > debug_users): return

                if list_unprocessed_users is not None:
                    # Skip users proven to have EMPTY values, after first preprocessing stage
                    if user in list_unprocessed_users:
                        print("Skipping entry with EMPTY data, userID:",)
                        continue
                
                print("Loading...", userIdx)

                self.original_data[expIdx][userIdx] = {}

                # Iterate videos
                for videoIdx,path in videoData.items():      
                    videoIdx = int(videoIdx)

                    data_video = None
                    try:

                        # Read the CSV file
                        file_object = BytesIO(myzip.read(path))
                        data_video = pd.read_csv(file_object)
                        # print(data_video.head())

                        data_video = data_video.drop(data_video.columns[[0]], axis=1)
                        data_video.columns = self.dataframe_headings
                        
                        # To numpy
                        data_video = data_video.to_numpy()

                        self.original_data[expIdx][userIdx][videoIdx] = data_video.copy()

                    except KeyError:
                        pass
        
        # Close dataset compressed file
        myzip.close()
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

        sampling_stats_headers = ["experiment","user","video","startingTime","endTime","N","magQuat","avTsampling","avFsampling",]

        sampling_stats_table = []

        for expIdx, (expKey, expData) in enumerate(self.original_data.items()):
            for userIdx, videoData in expData.items():
                for videoIdx, data in videoData.items():

                    start_timestamp = data[0,0]
                    end_timestamp = data[data.shape[0]-1,0]

                    samples = data.shape[0]

                    magQuat = np.average(np.linalg.norm(data[:,1:5], axis=1))

                    sampling_time = (end_timestamp-start_timestamp)/samples
                    sampling_freq = 1 / sampling_time

                    stats_per_video = [expIdx, userIdx, videoIdx, start_timestamp, end_timestamp, samples, magQuat, sampling_time, sampling_freq]
                    sampling_stats_table.append(stats_per_video)

        return pd.DataFrame(sampling_stats_table, columns=sampling_stats_headers)
        
    def resample_movement(self, experiment_id = 0, sampling_frequency = 30, starting_time = None, end_time = None):
        
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

        self.processed = self.original_data[experiment_id].copy()

        RESAMPLING_FREQUENCY = sampling_frequency # Hz
        SAMPLING_TIME = 1/RESAMPLING_FREQUENCY

        # Chunks of videos to process
        STARTING_TIME = starting_time
        END_TIME = end_time

        NUM_SAMPLES = round((END_TIME-STARTING_TIME)*RESAMPLING_FREQUENCY + 1) # +1 because the starting time includes a sample
        print("Each numpy array will be resampled to", NUM_SAMPLES, "samples. Timestamps from",STARTING_TIME, "to", END_TIME, "seconds")

        for user in self.processed.keys(): #[1,2]:
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
                    # Cols contain (playbackTime, q0, q1, q2, q3)
                    timestamp = original_movement[i,0]
                    quaternion = original_movement[i,1:5]
                    
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



    # def filter_subset_movement_data(self, experiment_id = 0, column_to_filter=0, min_value=0, max_value=1):
    #     """
    #     Replace the data for the filtered version according to a column an range 
    #     (including both extreme values)
    #     """

    #     filtered_data = self.original_data[experiment_id]

    #     num_user = len(filtered_data)
    #     for user in range(num_users): #[1,2]:
    #         for video in filtered_data[user].keys():
    #             original_movement = self.movement[user][video]
    #             rows, cols = original_movement.shape

    #             # Find the filtered data
    #             rows_to_keep = np.where( (original_movement[:,column_to_filter]>=min_value) & \
    #                                         (original_movement[:,column_to_filter]<=max_value))
    #             filtered_data = original_movement[rows_to_keep]

    #             # The new array with filtered data
    #             self.movement[user][video] = filtered_data.copy()
    #             print("USER:", user, "VIDEO:", str(video), "| NEW SHAPE:", self.movement[user][video].shape)
    #     return filtered_data


    def get_movement(self, experiment:int, user:int, video:int):
        """
        Returns the nparray of the specified user and video
        :param user: User ID
        :type user: int
        :param video: Video key
        :type video: int

        Example: data.get_movement(1,4)
        """
        return self.original_data[experiment][user][video]


    def get_movement_filtered(self, user, video, experiment_id = 0, column_to_filter=0, min_value=0, max_value=1):
        user_mov = self.original_data[experiment_id][user][video].copy()

        # Find the filtered data
        rows_to_keep = np.where( (user_mov[:,column_to_filter]>=min_value) \
                                    & (user_mov[:,column_to_filter]<=max_value))

        return user_mov[rows_to_keep]



def load_dataset_Tsinghua(labels_filename, timestamps_filename, dataset_filename):
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
    print('\t==== Class to load dataset from https://wuchlei-thu.github.io/')

    print('\tPathfile',sys.argv[0])
    if(len(sys.argv) != 1):
        print('\tFirst parameter',sys.argv[1])

# %%
