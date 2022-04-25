#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Luis Quintero | luisqtr.com
# Created Date: 2022/04
# =============================================================================
"""
Functions to load the datasets from Emteq Labs regarding behavioral analysis
in VR environments.

The functions loads dataset collected from the EmteqPRO mask, and code is based
on the scripts from: https://github.com/emteqlabs/demo-analysis-scripts

"""
# =============================================================================
# Imports
# =============================================================================

from ast import expr_context
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
THIS_PATH = str(os.path.dirname(os.path.abspath(__file__)))

# Import data manipulation libraries
from copy import deepcopy
from enum import Enum

# Import scientific 
import numpy as np
import pandas as pd

import utils
from quaternion_math import *

# =============================================================================
# Main
# =============================================================================


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



class SessionSegment(Enum):
    """
    Enum to access the dictionary with the data per video
    """
    FastMovement = "fast_movement"
    SlowMovement = "slow_movement"
    video1 = "video_1"
    video2 = "video_2"
    video3 = "video_3"
    video4 = "video_4"
    video5 = "video_5"

    def __str__(self):
        return super().value.__str__()


class DatasetEmteqLabsv2():
    """
    This class loads the data from the remote video experiment version 2.
    
    19 datasets, marked as `###_v2`. A session consists of three segments: 
        1) slow movement, 2) fast movement, and 3) videos.

    These datasets also contain an **expression calibration stage** where 
    the participants were asked to perform 3 repetitions of: `smile`, `frown`, 
    `surprise`. The calibration was performed at the start of the `slow_movement_segment`.
    
    The output of this class is a configured object that loads the list of available
    participants and their data. However, it does not load the whole dataset of the user
    because each dataset can be around 500MB. To load each participant's dataset
    use the method load_data_from_participant().

    Download link for original dataset: Request to Emteq Labs (emteqlabs.com/)

    The output of the data loading process is:
        - data[0]["folderid"] > Returns the id of the participant's data
        - data[0]["events"]["session"] > Returns a pandas dataframe with the events
                    that occurred in the session "session" of the user `0`
        - data[0]["data"]["session"] > Returns a `string` indicating where the data
                    is located for the user `0` and the session `session`. The data needs
                    to be loaded individually because each file >50MB (~8GB in total)

    `session` is the key of the part of the experiment: 
                ["fast_movement", "slow_movement", "video_1", 
                "video_2", "video_3", "video_4", "video_5"]
    """

    
    # Structure of the dataset containing the data.
    # The values of the dict correspond to filename where data is stored
    EXPERIMENT_SESSIONS_DICT = {"fast_movement": "",
              "slow_movement":"",
              "video_1":"",
              "video_2":"",
              "video_3":"",
              "video_4": "",
              "video_5":"",
             }

    PROCESSED_EVENTS_DICT = {
            "Session": [],
            "Timestamp":[],
            "Event":[],
        }

    # Structure of the filepaths per user
    PARTICIPANT_DATA_DICT = {
        "folderid": "",               # Name of the folder
        "events": deepcopy(EXPERIMENT_SESSIONS_DICT),  # Events are loaded immediately
        "data": deepcopy(EXPERIMENT_SESSIONS_DICT),    # Data is just a reference to the file (>50MB/each file)
    }

    ### CONSTANTS
    DATA_FILE_EXTENSION = ".csv"
    EVENTS_FILE_EXTENSION = ".json"

    # OUTPUT VALUES
    POST_PROCESSED_EVENTS_FILENAME = "events_postprocessed.csv"
    JSON_INDEX_FILENAME = "data_tree_index.json"

    # MAIN VARIABLES TO ACCESS DATA
    # Filenames
    folder_data_path = ""    # Root folder of the original dataset
    index_file_path = ""     # Filepath for the json file containing the index

    # Data Variables
    index = None            # Dictionary with the dataset's index
    events = None           # Dictionary of Pandas DataFrame with Events
    data = None             # Dictionary of Pandas DataFrame with Emteq Data

    def __init__(self, folder_path):
        """
        Initializes object that analyzes dataset

        :param folder_path: Input folder with data
        :type folder_path: str
        :param dictionary_path: Output path with directory from 
        :type dictionary_path: str
        """
        self.folder_data_path = folder_path
        self.index_file_path = os.path.join(self.folder_data_path, self.JSON_INDEX_FILENAME)

        self.load_or_create_index()
        return

    def load_or_create_index(self):
        """
        Analyzes the folder to see which files are available.
        Enables access to the variable `self.index`, which contains a 
        dictionary with path to key event and data files.
        It also creates the json file at the root of the dataset.

        :return: Nothing
        :rtype: None
        """

        # Entry condition
        if(self.__load_index_file() is not None):
            print("Index already exists: Loading from ", self.index_file_path)
            return
        
        ##### Create index from the dataset folder
        print("There is no index yet! Creating it in ", self.index_file_path)
    
        # Dictionary to store files
        files_index = {}

        # Look for zip files and extract all in the same directory
        counter_idx = 0
        with os.scandir(self.folder_data_path) as it:
            for directory in it:
                ### DIRECTORIES AS PARTICIPANTS
                if( not directory.name.startswith(".") and directory.is_dir() ):                    
                    # A folder is equivalent to a participant

                    # Add the participant data to the file index.
                    # The index is a sequential number from `counter_idx`
                    files_index[counter_idx] = deepcopy(self.PARTICIPANT_DATA_DICT)   # Empty dict for data
                    files_index[counter_idx]["folderid"] = directory.name.split("_")[1]

                    # print(f"\nDirectory >> {directory.name}")

                    # Store all the events in a new single .csv file
                    post_processed_events = pd.DataFrame( deepcopy(self.PROCESSED_EVENTS_DICT) )
                    post_processed_events_filepath = os.path.join(self.folder_data_path, directory.name, self.POST_PROCESSED_EVENTS_FILENAME)

                    # Scan participant's dir for specific files
                    with os.scandir(os.path.join(self.folder_data_path, directory.name)) as it2:
                        for file in it2:
                            
                            ## The session is defined by the filename (without extension)
                            session_name = file.name.split(".")[0]

                            if(file.name.endswith(self.EVENTS_FILE_EXTENSION)):
                                # File is an EVENT. Read it right away

                                # print(f"\t Event>> {session_name}")

                                dict_events = self.__load_single_event_file_into_dict(os.path.join(self.folder_data_path, 
                                                                                            directory.name, 
                                                                                            file.name), session_name)
                                # Attach events to the file
                                this_event_df = pd.DataFrame(deepcopy(dict_events))
                                post_processed_events = pd.concat([post_processed_events, this_event_df], ignore_index=True)

                            elif (file.name.endswith(self.DATA_FILE_EXTENSION) and (session_name in self.EXPERIMENT_SESSIONS_DICT.keys()) ):
                                # # File is DATA, too large, just store the path.
                                # print(f"\t Data>> {session_name}")
                                files_index[counter_idx]["data"][session_name] = os.path.join(directory.name, file.name)

                    # Save all event files in a single csv
                    post_processed_events.to_csv(post_processed_events_filepath, index=False)
                    files_index[counter_idx]["events"] = os.path.join(directory.name, self.POST_PROCESSED_EVENTS_FILENAME)
                    print(f"\t Events compiled in {post_processed_events_filepath}")

                    # Prepare for next data
                    counter_idx = counter_idx + 1

        print(f"A total of {counter_idx} folders were found in the dataset")

        # Store the files in a JSON
        utils.create_json(files_index, self.index_file_path)

        print(f"Json file with index of the dataset was saved in {self.index_file_path}")

        # Global variable for the index
        self.index = files_index.copy()
        return

    def __load_index_file(self):
        """
        Loads the dictionary with the index file into memory.
        If error, returns None
        """
        try:  
            self.index = utils.load_json(self.index_file_path)
            self.index = {int(k):v for k,v in self.index.items()}
            return 0
        except:
            return None

    def load_event_files(self):
        """
        Loads the dictionary containing the events from each participant.
        Access all the events in a DataFrame from the participant 0 as:
            - dataset_loader.events[0]
        
        :return: Events during all the experiment
        :rtype: Pandas DataFrame
        """
        if self.index is None:
            print("There is no index file loaded, loading index file...")
            self.load_or_create_index()
        else:
            ### Load events in dictionary
            self.events = {}
            for id, evt_path in self.index.items():
                # Iterate over participants
                self.events[id] = pd.read_csv(os.path.join(self.folder_data_path, evt_path["events"]))
        return

    def load_data_from_participant(self, participant_idx:int, session_part:str):
        """
        Loads the dictionary containing the events from each participant.
        Access all the events in a DataFrame from the participant 0 as:
            - dataset_loader.events[0]
        
        :param participant_idx: Index of the participant (generally from 0 to 15)
        :type participant_idx: int
        :param session_part: Unique key indicating which session segment to access.
        :type session_part: str
        :return: Large dataframe containing all the physiological data as recorded by the EmteqMask
        :rtype: Pandas DataFrame
        """
        return pd.read_csv( os.path.join(self.folder_data_path, self.index[participant_idx]["data"][session_part]), 
                            error_bad_lines=False)
    

    def __load_single_event_file_into_dict(self, event_filepath, session_name):
        """
        Loads a file with events into a structured dictionary
        """
        dict_from_json = utils.load_json(event_filepath)
        
        # Transform to simpler dict compatible with Pandas
        organized_dict = deepcopy(self.PROCESSED_EVENTS_DICT)

        for event_info in dict_from_json:
            for k,v in event_info.items():
                organized_dict[k].append(v)

        # Repeat the session name as much as needed. It facilitates filtering
        organized_dict["Session"] = [session_name] * len(organized_dict["Timestamp"])

        return organized_dict.copy()

    ########################################### OLD METHODS TO BE REPLACED

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

############################
#### ENTRY POINT
############################

import sys, argparse

def help():
    m = f"""
        Experiment execution with dataset ''
        Parameters:
            
        """
    # print(m)
    return m

def main(args):
    input_folder_path = args.datasetroot 
    print(f"Analyzing folder {input_folder_path}")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--datasetroot", type=str, required=True, help=f"Path to the dataset EMTEQ")

    # args = parser.parse_args()
    # main(args)

    print(" >>>> TESTING MANUALLY")
    data_loader_etl2 = DatasetEmteqLabsv2(os.path.join(THIS_PATH,"../../datasets/ETL2/"))

