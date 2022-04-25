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

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
THIS_PATH = str(os.path.dirname(os.path.abspath(__file__)))

# Import data manipulation libraries
from copy import deepcopy

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
    folder_data = ""    # Root folder of the original dataset
    index_file = ""     # Filepath for the json file containing the index

    # Data Variables
    dataset_index = {}
    events = {}

    def __init__(self, folder_path):
        """
        Initializes object that analyzes dataset

        :param folder_path: Input folder with data
        :type folder_path: str
        :param dictionary_path: Output path with directory from 
        :type dictionary_path: str
        """
        self.folder_data = folder_path

    def create_index(self, json_path = ""):
        """
        Analyzes the folder to see which files are available.

        :param json_path: Folder where the summary json will be stored
        :type json_path: str
        :return: Dictionary with the index and events of the experiment
        :rtype: dict
        """
        
        # Define the filename for the output index file
        if (json_path == ""):
            json_path = os.path.join(self.folder_data, self.JSON_INDEX_FILENAME)
        else:
            json_path = os.path.join(json_path, self.JSON_INDEX_FILENAME)

        # Index filepath
        self.index_file = json_path
        print("Output json filepath: ", self.index_file)
    
        # Dictionary to store files
        files_index = {}

        # Look for zip files and extract all in the same directory
        counter_idx = 0
        with os.scandir(self.folder_data) as it:
            for directory in it:
                ### DIRECTORIES AS PARTICIPANTS
                if( not directory.name.startswith(".") and directory.is_dir() ):                    
                    # A folder is equivalent to a participant

                    # Add the participant data to the file index.
                    # The index is a sequential number from `counter_idx`
                    files_index[counter_idx] = deepcopy(self.PARTICIPANT_DATA_DICT)   # Empty dict for data
                    files_index[counter_idx]["folderid"] = directory.name.split("_")[1]

                    print(f"\nDirectory >> {directory.name}")

                    # Store all the events in a new single .csv file
                    post_processed_events = pd.DataFrame( deepcopy(self.PROCESSED_EVENTS_DICT) )
                    post_processed_events_filepath = os.path.join(self.folder_data, directory.name, self.POST_PROCESSED_EVENTS_FILENAME)

                    # Scan participant's dir for specific files
                    with os.scandir(os.path.join(self.folder_data, directory.name)) as it2:
                        for file in it2:
                            
                            ## The session is defined by the filename (without extension)
                            session_name = file.name.split(".")[0]

                            if(file.name.endswith(self.EVENTS_FILE_EXTENSION)):
                                # # File is an EVENT. Read it right away
                                # print(f"\t Event>> {session_name}")
                                dict_events = self.__load_single_event_file_into_dict(os.path.join(self.folder_data, 
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
        utils.create_json(files_index, self.index_file)

        print(f"Json file with index of the dataset was saved in {self.index_file}")

        # Global variable for the index
        self.dataset_index = files_index.copy()

        return 0

    def load_data_from_participant():
        pass
    
    
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
    input_folder_path = args.inputpath 
    print(f"Analyzing folder {input_folder_path}")

    output_index_json = args.outindexfile if args.outindexfile else ""
    print(f"Storing index in {output_index_json}")

    data_loader_etl2 = DatasetEmteqLabsv2(os.path.join(THIS_PATH,input_folder_path))
    data_loader_etl2.create_index(output_index_json)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--inputpath", type=str, required=True, help=f"Path to the dataset EMTEQ")
    parser.add_argument("-o","--outindexfile", type=str, required=False, help=f"Path to store index")

    data_loader_etl2 = DatasetEmteqLabsv2(os.path.join(THIS_PATH,"../../datasets/ETL2/"))
    data_loader_etl2.create_index()

    # try:
    #     args = parser.parse_args()
    #     main(args)

    # except Exception as e:
    #     help()
    #     print(f"{e}")