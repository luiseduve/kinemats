#%% # Add to path
from pathlib import Path
import sys,os
this_path = str(Path().absolute())+"/"
print("File Path:", this_path)
sys.path.append(os.path.join(Path().absolute(), "src"))

# %%

# LIBRARIES

from data_loader.dataset_videos import DatasetHeadMovIMT, VideoList
from data_loader import utils

import pandas as pd

# %%

"""
Is the script executed with SnakeMake `with_sm`?
"""
with_sm = False
try:
    val = snakemake.input[0]
    with_sm = True
    print('Running with SnakeMake')
except NameError as e:
    print('Running without snakemake')
    with_sm = False

"""
INPUTS:
Setup paths either from snakemake or hardcoded relative paths
"""

prefix_filepath = "./" # Compressed dataset

dict_json_name = snakemake.input[0] if with_sm else prefix_filepath+"temp/files_index_per_user.json"
movement_data_filename= snakemake.input[1] if with_sm else prefix_filepath+"temp/hmd_movements.pickle"
general_data_filename = snakemake.input[2] if with_sm else prefix_filepath+"dataset/demographics_IMT.csv"

"""
OUTPUTS:
Setup paths either from snakemake or hardcoded relative paths
"""

# Path of JSON dictionary used to store the data per user
sampling_stats_filename = snakemake.output[0] if with_sm else prefix_filepath+"dataset/sampling_stats_IMT.csv" # Original sampling stats
movement_resampled_data_filename = snakemake.output[1] if with_sm else prefix_filepath+"temp/hmd_movements_resampled.pickle"


# %%

""" #################
######### MAIN SCRIPT
################# """

"""
Convert the original data into constant samples
"""

# Dataset container
data = DatasetHeadMovIMT('',dict_json_name)

# Load dataset demographics and movement data
data.general = pd.read_csv(general_data_filename)
#data.general is a pd.DataFrame
print("File with original demographics info was successfully loaded from",general_data_filename)
data.movement = utils.load_pickle(movement_data_filename) 
#data.movement[0]                   = returns a dictionary {'video_id': <nparray>}, video_id is got automatically from the Enum VideoList
#data.movement[0][VideoList.Paris]  = returns a nparray with hmd movement
print("File with original data movement was successfully loaded from",movement_data_filename)

### Delete videos that are used for training each user in VR
videos_to_delete = [VideoList.Elephant, VideoList.Rhino] ####!!!

# Delete head-movement data of specific video keys
data.delete_data_from_videos(videos_to_delete)
print("Removing data from specific video keys... Done!")

### Summary of original sampling frequencies
sampling_stats = data.create_original_sampling_summary()
sampling_stats.to_csv(sampling_stats_filename)

### Create resample dataset
data.resample_movement(sampling_frequency = 30, starting_time = 5, end_time = 35)
# Create pickle file with resampled head-movement data
utils.create_pickle(data.movement, movement_resampled_data_filename)

print("End")
