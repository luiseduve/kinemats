#%% # Add to path
from pathlib import Path
import sys,os
this_path = str(Path().absolute())+"/"
print("File Path:", this_path)
sys.path.append(os.path.join(Path().absolute(), "src"))

# %%

# LIBRARIES

from data_loader.dataset_videos import DatasetHeadMovIMT
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
dataset_path = snakemake.input[0] if with_sm else prefix_filepath+"dataset/dataset.tar.gz"

"""
OUTPUTS:
Setup paths either from snakemake or hardcoded relative paths
"""

# Path of JSON dictionary used to store the data per user
dict_json_name = snakemake.output[0] if with_sm else prefix_filepath+"temp/files_index_per_user.json"

# Filename of the file containing demographics and HMD movements data
movement_data_filename= snakemake.output[1] if with_sm else prefix_filepath+"temp/hmd_movements.pickle"
general_data_filename = snakemake.output[2] if with_sm else prefix_filepath+"dataset/demographics_IMT.csv"

# %%

""" #################
######### MAIN SCRIPT
################# """

"""
Extracts the whole data from the .tar file
The final pickle can be accessed through
"""

# Dataset container
data = DatasetHeadMovIMT(dataset_path,dict_json_name)

# Create JSON with dictionary of structured data
data.generate_file_index()

files_index = utils.load_json(dict_json_name)
print("Number of users in file index:", len(files_index.keys()))


# Indices with corrupted samples
omit_users = [14, 33, 52, 61, 62]

# Transform the paths in the compressed file into bytes
res = data.uncompress_data(files_index,
                        #debug_users = 15, # Load just a user index
                        # Users ID with empty data
                        list_unprocessed_users = omit_users
                    )

if res==0:
    # Save CSV with dataframe of general data
    data.general.to_csv(general_data_filename)
    utils.create_pickle(data.movement, movement_data_filename)
else:
    print("There was an error uncompressing the data")

print("End")
