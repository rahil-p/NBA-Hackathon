import pandas as pd
import pickle

def read_data(files_dict):
    dfs = []
    for file in sorted(files_dict, reverse=True):
        dfs.append(pd.read_csv(files_dict[file]))
        
    return dfs

def pickle_dump(file_data_array, switch=0):
    if switch == 1:
        for pair in file_data_array:
            pickle.dump(pair[1], open(pair[0],"wb"))
