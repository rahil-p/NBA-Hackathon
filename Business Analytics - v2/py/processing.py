import pandas as pd

def read_data(files_dict):
    dfs = []
    for file in sorted(files_dict, reverse=True):
        dfs.append(pd.read_csv(files_dict[file]))
    return dfs
