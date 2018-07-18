import pandas as pd

def read_data(files_dict):
    dfs = []
    for file in sorted(files_dict, reverse=True):
        dfs.append(pd.read_table(files_dict[file]))
    return dfs

def sort_pbp(pbp):
    pbp.sort_values(['Game_id', 'Period', 'PC_Time', 'WC_Time', 'Event_Num'],
                    ascending=[True, True, False, True, True])
    pbp.reset_index(drop=True)
    return pbp

def sort_ec(ec):
    ec.sort_values(['Event_Msg_Type', 'Action_Type'])
    ec = ec.reset_index(drop=True)
    return ec
