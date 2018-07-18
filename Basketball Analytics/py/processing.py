import pandas as pd

def read_data(files_dict):
    dfs = []
    for file in sorted(files_dict, reverse=True):
        dfs.append(pd.read_table(files_dict[file]))
    return dfs

def sort_pbp(pbp):
    pbp.sort_values(['Game_id', 'Period', 'PC_Time', 'WC_Time', 'Event_Num'],
                    ascending=[True, True, False, True, True])
    pbp = pbp.reset_index(drop=True)
    return pbp

def sort_ec(ec):
    ec.sort_values(['Event_Msg_Type', 'Action_Type'])
    ec = ec.reset_index(drop=True)
    return ec

def get_output(stats_dict):
    stats = []
    for key1 in stats_dict.keys():
        for key2 in stats_dict[key1].keys():
            value = stats_dict[key1][key2]
            stats.append([key1, key2, value])
    output_df = pd.DataFrame(stats, columns=['Game_ID', 'Player_ID', 'Player_Plus/Minus'])

    return output_df
