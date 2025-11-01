import pandas as pd
import json
import numpy as np

def get_mft(file_pth):
    mft_df = pd.read_csv(file_pth)
    mft_df.insert(0, "statement_id", range(0, len(mft_df)))
    mft_list = [(i, item)  for i, item in enumerate(mft_df["text"].to_list())]
    return mft_list, mft_df

def get_value10(file_pth):
    value10_df = pd.read_csv(file_pth)
    if 'statement_id' not in value10_df.columns.tolist():
        value10_df.insert(0, "statement_id", range(0, len(value10_df)))
    value10_list = [(i, item) for i, item in enumerate(value10_df["text"].to_list())]
    value10_df = value10_df.reset_index(drop=True)
    value10_df = value10_df.rename(columns={"text": "sentence", "SECURITY": "security_label", "BENEVOLENCE": "benevolence_label",
                                    "STIMULATION": "stimulation_label", "UNIVERSALISM": "universalism_label", "CONFORMITY": "conformity_label", "HEDONISM": "hedonism_label", "POWER": "power_label","TRADITION": "tradition_label","ACHIEVEMENT": "achievement_label","SELF-DIRECTION": "selfdirection_label"})
    return value10_list, value10_df

def get_value20(file_pth):
    # Expecting columns: Premise, Stance, Conclusion, and all 10 value dimensions: SECURITY,BENEVOLENCE,STIMULATION,UNIVERSALISM,CONFORMITY,HEDONISM,POWER,TRADITION,ACHIEVEMENT,SELF-DIRECTION
    value20_df = pd.read_csv(file_pth)
    if 'statement_id' not in value20_df.columns.tolist():
        value20_df.insert(0, "statement_id", range(0, len(value20_df)))
    value20_list = [(i, item) for i, item in enumerate(value20_df["text"].to_list())]
    value20_df = value20_df.reset_index(drop=True)
    value20_df = value20_df.rename(columns={"text": "sentence", "SECURITY": "security_label", "BENEVOLENCE": "benevolence_label",
                                    "STIMULATION": "stimulation_label", "UNIVERSALISM": "universalism_label", "CONFORMITY": "conformity_label", "HEDONISM": "hedonism_label", "POWER": "power_label","TRADITION": "tradition_label","ACHIEVEMENT": "achievement_label","SELF-DIRECTION": "selfdirection_label"})
    return value20_list, value20_df

def get_mac(file_pth):
    mac_df = pd.read_csv(file_pth)
    if 'statement_id' not in mac_df.columns.tolist():
        mac_df.insert(0, "statement_id", range(0, len(mac_df)))
    mac_list = [(i, item) for i, item in enumerate(mac_df["text"].to_list())]
    mac_df = mac_df.reset_index(drop=True)
    mac_df = mac_df.rename(columns={"text": "sentence", "Family (HC)": "family_label", "Group (HC)": "group_label",
                                    "Reciprocity (HC)": "reciprocity_label", "Heroism (HC)": "heroism_label", "Deference (HC)": "deference_label", "Fairness (HC)": "fairness_label", "Property (HC)": "property_label"})
    return mac_list, mac_df

def get_common_morality(file_pth):
    cm_df = pd.read_csv(file_pth)
    if 'statement_id' not in cm_df.columns.tolist():
        cm_df.insert(0, "statement_id", range(0, len(cm_df)))
    cm_list = [(i, item) for i, item in enumerate(cm_df["text"].to_list())]
    return cm_list, cm_df


def get_stats(file_pth, mode, dimension = None):
    if mode == "mft":
        moral_stat_list, moral_stat_df = get_mft(file_pth = file_pth)
        return moral_stat_list, moral_stat_df
    elif mode == "mac":
        moral_stat_list, moral_stat_df = get_mac(file_pth = file_pth)
        return moral_stat_list, moral_stat_df
    elif mode == "value10":
        moral_stat_list, moral_stat_df = get_value10(file_pth = file_pth)
        return moral_stat_list, moral_stat_df
    elif mode == "value20":
        moral_stat_list,  moral_stat_df = get_value20(file_pth = file_pth)
        return moral_stat_list,  moral_stat_df
    elif mode == "common_morality":
        moral_stat_list,  moral_stat_df = get_common_morality(file_pth = file_pth)
        return moral_stat_list,  moral_stat_df
    else:
        raise ValueError(f"Unknown mode: {mode}. Supported modes are: mft, mac, value10, value20.")
