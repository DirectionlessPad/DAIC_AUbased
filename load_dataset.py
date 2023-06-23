""""""
from pathlib import Path
import pandas as pd
import numpy as np


### load original daic data
def load_original_daic(file_path, file_type):
    """TODO
    file_type: AUs, features, gaze, pose
    """
    file_path = Path(file_path)
    if not Path.exists(file_path):
        print("Directory does not exist. Check input feature directory")
    loaded_features = {}
    for participant_id in range(300, 500):
        participant_id = str(participant_id)
        full_path = file_path / (
            participant_id + "/" + participant_id + "_CLNF_" + file_type + ".txt"
        )
        if Path.exists(full_path):
            participant_df = pd.read_csv(full_path, sep=",", low_memory=False)
            participant_df.columns = participant_df.columns.str.replace(" ", "")
            participant_df.drop(
                columns=["frame", "timestamp", "confidence", "success"], inplace=True
            )
            loaded_features[participant_id] = participant_df
    if not loaded_features:
        print(
            "No samples loaded, check the samples are available in the input directory."
        )
    return loaded_features


def load_labels(label_path):
    """TODO"""
    loaded_labels = {
        "dev": [],
        "train": [],
        "test": [],
    }
    paths = {
        "dev": label_path / "dev_split_Depression_AVEC2017.csv",
        "train": label_path / "train_split_Depression_AVEC2017.csv",
        "test": label_path / "full_test_split.csv",
    }
    for subset, path in paths.items():
        subset_df = pd.read_csv(path)
        subset_df.columns = subset_df.columns.str.lower()
        # subset_dict = subset_df.to_dict()
        if subset == "test":
            column_name = "phq_binary"
        else:
            column_name = "phq8_binary"
        for i in range(len(subset_df["participant_id"])):
            # participant = str(subset_df["participant_id"][i])
            loaded_labels[subset].append(subset_df[column_name][i])
            # to reimplement this code would require initilisation
            # of loaded labels as a dict of dicts. And change keys.
            # loaded_labels[subset][participant] = {
            #     "PHQ_Binary": subset_dict["PHQ_Binary"][i],
            #     "PHQ_Score": subset_dict["PHQ_Score"][i],
            # }
    return loaded_labels
