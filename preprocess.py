"""Functions for preprocessing data."""
from pathlib import Path
from typing import Dict, List
import pandas as pd

# see your notes for week starting 8th of May for concerns about "success" == 0 on some frames


def mfcc_timestamp_average(
    mfcc_features: Dict[str, Dict[str, pd.DataFrame]],
    openface_features: Dict[str, Dict[str, pd.DataFrame]],
    save_to_directory_name: str,
) -> List[Dict[str, Dict[str, pd.DataFrame]]]:
    """Function that averages the mfcc features for each video frame."""
    averaged_mfcc_features: Dict[str, Dict] = {"dev": {}, "train": {}, "test": {}}
    for subset, subjects in mfcc_features.items():
        for subject_id, subject_df in subjects.items():
            video_frames = openface_features[subset][subject_id].shape[0]
            audio_frames = subject_df.shape[0]
            audio_index = 0
            averaged_mfcc_features[subset][subject_id] = pd.DataFrame(
                columns=subject_df.columns
            )
            averaged_mfcc_features[subset][subject_id].drop(
                columns=["name", "frameTime"], inplace=True
            )
            for video_index in range(1, video_frames):
                lower_video_timestamp = openface_features[subset][subject_id][
                    "timestamp"
                ][video_index - 1]
                upper_video_timestamp = openface_features[subset][subject_id][
                    "timestamp"
                ][video_index]
                running_sum = 0
                count = 0
                while (
                    audio_index < audio_frames
                    and subject_df["frameTime"][audio_index] < upper_video_timestamp
                ):
                    if subject_df["frameTime"][audio_index] < lower_video_timestamp:
                        audio_index += 1
                    else:
                        mfcc_sample = subject_df.iloc[audio_index, 2:]  # type: ignore
                        running_sum = running_sum + mfcc_sample
                        audio_index += 1
                        count += 1
                try:
                    average_mfcc = running_sum / count
                    averaged_mfcc_features[subset][subject_id].loc[
                        video_index - 1
                    ] = average_mfcc
                except:
                    continue
                ### Going to end up with a couple frames less than in video files. ###
                ### Might have to drop a frame or two of video. ###
            frames_to_drop = (
                video_index - averaged_mfcc_features[subset][subject_id].shape[0]
            )
            for i in range(0, frames_to_drop):
                openface_features[subset][subject_id].drop(
                    video_index - (i + 1), inplace=True
                )
            Path(
                save_to_directory_name
                + "/mfcc_features/"
                + subset
                + "/"
                + subject_id
                + "_P/features"
            ).mkdir(parents=True, exist_ok=True)
            averaged_mfcc_features[subset][subject_id].to_csv(
                save_to_directory_name
                + "/mfcc_features/"
                + subset
                + "/"
                + subject_id
                + "_P/features/"
                + subject_id
                + "_OpenSMILE2.3.0_mfcc.csv",
                index=False,
                sep=";",
            )
            Path(
                save_to_directory_name
                + "/openface_features/"
                + subset
                + "/"
                + subject_id
                + "_P/features"
            ).mkdir(parents=True, exist_ok=True)
            openface_features[subset][subject_id].to_csv(
                save_to_directory_name
                + "/openface_features/"
                + subset
                + "/"
                + subject_id
                + "_P/features/"
                + subject_id
                + "_OpenFace2.1.0_Pose_gaze_AUs.csv",
                index=False,
                sep=",",
            )
            # breakpoint()
            print(subject_id)
    return [openface_features, averaged_mfcc_features]
