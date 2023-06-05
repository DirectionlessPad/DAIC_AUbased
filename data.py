"""A class for loading and storing data."""
from pathlib import Path
from typing import Dict
import pandas as pd


# def load_concatenated_aud_vid(
#     file_path: Path, feature_type: str, delimeter: str
# ) -> Dict[str, Dict[str, pd.DataFrame]]:
#     """TODO"""
#     if not Path.exists(file_path):
#         print("Directory does not exist. Check input feature directory")
#     loaded_features: Dict = {}
#     generator = (file_path).glob("*")
#     for path in generator:
#         str_path = str(path)
#         start = str_path.rindex("\\")
#         end = str_path.rindex("_")
#         participant_id = str_path[start + 1 : end]
#         full_path = path / (participant_id + feature_type)
#         participant_id_df = pd.read_csv(full_path, sep=delimeter)
#         participant_id_df.columns = participant_id_df.columns.str.replace(" ", "")
#         loaded_features[participant_id] = participant_id_df
#     if not loaded_features:
#         print(
#             "No samples loaded, check the samples are available in the input directory."
#         )
#     return loaded_features


class Dataset:
    """A class that can be used to load and store the dataset."""

    def __init__(self, path):
        self._path = path
        self._daic_labels = self.load_daic_labels(path)
        # self._daic_openface_features = None
        # self._daic_mfcc_features = None
        self._daic_openface_features = self.load_daic_openface_features()
        self._daic_mfcc_features = self.load_daic_mfcc_features()
        # self._min_max_values = self.find_min_max()

    @property
    def path(self):
        "Protection for path variable"
        return self._path

    @property
    def daic_labels(self):
        """Protection for the dataset."""
        return self._daic_labels

    @property
    def daic_openface_features(self):
        """Protection for the dataset."""
        return self._daic_openface_features

    @property
    def daic_mfcc_features(self):
        """Protection for the dataset."""
        return self._daic_mfcc_features

    # @property
    # def min_max_values(self):
    #     """Protection for the dataset."""
    #     return self._min_max_values

    def load_daic_labels(self, path: Path) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Loads the depression labels for all subjects."""
        label_path = path / "daic_labels"
        if not Path.exists(label_path):
            print("Directory does not exist. Check input feature directory.")
        loaded_labels: Dict[str, Dict] = {"dev": {}, "train": {}, "test": {}}
        paths = {
            "dev": label_path / "dev_split.csv",
            "train": label_path / "train_split.csv",
            "test": label_path / "test_split.csv",
        }
        for dataset_split, path in paths.items():
            split_df = pd.read_csv(path)
            split_dict = split_df.to_dict()
            for i in range(len(split_dict["Participant_ID"])):
                participant = str(split_dict["Participant_ID"][i])
                loaded_labels[dataset_split][participant] = {
                    "PHQ_Binary": split_dict["PHQ_Binary"][i],
                    "PHQ_Score": split_dict["PHQ_Score"][i],
                }
        return loaded_labels

    def load_daic_base_features(
        self,
        file_path: Path,
        feature_type: str,
        delimeter: str,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """TODO"""
        if not Path.exists(file_path):
            print("Directory does not exist. Check input feature directory")
        loaded_features: Dict[str, Dict] = {"dev": {}, "train": {}, "test": {}}
        dev_path_generator = (file_path / "dev").glob("*")
        train_path_generator = (file_path / "train").glob("*")
        test_path_generator = (file_path / "test").glob("*")
        generators = {
            "dev": dev_path_generator,
            "train": train_path_generator,
            "test": test_path_generator,
        }
        for dataset_split, subset_dict in loaded_features.items():
            gen = generators[dataset_split]
            for path in gen:
                str_path = str(path)
                start = str_path.rindex("\\")
                end = str_path.rindex("_")
                participant_id = str_path[start + 1 : end]
                full_path = path / (
                    "features/"
                    + participant_id
                    + feature_type  # "_OpenFace2.1.0_Pose_gaze_AUs.csv"
                )
                participant_id_df = pd.read_csv(full_path, sep=delimeter)
                participant_id_df.columns = participant_id_df.columns.str.replace(
                    " ", ""
                )
                subset_dict[participant_id] = participant_id_df
                # loaded_features[dataset_split][participant_id] = feature
            if not subset_dict:
                print(
                    "No samples loaded, check the samples are available in the input directory."
                )
        return loaded_features

    def load_daic_openface_features(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """TODO"""
        feature_path = self.path / "openface_features"
        feature_type = "_OpenFace2.1.0_Pose_gaze_AUs.csv"
        delimeter = ","
        return self.load_daic_base_features(feature_path, feature_type, delimeter)

    def load_daic_mfcc_features(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """TODO"""
        feature_path = self.path / "mfcc_features"
        feature_type = "_OpenSMILE2.3.0_mfcc.csv"
        delimeter = ";"
        return self.load_daic_base_features(feature_path, feature_type, delimeter)

    def load_daic_processed_features(
        self, file_path: Path, feature_type: str, delimeter: str
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """TODO"""
        if not Path.exists(file_path):
            print("Directory does not exist. Check input feature directory")
        loaded_features: Dict = {}
        generator = (file_path).glob("*")
        for path in generator:
            str_path = str(path)
            start = str_path.rindex("\\")
            end = str_path.rindex("_")
            participant_id = str_path[start + 1 : end]
            full_path = path / ("features/" + participant_id + feature_type)
            participant_id_df = pd.read_csv(full_path, sep=delimeter)
            participant_id_df.columns = participant_id_df.columns.str.replace(" ", "")
            loaded_features[participant_id] = participant_id_df
        if not loaded_features:
            print(
                "No samples loaded, check the samples are available in the input directory."
            )
        return loaded_features

    def load_processed_openface(self, subset: str):
        """TODO"""
        feature_path = Path("daic_processed_1/openface_features/" + subset)
        feature_type = "_OpenFace2.1.0_Pose_gaze_AUs.csv"
        delimeter = ","
        return self.load_daic_processed_features(feature_path, feature_type, delimeter)

    def load_processed_mfcc(self, subset: str):
        """TODO"""
        feature_path = Path("daic_processed_1/mfcc_features/" + subset)
        feature_type = "_OpenSMILE2.3.0_mfcc.csv"
        delimeter = ";"
        return self.load_daic_processed_features(feature_path, feature_type, delimeter)

    def find_min_max(self):
        """Finds the minimum and maximum values of all feature for normalisation."""
        # only for openface features at the moment

        # !!As of now this doesn't work correctly!!
        # LEAVE THIS TO DO AFTER PREPROCESSING, some of feature values for success == 0 are ridiculous and will most likely need to be zeroed
        min_max = {
            "min": {
                "pose_Tx": 1000,
                "pose_Ty": 1000,
                "pose_Tz": 1000,
                "pose_Rx": 1000,
                "pose_Ry": 1000,
                "pose_Rz": 1000,
                "gaze_0_x": 1000,
                "gaze_0_y": 1000,
                "gaze_0_z": 1000,
                "gaze_1_x": 1000,
                "gaze_1_y": 1000,
                "gaze_1_z": 1000,
                "gaze_angle_x": 1000,
                "gaze_angle_y": 1000,
                "AU01_r": 1000,
                "AU02_r": 1000,
                "AU04_r": 1000,
                "AU05_r": 1000,
                "AU06_r": 1000,
                "AU07_r": 1000,
                "AU09_r": 1000,
                "AU10_r": 1000,
                "AU12_r": 1000,
                "AU14_r": 1000,
                "AU15_r": 1000,
                "AU17_r": 1000,
                "AU20_r": 1000,
                "AU23_r": 1000,
                "AU25_r": 1000,
                "AU26_r": 1000,
                "AU45_r": 1000,
                "pcm_fftMag_mfcc[0]": 1000,
                "pcm_fftMag_mfcc[1]": 1000,
                "pcm_fftMag_mfcc[2]": 1000,
                "pcm_fftMag_mfcc[3]": 1000,
                "pcm_fftMag_mfcc[4]": 1000,
                "pcm_fftMag_mfcc[5]": 1000,
                "pcm_fftMag_mfcc[6]": 1000,
                "pcm_fftMag_mfcc[7]": 1000,
                "pcm_fftMag_mfcc[8]": 1000,
                "pcm_fftMag_mfcc[9]": 1000,
                "pcm_fftMag_mfcc[10]": 1000,
                "pcm_fftMag_mfcc[11]": 1000,
                "pcm_fftMag_mfcc[12]": 1000,
                "pcm_fftMag_mfcc_de[0]": 1000,
                "pcm_fftMag_mfcc_de[1]": 1000,
                "pcm_fftMag_mfcc_de[2]": 1000,
                "pcm_fftMag_mfcc_de[3]": 1000,
                "pcm_fftMag_mfcc_de[4]": 1000,
                "pcm_fftMag_mfcc_de[5]": 1000,
                "pcm_fftMag_mfcc_de[6]": 1000,
                "pcm_fftMag_mfcc_de[7]": 1000,
                "pcm_fftMag_mfcc_de[8]": 1000,
                "pcm_fftMag_mfcc_de[9]": 1000,
                "pcm_fftMag_mfcc_de[10]": 1000,
                "pcm_fftMag_mfcc_de[11]": 1000,
                "pcm_fftMag_mfcc_de[12]": 1000,
                "pcm_fftMag_mfcc_de_de[0]": 1000,
                "pcm_fftMag_mfcc_de_de[1]": 1000,
                "pcm_fftMag_mfcc_de_de[2]": 1000,
                "pcm_fftMag_mfcc_de_de[3]": 1000,
                "pcm_fftMag_mfcc_de_de[4]": 1000,
                "pcm_fftMag_mfcc_de_de[5]": 1000,
                "pcm_fftMag_mfcc_de_de[6]": 1000,
                "pcm_fftMag_mfcc_de_de[7]": 1000,
                "pcm_fftMag_mfcc_de_de[8]": 1000,
                "pcm_fftMag_mfcc_de_de[9]": 1000,
                "pcm_fftMag_mfcc_de_de[10]": 1000,
                "pcm_fftMag_mfcc_de_de[11]": 1000,
                "pcm_fftMag_mfcc_de_de[12]": 1000,
            },
            "max": {
                "pose_Tx": 0,
                "pose_Ty": 0,
                "pose_Tz": 0,
                "pose_Rx": 0,
                "pose_Ry": 0,
                "pose_Rz": 0,
                "gaze_0_x": 0,
                "gaze_0_y": 0,
                "gaze_0_z": 0,
                "gaze_1_x": 0,
                "gaze_1_y": 0,
                "gaze_1_z": 0,
                "gaze_angle_x": 0,
                "gaze_angle_y": 0,
                "AU01_r": 0,
                "AU02_r": 0,
                "AU04_r": 0,
                "AU05_r": 0,
                "AU06_r": 0,
                "AU07_r": 0,
                "AU09_r": 0,
                "AU10_r": 0,
                "AU12_r": 0,
                "AU14_r": 0,
                "AU15_r": 0,
                "AU17_r": 0,
                "AU20_r": 0,
                "AU23_r": 0,
                "AU25_r": 0,
                "AU26_r": 0,
                "AU45_r": 0,
                "pcm_fftMag_mfcc[0]": -1000,
                "pcm_fftMag_mfcc[1]": -1000,
                "pcm_fftMag_mfcc[2]": -1000,
                "pcm_fftMag_mfcc[3]": -1000,
                "pcm_fftMag_mfcc[4]": -1000,
                "pcm_fftMag_mfcc[5]": -1000,
                "pcm_fftMag_mfcc[6]": -1000,
                "pcm_fftMag_mfcc[7]": -1000,
                "pcm_fftMag_mfcc[8]": -1000,
                "pcm_fftMag_mfcc[9]": -1000,
                "pcm_fftMag_mfcc[10]": -1000,
                "pcm_fftMag_mfcc[11]": -1000,
                "pcm_fftMag_mfcc[12]": -1000,
                "pcm_fftMag_mfcc_de[0]": -1000,
                "pcm_fftMag_mfcc_de[1]": -1000,
                "pcm_fftMag_mfcc_de[2]": -1000,
                "pcm_fftMag_mfcc_de[3]": -1000,
                "pcm_fftMag_mfcc_de[4]": -1000,
                "pcm_fftMag_mfcc_de[5]": -1000,
                "pcm_fftMag_mfcc_de[6]": -1000,
                "pcm_fftMag_mfcc_de[7]": -1000,
                "pcm_fftMag_mfcc_de[8]": -1000,
                "pcm_fftMag_mfcc_de[9]": -1000,
                "pcm_fftMag_mfcc_de[10]": -1000,
                "pcm_fftMag_mfcc_de[11]": -1000,
                "pcm_fftMag_mfcc_de[12]": -1000,
                "pcm_fftMag_mfcc_de_de[0]": -1000,
                "pcm_fftMag_mfcc_de_de[1]": -1000,
                "pcm_fftMag_mfcc_de_de[2]": -1000,
                "pcm_fftMag_mfcc_de_de[3]": -1000,
                "pcm_fftMag_mfcc_de_de[4]": -1000,
                "pcm_fftMag_mfcc_de_de[5]": -1000,
                "pcm_fftMag_mfcc_de_de[6]": -1000,
                "pcm_fftMag_mfcc_de_de[7]": -1000,
                "pcm_fftMag_mfcc_de_de[8]": -1000,
                "pcm_fftMag_mfcc_de_de[9]": -1000,
                "pcm_fftMag_mfcc_de_de[10]": -1000,
                "pcm_fftMag_mfcc_de_de[11]": -1000,
                "pcm_fftMag_mfcc_de_de[12]": -1000,
            },
        }
        for _, subjects in self.daic_openface_features.items():
            for subject, dataframe in subjects.items():
                for feature, _ in min_max["min"].items():
                    min_value = dataframe[feature].min()
                    max_value = dataframe[feature].max()
                    if min_value < -1000:
                        print(subject)
                    if max_value > 1000:
                        print(subject)
                    if min_value < min_max["min"][feature]:
                        min_max["min"][feature] = min_value
                    if max_value > min_max["max"][feature]:
                        min_max["max"][feature] = max_value
        return min_max
