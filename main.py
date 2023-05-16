"""TODO"""
from data import Dataset
from pathlib import Path
from preprocess import mfcc_timestamp_average

daic_path = Path("daic_dataset")

DAIC = Dataset(daic_path)

averaged_mfcc = mfcc_timestamp_average(
    DAIC.daic_mfcc_features, DAIC.daic_openface_features
)
