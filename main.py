"""TODO"""
from data import Dataset
from pathlib import Path
from preprocess import mfcc_timestamp_average
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.preprocessing import sequence

daic_path = Path("daic_dataset")
processed_storage = "daic_processed_1"

DAIC = Dataset(daic_path)

averaged_mfcc = mfcc_timestamp_average(
    DAIC.daic_mfcc_features,
    DAIC.daic_openface_features,
    processed_storage,
)
# breakpoint()
