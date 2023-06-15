from zipfile import ZipFile
from pathlib import Path
import requests

dev = [
    302,
    307,
    331,
    335,
    346,
    367,
    377,
    381,
    382,
    388,
    389,
    390,
    395,
    403,
    404,
    406,
    413,
    417,
    418,
    420,
    422,
    436,
    439,
    440,
    451,
    458,
    472,
    476,
    477,
    482,
    483,
    484,
    489,
    490,
    492,
]
test = [
    300,
    301,
    306,
    308,
    309,
    311,
    314,
    323,
    329,
    332,
    334,
    337,
    349,
    354,
    359,
    361,
    365,
    373,
    378,
    384,
    387,
    396,
    399,
    405,
    407,
    408,
    410,
    411,
    421,
    424,
    431,
    432,
    435,
    438,
    442,
    450,
    452,
    453,
    461,
    462,
    465,
    466,
    467,
    469,
    470,
    480,
    481,
]
train = [
    303,
    304,
    305,
    310,
    312,
    313,
    315,
    316,
    317,
    318,
    319,
    320,
    321,
    322,
    324,
    325,
    326,
    327,
    328,
    330,
    333,
    336,
    338,
    339,
    340,
    341,
    343,
    344,
    345,
    347,
    348,
    350,
    351,
    352,
    353,
    355,
    356,
    357,
    358,
    360,
    362,
    363,
    364,
    366,
    368,
    369,
    370,
    371,
    372,
    374,
    375,
    376,
    379,
    380,
    383,
    385,
    386,
    391,
    392,
    393,
    397,
    400,
    401,
    402,
    409,
    412,
    414,
    415,
    416,
    419,
    423,
    425,
    426,
    427,
    428,
    429,
    430,
    433,
    434,
    437,
    441,
    443,
    444,
    445,
    446,
    447,
    448,
    449,
    454,
    455,
    456,
    457,
    459,
    463,
    464,
    468,
    471,
    473,
    474,
    475,
    478,
    479,
    485,
    486,
    487,
    488,
    491,
]

for i in range(300, 500):
    print("fetching " + str(i))
    if i in dev:
        location = "dev"
    elif i in test:
        location = "test"
    elif i in train:
        location = "train"
    else:
        location = ""
    data_sample = str(i)
    # file_path = Path("Data/" + data_sample + "_P")
    # if file_path.is_file():
    #     file_i_want = (
    #         data_sample
    #         + "_P/features/"
    #         + data_sample
    #         + "_OpenFace2.1.0_Pose_gaze_AUs.csv"
    #     )

    #     with tarfile.open(file_path) as tar:
    #         tar.extract(member=file_i_want, path=Path("openface_features/" + location))
    file_AUs = data_sample + "_CLNF_AUs.txt"
    file_2d = data_sample + "_CLNF_features.txt"
    file_gaze = data_sample + "_CLNF_gaze.txt"
    file_pose = data_sample + "_CLNF_pose.txt"
    files_i_want = [file_AUs, file_2d, file_gaze, file_pose]
    if not Path.exists(Path("original_daic/" + location + "/" + data_sample)):
        try:
            URL = "https://dcapswoz.ict.usc.edu/wwwdaicwoz/" + data_sample + "_P.zip"
            req = requests.get(URL, allow_redirects=True, timeout=900)
            if req.headers.get("content-type") == "application/zip":
                temp_file = data_sample + "_P.zip"
                open(temp_file, "wb").write(req.content)
                file_path = Path(temp_file)
                with ZipFile(file_path, "r") as zip:
                    for file in files_i_want:
                        zip.extract(
                            file, path="original_daic/" + location + "/" + data_sample
                        )
                # with tarfile.open(file_path) as tar:
                #     tar.extract(
                #         member=file_i_want, path=Path("openface_features/" + location)
                #     )
                file_path.unlink()
        except requests.exceptions.ConnectionError:
            print(
                "FAILED TO CONNECT AND DOWNLOAD SAMPLE NUMBER "
                + str(i)
                + " RERUN SCRIPT!!"
            )
