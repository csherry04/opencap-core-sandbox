import os

import numpy as np
import pandas as pd


BASE = "/Users/callumsherry/opencap-sandboxes/opencap-core-sandbox"

SWAPPED_DIR = os.path.join(BASE, "diagnostics/openpose_swapped")

PATHS = {
    "ref_mot": os.path.join(
        BASE, "tests/opencap-test-data/Data/sync_2-cameras/OutputReference/squats.mot"
    ),
    "diag_mot": os.path.join(
        SWAPPED_DIR, "Kinematics/squats_openpose_swapped.mot"
    ),
    "current_mot": os.path.join(
        BASE,
        "tests/opencap-test-data/Data/sync_2-cameras/OpenSimData/Kinematics/squats.mot",
    ),
    "ref_trc_bad_model_mot": os.path.join(
        BASE,
        "diagnostics/reference_trc_current_model_ik/reference_trc_current_model.mot",
    ),
    "ref_trc_new_model_mot": os.path.join(
        BASE,
        "diagnostics/ref_trc_new_model_ik/ref_trc_new_model.mot",
    ),
}


def read_storage(path):
    with open(path) as f:
        lines = f.readlines()
    start = 0
    for i, line in enumerate(lines):
        if line.strip() == "endheader":
            start = i + 1
            break
    return pd.read_csv(path, sep=r"\s+", skiprows=start)


for key, path in PATHS.items():
    df = read_storage(path)
    print()
    print(key, path)
    print("shape", df.shape)
    print("cols", list(df.columns[:8]))
    for col in [
        "pelvis_tx",
        "pelvis_ty",
        "pelvis_tz",
        "pelvis_tilt",
        "hip_flexion_r",
        "knee_angle_r",
        "ankle_angle_r",
    ]:
        if col in df:
            print(
                col,
                "median",
                round(float(df[col].median()), 4),
                "range",
                round(float(df[col].max() - df[col].min()), 4),
            )

for a, b in [
    ("ref_mot", "diag_mot"),
    ("current_mot", "diag_mot"),
    ("ref_mot", "current_mot"),
    ("ref_mot", "ref_trc_bad_model_mot"),
    ("ref_mot", "ref_trc_new_model_mot"),
    ("diag_mot", "ref_trc_new_model_mot"),
]:
    data_a = read_storage(PATHS[a])
    data_b = read_storage(PATHS[b])
    common = [col for col in data_a.columns if col in data_b.columns and col != "time"]
    frame_count = min(len(data_a), len(data_b))
    err = (
        data_b[common].iloc[:frame_count].to_numpy()
        - data_a[common].iloc[:frame_count].to_numpy()
    )

    print()
    print("DIFF", b, "minus", a, "cols", len(common))
    print(
        "mean abs all",
        round(float(np.nanmean(np.abs(err))), 4),
        "max abs",
        round(float(np.nanmax(np.abs(err))), 4),
    )
    for col in ["pelvis_tx", "pelvis_ty", "pelvis_tz", "hip_flexion_r", "knee_angle_r"]:
        if col in common:
            i = common.index(col)
            print(
                col,
                "median diff",
                round(float(np.nanmedian(err[:, i])), 4),
                "mean abs",
                round(float(np.nanmean(np.abs(err[:, i]))), 4),
            )
