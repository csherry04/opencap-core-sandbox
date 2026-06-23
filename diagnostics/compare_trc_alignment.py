import os
import sys

import numpy as np

BASE = "/Users/callumsherry/opencap-sandboxes/opencap-core-sandbox"
sys.path.insert(0, BASE)

import utilsDataman


SWAPPED_DIR = os.path.join(BASE, "diagnostics/openpose_swapped")

FILES = {
    "output_ref_post_trc": os.path.join(
        BASE, "tests/opencap-test-data/Data/sync_2-cameras/OutputReference/squats.trc"
    ),
    "current_post_trc": os.path.join(
        BASE,
        "tests/opencap-test-data/Data/sync_2-cameras/MarkerData/PostAugmentation/squats.trc",
    ),
    "current_neutral_post_trc": os.path.join(
        BASE,
        "tests/opencap-test-data/Data/sync_2-cameras/MarkerData/PostAugmentation/neutral.trc",
    ),
    "diag_swapped_pre": os.path.join(
        SWAPPED_DIR, "squats_openpose_swapped_pre.trc"
    ),
    "diag_swapped_post": os.path.join(
        SWAPPED_DIR, "squats_openpose_swapped_post.trc"
    ),
    "diag_swapped_neutral_post": os.path.join(
        SWAPPED_DIR, "neutral_openpose_swapped_post.trc"
    ),
}


def data_for(path):
    trc = utilsDataman.TRCFile(path)
    names = list(trc.marker_names)
    arr = np.stack([trc.marker(marker) for marker in names], axis=1)
    return names, arr


def summary(name, path):
    names, arr = data_for(path)
    flat = arr.reshape(-1, 3)
    flat = flat[np.isfinite(flat).all(axis=1)]
    mn, mx = flat.min(axis=0), flat.max(axis=0)
    center = np.nanmedian(arr.reshape(-1, 3), axis=0)

    print()
    print(name)
    print("path", path)
    print("markers", len(names), "frames", arr.shape[0])
    print("bounds min", np.round(mn, 4), "max", np.round(mx, 4))
    print("range", np.round(mx - mn, 4))
    print("median center", np.round(center, 4))

    for marker in [
        "midHip",
        "Neck",
        "RHip",
        "LHip",
        "RAnkle",
        "LAnkle",
        "r.ASIS_study",
        "L.ASIS_study",
        "C7_study",
    ]:
        if marker in names:
            point = np.nanmedian(arr[:, names.index(marker), :], axis=0)
            print(marker, np.round(point, 4))

    return names, arr


all_data = {
    key: summary(key, path) for key, path in FILES.items() if os.path.exists(path)
}

for a, b in [
    ("output_ref_post_trc", "diag_swapped_post"),
    ("current_post_trc", "diag_swapped_post"),
    ("output_ref_post_trc", "current_post_trc"),
    ("output_ref_post_trc", "current_neutral_post_trc"),
    ("diag_swapped_post", "diag_swapped_neutral_post"),
]:
    if a not in all_data or b not in all_data:
        continue

    names_a, arr_a = all_data[a]
    names_b, arr_b = all_data[b]
    common = [marker for marker in names_a if marker in names_b]
    frame_count = min(arr_a.shape[0], arr_b.shape[0])

    diffs = []
    for marker in common:
        data_a = arr_a[:frame_count, names_a.index(marker), :]
        data_b = arr_b[:frame_count, names_b.index(marker), :]
        diffs.append(np.nanmedian(data_b - data_a, axis=0))
    diffs = np.array(diffs)

    print()
    print("OFFSET", b, "minus", a, "common", len(common))
    print("median xyz", np.round(np.nanmedian(diffs, axis=0), 4))
    print("mean abs xyz", np.round(np.nanmean(np.abs(diffs), axis=0), 4))
