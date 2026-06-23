import os
import pickle
import sys

import numpy as np
import pytest

thisDir = os.path.dirname(os.path.realpath(__file__))
repoDir = os.path.abspath(os.path.join(thisDir, '../'))
sys.path.append(repoDir)

from utils import importMetadata


SESSION_NAME = 'sync_2-cameras'
TRIAL_NAMES = ['neutral', 'squats-with-arm-raise', 'squats', 'walk']
DYNAMIC_TRIAL_NAMES = ['squats-with-arm-raise', 'squats', 'walk']
POSE_OUTPUT_FOLDER = 'OutputPkl_mmpose_0.8'


@pytest.fixture
def session_dir():
    return os.path.join(thisDir, 'opencap-test-data', 'Data', SESSION_NAME)


@pytest.fixture
def videos_dir(session_dir):
    return os.path.join(session_dir, 'Videos')


def camera_names(videos_dir):
    return sorted(
        name for name in os.listdir(videos_dir)
        if name.startswith('Cam')
        and os.path.isdir(os.path.join(videos_dir, name))
    )


def load_pickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def test_metadata_camera_entries_match_video_folders(session_dir, videos_dir):
    metadata = importMetadata(os.path.join(session_dir, 'sessionMetadata.yaml'))
    cameras_from_metadata = sorted(metadata['iphoneModel'].keys())

    assert cameras_from_metadata == camera_names(videos_dir)


def test_mapping_cam_device_matches_video_folders(videos_dir):
    mapping = load_pickle(os.path.join(videos_dir, 'mappingCamDevice.pickle'))
    expected_camera_indices = sorted(
        int(camera.replace('Cam', '')) for camera in camera_names(videos_dir)
    )

    assert sorted(mapping.values()) == expected_camera_indices
    assert len(set(mapping.values())) == len(mapping)


@pytest.mark.parametrize("trial_name", TRIAL_NAMES)
def test_each_camera_has_matching_input_video_for_trial(videos_dir, trial_name):
    for camera in camera_names(videos_dir):
        input_video = os.path.join(
            videos_dir, camera, 'InputMedia', trial_name, f'{trial_name}.mov'
        )

        assert os.path.exists(input_video), (
            f"Missing input video for {camera} trial {trial_name}: "
            f"{input_video}"
        )


@pytest.mark.parametrize("trial_name", DYNAMIC_TRIAL_NAMES)
def test_each_camera_has_matching_pose_pickle_for_dynamic_trial(
    videos_dir, trial_name
):
    for camera in camera_names(videos_dir):
        pose_pickle = os.path.join(
            videos_dir, camera, POSE_OUTPUT_FOLDER, trial_name,
            f'{trial_name}_rotated_pp.pkl'
        )

        assert os.path.exists(pose_pickle), (
            f"Missing pose pickle for {camera} trial {trial_name}: "
            f"{pose_pickle}"
        )


def test_camera_parameter_files_are_distinct_and_well_formed(videos_dir):
    camera_params = {}
    for camera in camera_names(videos_dir):
        camera_params[camera] = load_pickle(
            os.path.join(videos_dir, camera, 'cameraIntrinsicsExtrinsics.pickle')
        )

    for camera, params in camera_params.items():
        assert params['intrinsicMat'].shape == (3, 3), camera
        assert params['distortion'].shape == (1, 5), camera
        assert params['imageSize'].shape == (2, 1), camera
        assert params['rotation'].shape == (3, 3), camera
        assert params['translation'].shape == (3, 1), camera
        assert np.isfinite(params['intrinsicMat']).all(), camera
        assert np.isfinite(params['rotation']).all(), camera
        assert np.isfinite(params['translation']).all(), camera

    cameras = camera_names(videos_dir)
    for i, camera_a in enumerate(cameras):
        for camera_b in cameras[i + 1:]:
            assert not np.allclose(
                camera_params[camera_a]['translation'],
                camera_params[camera_b]['translation'],
            ), f"{camera_a} and {camera_b} have matching extrinsics"
