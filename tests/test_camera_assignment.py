import os
import sys

import pytest

thisDir = os.path.dirname(os.path.realpath(__file__))
repoDir = os.path.abspath(os.path.join(thisDir, '../'))
sys.path.append(repoDir)

from camera_assignment import (
    detect_and_optionally_fix_full_camera_swap,
    swap_camera_keypoints,
)
from utils import loadCameraParameters
from utilsSync import synchronizeVideos


TRIAL_NAME = 'squats'


def load_camera_inputs(session_dir):
    videos_dir = os.path.join(session_dir, 'Videos')
    camera_directories = {
        'Cam0': os.path.join(videos_dir, 'Cam0'),
        'Cam1': os.path.join(videos_dir, 'Cam1'),
    }
    camera_param_dict = {
        cam_name: loadCameraParameters(
            os.path.join(cam_dir, 'cameraIntrinsicsExtrinsics.pickle')
        )
        for cam_name, cam_dir in camera_directories.items()
    }
    return camera_directories, camera_param_dict


@pytest.fixture
def synchronized_trial(session_dir):
    camera_directories, camera_param_dict = load_camera_inputs(session_dir)
    trial_relative_path = os.path.join(
        'InputMedia', TRIAL_NAME, f'{TRIAL_NAME}.mov'
    )
    keypoints2d, confidence, _, _, _, _, cameras2use = synchronizeVideos(
        camera_directories,
        trial_relative_path,
        '',
        undistortPoints=True,
        CamParamDict=camera_param_dict,
        filtFreqs={'gait': 12, 'default': 500},
        confidenceThreshold=0.4,
        imageBasedTracker=False,
        cams2Use=['all'],
        poseDetector='mmpose',
        trialName=TRIAL_NAME,
        resolutionPoseDetection='default',
        syncVer='1.1',
    )
    return camera_param_dict, keypoints2d, confidence, cameras2use


def test_detect_full_camera_swap_accepts_correct_assignment(synchronized_trial):
    camera_param_dict, keypoints2d, confidence, cameras2use = synchronized_trial
    result = detect_and_optionally_fix_full_camera_swap(
        camera_param_dict,
        keypoints2d,
        confidence,
        cams_to_use=cameras2use,
    )

    assert result['swap_is_better'] is False
    assert result['original_error'] <= result['swapped_error']


def test_detect_full_camera_swap_finds_and_fixes_swapped_assignment(
    synchronized_trial,
):
    camera_param_dict, keypoints2d, confidence, cameras2use = synchronized_trial
    swapped_keypoints, swapped_confidence = swap_camera_keypoints(
        keypoints2d,
        confidence,
        cameras2use[0],
        cameras2use[1],
    )

    result = detect_and_optionally_fix_full_camera_swap(
        camera_param_dict,
        swapped_keypoints,
        swapped_confidence,
        cams_to_use=cameras2use,
        auto_fix=True,
    )

    assert result['swap_is_better'] is True
    assert result['swapped_error'] < result['original_error']
    for cam_name in cameras2use:
        assert result['keypoint_dict'][cam_name].shape == keypoints2d[cam_name].shape
