import os
import sys

import pytest

thisDir = os.path.dirname(os.path.realpath(__file__))
repoDir = os.path.abspath(os.path.join(thisDir, '../'))
sys.path.append(repoDir)

from ReproducePaperResults.camera_routing import (
    select_camera_setup_for_trial,
    select_cameras_for_trial,
)


RECORDED_CAMERAS = ['Cam0', 'Cam1', 'Cam2', 'Cam3', 'Cam4']


@pytest.mark.parametrize(
    "is_extrinsics_trial,scale_model",
    [
        (True, False),
        (False, True),
    ],
)
def test_all_recorded_cameras_used_for_calibration_and_static(
    is_extrinsics_trial, scale_model
):
    selected_cameras = select_cameras_for_trial(
        RECORDED_CAMERAS,
        ['Cam1', 'Cam3'],
        is_extrinsics_trial=is_extrinsics_trial,
        scale_model=scale_model,
    )

    assert selected_cameras == RECORDED_CAMERAS


@pytest.mark.parametrize(
    "requested_cameras",
    [
        ['Cam1', 'Cam3'],
        ['Cam0', 'Cam2', 'Cam4'],
        ['Cam0', 'Cam1', 'Cam2', 'Cam3', 'Cam4'],
    ],
)
def test_dynamic_trials_use_requested_camera_subset(requested_cameras):
    selected_cameras = select_cameras_for_trial(
        RECORDED_CAMERAS,
        requested_cameras,
        is_extrinsics_trial=False,
        scale_model=False,
    )

    assert selected_cameras == requested_cameras


@pytest.mark.parametrize(
    "is_extrinsics_trial,scale_model,expected_setup",
    [
        (True, False, 'all-cameras'),
        (False, True, 'all-cameras'),
        (False, False, 'selected-cameras'),
    ],
)
def test_camera_setup_name_matches_trial_type(
    is_extrinsics_trial, scale_model, expected_setup
):
    selected_setup = select_camera_setup_for_trial(
        'all-cameras',
        'selected-cameras',
        is_extrinsics_trial=is_extrinsics_trial,
        scale_model=scale_model,
    )

    assert selected_setup == expected_setup


def test_dynamic_trials_require_at_least_two_cameras():
    with pytest.raises(ValueError, match="At least two cameras"):
        select_cameras_for_trial(
            RECORDED_CAMERAS,
            ['Cam1'],
            is_extrinsics_trial=False,
            scale_model=False,
        )


def test_dynamic_trials_require_requested_cameras_to_be_recorded():
    with pytest.raises(ValueError, match="subset of recorded cameras"):
        select_cameras_for_trial(
            RECORDED_CAMERAS,
            ['Cam1', 'Cam9'],
            is_extrinsics_trial=False,
            scale_model=False,
        )
