import os
import pickle
import shutil

import cv2
import numpy as np
import pytest

thisDir = os.path.dirname(os.path.realpath(__file__))
repoDir = os.path.abspath(os.path.join(thisDir, '../'))

import sys
sys.path.append(repoDir)

from utilsChecker import calcExtrinsics, calcExtrinsicsFromVideo


CHECKERBOARD_PARAMS = {
    'dimensions': (5, 4),
    'squareSize': 35.0,
}


UTAH_CALIBRATION_CASES = [
    (
        '39af77bd-6df6-458e-8bf6-cd7777aea12e-Cam0',
        os.path.join(
            repoDir,
            'utah_calibration_tests',
            '39af77bd-6df6-458e-8bf6-cd7777aea12e',
            'Videos',
            'Cam0',
            'InputMedia',
            'calibration',
            '4999f23d-8491-4d4e-89d8-0daad86ec5f1.mov',
        ),
        os.path.join(
            repoDir,
            'utah_calibration_tests',
            '39af77bd-6df6-458e-8bf6-cd7777aea12e',
            'Videos',
            'Cam0',
            'InputMedia',
            'calibration',
            'cameraIntrinsicsExtrinsics_soln0.pickle',
        ),
    ),
    (
        '39af77bd-6df6-458e-8bf6-cd7777aea12e-Cam1',
        os.path.join(
            repoDir,
            'utah_calibration_tests',
            '39af77bd-6df6-458e-8bf6-cd7777aea12e',
            'Videos',
            'Cam1',
            'InputMedia',
            'calibration',
            '4999f23d-8491-4d4e-89d8-0daad86ec5f1.mov',
        ),
        os.path.join(
            repoDir,
            'utah_calibration_tests',
            '39af77bd-6df6-458e-8bf6-cd7777aea12e',
            'Videos',
            'Cam1',
            'InputMedia',
            'calibration',
            'cameraIntrinsicsExtrinsics_soln1.pickle',
        ),
    ),
    (
        '3fb608eb-a099-4c1e-a52e-ad71dbf35890-Cam0',
        os.path.join(
            repoDir,
            'utah_calibration_tests',
            '3fb608eb-a099-4c1e-a52e-ad71dbf35890',
            'Videos',
            'Cam0',
            'InputMedia',
            'calibration',
            '71a27a9c-ded6-46a8-afc0-5c65c50d611c.mov',
        ),
        os.path.join(
            repoDir,
            'utah_calibration_tests',
            '3fb608eb-a099-4c1e-a52e-ad71dbf35890',
            'Videos',
            'Cam0',
            'InputMedia',
            'calibration',
            'cameraIntrinsicsExtrinsics_soln0.pickle',
        ),
    ),
    (
        '3fb608eb-a099-4c1e-a52e-ad71dbf35890-Cam1',
        os.path.join(
            repoDir,
            'utah_calibration_tests',
            '3fb608eb-a099-4c1e-a52e-ad71dbf35890',
            'Videos',
            'Cam1',
            'InputMedia',
            'calibration',
            '71a27a9c-ded6-46a8-afc0-5c65c50d611c.mov',
        ),
        os.path.join(
            repoDir,
            'utah_calibration_tests',
            '3fb608eb-a099-4c1e-a52e-ad71dbf35890',
            'Videos',
            'Cam1',
            'InputMedia',
            'calibration',
            'cameraIntrinsicsExtrinsics_soln1.pickle',
        ),
    ),
]


def camera_params_for_image(width=640, height=480):
    focal_length = max(width, height)
    return {
        'intrinsicMat': np.array([
            [focal_length, 0.0, width / 2],
            [0.0, focal_length, height / 2],
            [0.0, 0.0, 1.0],
        ]),
        'distortion': np.zeros((5, 1)),
        'imageSize': np.array([height, width]),
    }


def write_blank_image(path, width=640, height=480):
    image = np.full((height, width, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(path, image)


def calibration_media_path(tmp_path, filename):
    media_dir = tmp_path / 'Data' / 'test_session' / 'Videos' / 'Cam0' / 'InputMedia' / 'calibration'
    media_dir.mkdir(parents=True, exist_ok=True)
    return media_dir / filename


def load_camera_params(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def copy_utah_video_to_tmp(tmp_path, video_path):
    local_video_path = calibration_media_path(tmp_path, os.path.basename(video_path))
    shutil.copy2(video_path, local_video_path)
    return str(local_video_path)


def require_utah_case(video_path, camera_params_path):
    if not os.path.exists(video_path):
        pytest.skip(f'Missing Utah calibration video: {video_path}')
    if not os.path.exists(camera_params_path):
        pytest.skip(f'Missing Utah camera parameters: {camera_params_path}')


def assert_valid_extrinsics(camera_params):
    assert camera_params is not None
    for key in ('rotation', 'translation', 'rotation_EulerAngles'):
        assert key in camera_params
        assert np.all(np.isfinite(camera_params[key]))
    assert camera_params['rotation'].shape == (3, 3)
    assert camera_params['translation'].shape in ((3, 1), (3,))
    assert 0 < np.linalg.norm(camera_params['translation']) < 10000


def test_calc_extrinsics_uses_sb_fallback_when_primary_detector_fails(
    tmp_path, monkeypatch
):
    image_path = str(calibration_media_path(tmp_path, 'checkerboard.png'))
    write_blank_image(image_path)
    n_corners = CHECKERBOARD_PARAMS['dimensions'][0] * CHECKERBOARD_PARAMS['dimensions'][1]
    corners = np.zeros((n_corners, 1, 2), dtype=np.float32)
    corners[:, 0, 0] = np.tile(np.arange(CHECKERBOARD_PARAMS['dimensions'][0]), 4) * 40 + 120
    corners[:, 0, 1] = np.repeat(np.arange(CHECKERBOARD_PARAMS['dimensions'][1]), 5) * 40 + 90
    calls = {'primary': 0, 'fallback': 0}

    def primary_detector(*args, **kwargs):
        calls['primary'] += 1
        return False, None

    def fallback_detector(*args, **kwargs):
        calls['fallback'] += 1
        return True, corners, np.zeros((n_corners, 1), dtype=np.uint8)

    def corner_subpix(*args, **kwargs):
        raise AssertionError('cornerSubPix should not run for SB fallback corners')

    def solve_pnp_generic(*args, **kwargs):
        return (
            2,
            [np.array([[0.1], [0.2], [0.3]]), np.array([[0.2], [0.1], [0.4]])],
            [np.array([[10.0], [20.0], [1000.0]]), np.array([[15.0], [25.0], [900.0]])],
            np.array([[0.1], [0.2]]),
        )

    monkeypatch.setattr(cv2, 'findChessboardCorners', primary_detector)
    monkeypatch.setattr(cv2, 'findChessboardCornersSBWithMeta', fallback_detector)
    monkeypatch.setattr(cv2, 'cornerSubPix', corner_subpix)
    monkeypatch.setattr(cv2, 'solvePnPGeneric', solve_pnp_generic)
    monkeypatch.setattr(cv2, 'drawFrameAxes', lambda image, *args, **kwargs: image)
    monkeypatch.setattr(cv2, 'imwrite', lambda *args, **kwargs: True)

    result = calcExtrinsics(
        image_path,
        camera_params_for_image(),
        CHECKERBOARD_PARAMS,
    )

    assert calls == {'primary': 1, 'fallback': 1}
    assert_valid_extrinsics(result)


def test_calc_extrinsics_returns_none_without_checkerboard(tmp_path):
    image_path = str(tmp_path / 'blank.png')
    write_blank_image(image_path)

    result = calcExtrinsics(
        image_path,
        camera_params_for_image(),
        CHECKERBOARD_PARAMS,
    )

    assert result is None


@pytest.mark.parametrize(
    'case_name, video_path, camera_params_path',
    UTAH_CALIBRATION_CASES,
    ids=[case[0] for case in UTAH_CALIBRATION_CASES],
)
def test_utah_calibration_videos_calibrate_with_primary_detector(
    case_name, video_path, camera_params_path, tmp_path, monkeypatch
):
    require_utah_case(video_path, camera_params_path)
    local_video_path = copy_utah_video_to_tmp(tmp_path, video_path)
    camera_params = load_camera_params(camera_params_path)
    detector_calls = {'primary': 0, 'fallback': 0}
    original_primary = cv2.findChessboardCorners
    original_fallback = cv2.findChessboardCornersSBWithMeta

    def primary_detector(*args, **kwargs):
        detector_calls['primary'] += 1
        return original_primary(*args, **kwargs)

    def fallback_detector(*args, **kwargs):
        detector_calls['fallback'] += 1
        return original_fallback(*args, **kwargs)

    monkeypatch.setattr(cv2, 'findChessboardCorners', primary_detector)
    monkeypatch.setattr(cv2, 'findChessboardCornersSBWithMeta', fallback_detector)

    result = calcExtrinsicsFromVideo(
        local_video_path,
        camera_params,
        CHECKERBOARD_PARAMS,
        visualize=False,
        imageUpsampleFactor=2,
    )

    assert detector_calls['primary'] > 0
    assert detector_calls['fallback'] == 0
    assert_valid_extrinsics(result)


@pytest.mark.parametrize(
    'case_name, video_path, camera_params_path',
    UTAH_CALIBRATION_CASES,
    ids=[case[0] for case in UTAH_CALIBRATION_CASES],
)
def test_utah_calibration_videos_calibrate_with_sb_fallback(
    case_name, video_path, camera_params_path, tmp_path, monkeypatch
):
    require_utah_case(video_path, camera_params_path)
    local_video_path = copy_utah_video_to_tmp(tmp_path, video_path)
    camera_params = load_camera_params(camera_params_path)
    detector_calls = {'primary': 0, 'fallback': 0}
    original_fallback = cv2.findChessboardCornersSBWithMeta

    def primary_detector(*args, **kwargs):
        detector_calls['primary'] += 1
        return False, None

    def fallback_detector(*args, **kwargs):
        detector_calls['fallback'] += 1
        return original_fallback(*args, **kwargs)

    monkeypatch.setattr(cv2, 'findChessboardCorners', primary_detector)
    monkeypatch.setattr(cv2, 'findChessboardCornersSBWithMeta', fallback_detector)

    result = calcExtrinsicsFromVideo(
        local_video_path,
        camera_params,
        CHECKERBOARD_PARAMS,
        visualize=False,
        imageUpsampleFactor=2,
    )

    assert detector_calls['primary'] > 0
    assert detector_calls['fallback'] > 0
    assert_valid_extrinsics(result)
