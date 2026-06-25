import logging
import os
import sys
import numpy as np
import pandas as pd
import pytest

thisDir = os.path.dirname(os.path.realpath(__file__))
repoDir = os.path.abspath(os.path.join(thisDir, '../'))
sys.path.append(repoDir)
from main import main

# Helper functions to load and compare TRC and MOT files
def load_trc(file, num_metadata_lines=5):
    with open(file, 'r') as f:
        lines = f.readlines()
        metadata = lines[:num_metadata_lines]
    df = pd.read_csv(file, sep='\t', skiprows=num_metadata_lines + 1, header=None)
    return df, metadata

def load_mot(file, num_metadata_lines=10):
    with open(file, 'r') as f:
        lines = f.readlines()
        metadata = lines[:num_metadata_lines]
    df = pd.read_csv(file, sep='\t', skiprows=num_metadata_lines)
    return df, metadata


def load_binary(file):
    with open(file, 'rb') as f:
        return f.read()


def load_trc_marker_names(file):
    with open(file, 'r') as f:
        marker_line = f.readlines()[3]
    return [marker for marker in marker_line.strip().split('\t')[2:] if marker]


def calc_rmse(series1, series2):
    return np.sqrt(((series1 - series2) ** 2).mean())

def compare_mot(output_mot_df, ref_mot_df, t0, tf):
    '''Function to compare MOT dataframes within a time range [t0, tf].
    We use the specific time range to analyze the range with the motion
    of interest. In particular, the arm raise can create larger differences
    on single frames.

    - Time column is checked for equality (IK is frame-by-frame).
    - Translation error is checked within 2 mm max per frame, RMSE within 
      1 mm.
    - Rotation error for wrist pronation/supination (coordinates pro_sup_r
      and pro_sup_l) are checked within 5.0 degrees max per frame, RMSE
      within 1.0 degrees.
    - Rotation error for all other coordinates are tighter and checked 
      within 2.5 degrees max per frame, RMSE within 0.5 degrees.
    '''
    output_mot_df_slice = output_mot_df[(output_mot_df['time'] >= t0) & (output_mot_df['time'] <= tf)]
    ref_mot_df_slice = ref_mot_df[(ref_mot_df['time'] >= t0) & (ref_mot_df['time'] <= tf)]
    for col in ref_mot_df.columns:
        # time column should be equal since IK is frame-by-frame
        if col == 'time':
            pd.testing.assert_series_equal(output_mot_df[col], ref_mot_df[col])

        # check translational within 2 mm max error, rmse within 1 mm
        elif any(substr in col for substr in ['tx', 'ty', 'tz']):
            pd.testing.assert_series_equal(
                output_mot_df_slice[col], ref_mot_df_slice[col], atol=0.002
            )
            rmse = calc_rmse(output_mot_df_slice[col], ref_mot_df_slice[col])
            assert rmse <= 0.001

        elif 'pro_sup' in col:
            pd.testing.assert_series_equal(
                output_mot_df_slice[col], ref_mot_df_slice[col], atol=5.0
            )
            rmse = calc_rmse(output_mot_df_slice[col], ref_mot_df_slice[col])
            assert rmse <= 1.0

        # check rotational within 2.5 degrees max error, rmse within 0.5 degrees
        else:
            pd.testing.assert_series_equal(
                output_mot_df_slice[col], ref_mot_df_slice[col], atol=2.5
            )
            rmse = calc_rmse(output_mot_df_slice[col], ref_mot_df_slice[col])
            assert rmse <= 0.5

# End to end tests with different sync methods (hand, gait, general).
# Also check that syncVer updates with main changes.
# Note: no pose detection, uses pre-scaled opensim model
@pytest.mark.parametrize("syncVer", ['1.0', '1.1'])
@pytest.mark.parametrize("trialName, t0, tf", [
    ('squats-with-arm-raise', 5.0, 10.0),
    ('squats', 3.0, 8.0),
    ('walk', 1.0, 5.0),
])
def test_main(trialName, t0, tf, syncVer, caplog):
    caplog.set_level(logging.INFO)

    sessionName = 'sync_2-cameras'
    trialID = trialName
    dataDir = os.path.join(thisDir, 'opencap-test-data')
    main(
        sessionName,
        trialName,
        trialID,
        dataDir=dataDir,
        genericFolderNames=True,
        poseDetector='hrnet',
        syncVer=syncVer,
    )
    assert f"Synchronizing Keypoints using version {syncVer}" in caplog.text

    # Compare marker data
    output_trc = os.path.join(dataDir,
        'Data',
        sessionName,
        'MarkerData',
        'PostAugmentation',
        f'{trialName}.trc',
    )
    ref_trc = os.path.join(
        dataDir,
        'Data',
        sessionName,
        'OutputReference',
        f'{trialName}.trc',
    )
    output_trc_df, _ = load_trc(output_trc)
    ref_trc_df, _ = load_trc(ref_trc)
    pd.testing.assert_frame_equal(
        output_trc_df, ref_trc_df, check_exact=False, atol=1e-3
    )

    # Compare IK data
    output_mot = os.path.join(
        dataDir,
        'Data',
        sessionName,
        'OpenSimData',
        'Kinematics',
        f'{trialName}.mot',
    )
    ref_mot = os.path.join(
        dataDir,
        'Data',
        sessionName,
        'OutputReference',
        f'{trialName}.mot',
    )
    output_mot_df, _ = load_mot(output_mot)
    ref_mot_df, _ = load_mot(ref_mot)
    pd.testing.assert_index_equal(output_mot_df.columns, ref_mot_df.columns)
    compare_mot(output_mot_df, ref_mot_df, t0, tf)

def test_main_calibration_and_neutral(caplog):
    caplog.set_level(logging.INFO)

    sessionName = 'sync_2-cameras'
    dataDir = os.path.join(thisDir, 'opencap-test-data')
    sessionDir = os.path.join(dataDir, 'Data', sessionName)
    camera_param_paths = [
        os.path.join(
            sessionDir, 'Videos', camName, 'cameraIntrinsicsExtrinsics.pickle'
        )
        for camName in ('Cam0', 'Cam1')
    ]
    neutral_image_paths = [
        os.path.join(sessionDir, 'NeutralPoseImages', f'{camName}_0.png')
        for camName in ('Cam0', 'Cam1')
    ]
    neutral_pre_trc = os.path.join(
        sessionDir,
        'MarkerData',
        'OpenPose_default',
        'PreAugmentation',
        'neutral.trc',
    )
    neutral_post_trc = os.path.join(
        sessionDir,
        'MarkerData',
        'OpenPose_default',
        'PostAugmentation_v0.3',
        'neutral_LSTM.trc',
    )
    scaled_model_mot = os.path.join(
        sessionDir,
        'OpenSimData',
        'OpenPose_default',
        'Model',
        'LaiUhlrich2022_scaled.mot',
    )
    neutral_visualizer_json = os.path.join(
        sessionDir,
        'VisualizerJsons',
        'neutral',
        'neutral.json',
    )

    ref_camera_params = {path: load_binary(path) for path in camera_param_paths}
    ref_neutral_images = {path: load_binary(path) for path in neutral_image_paths}
    ref_pre_trc_df, _ = load_trc(neutral_pre_trc)
    ref_post_trc_df, _ = load_trc(neutral_post_trc)
    ref_scaled_model_mot_df, _ = load_mot(scaled_model_mot, num_metadata_lines=6)
    with open(neutral_visualizer_json, 'r') as f:
        ref_neutral_visualizer_json = f.read()

    main(
        sessionName,
        'calibration',
        'calibration',
        dataDir=dataDir,
        extrinsicsTrial=True,
    )
    assert 'Load extrinsics for Cam0 - already existing' in caplog.text
    assert 'Load extrinsics for Cam1 - already existing' in caplog.text
    for path, ref_params in ref_camera_params.items():
        assert load_binary(path) == ref_params

    main(
        sessionName,
        'neutral',
        'neutral',
        dataDir=dataDir,
        scaleModel=True,
    )

    for path, ref_image in ref_neutral_images.items():
        assert load_binary(path) == ref_image

    output_pre_trc_df, _ = load_trc(neutral_pre_trc)
    pd.testing.assert_frame_equal(
        output_pre_trc_df, ref_pre_trc_df, check_exact=False, atol=1e-3
    )

    output_post_trc_df, _ = load_trc(neutral_post_trc)
    pd.testing.assert_frame_equal(
        output_post_trc_df, ref_post_trc_df, check_exact=False, atol=1e-3
    )

    output_scaled_model_mot_df, _ = load_mot(
        scaled_model_mot, num_metadata_lines=6
    )
    pd.testing.assert_index_equal(
        output_scaled_model_mot_df.columns, ref_scaled_model_mot_df.columns
    )
    pd.testing.assert_frame_equal(
        output_scaled_model_mot_df,
        ref_scaled_model_mot_df,
        check_exact=False,
        atol=1e-6,
    )

    with open(neutral_visualizer_json, 'r') as f:
        assert f.read() == ref_neutral_visualizer_json


def test_lab_validation_static_downloaded_reference():
    labValidationDir = os.path.join(repoDir, 'Data', 'LabValidation')
    downloaded_static_trc = os.path.join(
        labValidationDir,
        'subject2',
        'MarkerData',
        'Video',
        'OpenPose_default',
        '5-cameras',
        'static1_videoAndMocap.trc',
    )
    reproduced_static_trc = os.path.join(
        labValidationDir,
        'Data',
        'subject2_Session0',
        'MarkerData',
        'OpenPose_default',
        '5-cameras',
        'PostAugmentation_v0.2',
        'static1_LSTM.trc',
    )
    downloaded_scale_mot = os.path.join(
        labValidationDir,
        'subject2',
        'OpenSimData',
        'Video',
        'OpenPose_default',
        '5-cameras',
        'Model',
        'LaiArnoldModified2017_poly_withArms_weldHand_scaled.mot',
    )
    reproduced_scale_mot = os.path.join(
        labValidationDir,
        'Data',
        'subject2_Session0',
        'OpenSimData',
        'OpenPose_default',
        '5-cameras',
        'Model',
        'LaiUhlrich2022_scaled.mot',
    )
    required_files = [
        downloaded_static_trc,
        reproduced_static_trc,
        downloaded_scale_mot,
        reproduced_scale_mot,
    ]
    if not all(os.path.exists(path) for path in required_files):
        pytest.skip('LabValidation downloaded dataset and reproduced outputs are unavailable.')

    downloaded_marker_names = set(load_trc_marker_names(downloaded_static_trc))
    reproduced_marker_names = set(load_trc_marker_names(reproduced_static_trc))
    assert reproduced_marker_names.issubset(downloaded_marker_names)

    reproduced_trc_df, _ = load_trc(reproduced_static_trc)
    assert len(reproduced_trc_df) > 10

    downloaded_scale_mot_df, _ = load_mot(
        downloaded_scale_mot, num_metadata_lines=6
    )
    reproduced_scale_mot_df, _ = load_mot(
        reproduced_scale_mot, num_metadata_lines=6
    )
    pd.testing.assert_index_equal(
        reproduced_scale_mot_df.columns, downloaded_scale_mot_df.columns
    )

    joint_value_cols = [
        col for col in downloaded_scale_mot_df.columns
        if (
            col.endswith('/value')
            and 'ground_pelvis/pelvis_tx' not in col
            and 'ground_pelvis/pelvis_ty' not in col
            and 'ground_pelvis/pelvis_tz' not in col
        )
    ]
    pd.testing.assert_frame_equal(
        reproduced_scale_mot_df[joint_value_cols],
        downloaded_scale_mot_df[joint_value_cols],
        check_exact=False,
        atol=0.25,
    )
# TODO: > 2 cameras
# TODO: augmenter versions
