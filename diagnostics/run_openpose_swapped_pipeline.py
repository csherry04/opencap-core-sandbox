"""Run OpenPose triangulation with Cam0/Cam1 keypoint swap, then augment/scale/IK.

The test session's OpenPose 2D keypoints are paired with the wrong cameras.
Swapping Cam0 and Cam1 keypoints before triangulation aligns pre-augmentation
TRCs with the HRNet reference frame (~1 m X offset disappears).
"""
import argparse
import copy
import glob
import logging
import os
import sys

import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE)

from utils import importMetadata, loadCameraParameters
from utilsAugmenter import augmentTRC
from utilsChecker import triangulateMultiviewVideo, writeTRCfrom3DKeypoints
from utilsOpenSim import generateVisualizerJson, getScaleTimeRange, runIKTool, runScaleTool
from utilsSync import synchronizeVideos

DEFAULT_DATA_DIR = os.path.join(BASE, "tests/opencap-test-data")
DEFAULT_OUT_DIR = os.path.join(BASE, "diagnostics/openpose_swapped")


def swap_camera_keypoints(keypoints2D, confidence):
    cams = list(keypoints2D.keys())
    if len(cams) != 2:
        raise ValueError(f"Expected 2 cameras for swap, got {cams}")
    c0, c1 = cams[0], cams[1]
    kp = copy.deepcopy(keypoints2D)
    conf = copy.deepcopy(confidence)
    kp[c0], kp[c1] = keypoints2D[c1].copy(), keypoints2D[c0].copy()
    conf[c0], conf[c1] = confidence[c1].copy(), confidence[c0].copy()
    logging.info("Swapped 2D keypoints/confidence for %s <-> %s", c0, c1)
    return kp, conf


def load_session(session_name, data_dir):
    session_dir = os.path.join(data_dir, "Data", session_name)
    metadata = importMetadata(os.path.join(session_dir, "sessionMetadata.yaml"))
    camera_directories = {}
    for path_cam in sorted(glob.glob(os.path.join(session_dir, "Videos", "Cam*"))):
        camera_directories[os.path.basename(path_cam)] = path_cam
    cam_param_dict = {
        cam: loadCameraParameters(os.path.join(cam_dir, "cameraIntrinsicsExtrinsics.pickle"))
        for cam, cam_dir in camera_directories.items()
    }
    return session_dir, metadata, camera_directories, cam_param_dict


def rotation_angles_from_metadata(metadata):
    checker_board_mount = metadata["checkerBoard"]["placement"]
    if checker_board_mount in ("backWall", "Perpendicular"):
        return {"y": 90, "z": 180}
    if checker_board_mount in ("ground", "Lying"):
        return {"x": 90, "y": 90}
    raise ValueError(f"Unsupported checkerBoard placement: {checker_board_mount}")


def process_trial(
    session_dir,
    metadata,
    camera_directories,
    cam_param_dict,
    trial_name,
    trial_id,
    out_dir,
    swap_cameras=True,
    run_scaling=False,
    run_ik=False,
    scaled_model_path=None,
    sync_ver="1.1",
):
    os.makedirs(out_dir, exist_ok=True)
    trial_relative_path = os.path.join("InputMedia", trial_name, trial_id + ".mov")
    rotation_angles = rotation_angles_from_metadata(metadata)
    filt_freqs = {"gait": 12, "default": 500}

    cam_dirs = copy.deepcopy(camera_directories)
    cam_params = copy.deepcopy(cam_param_dict)
    keypoints2D, confidence, keypoint_names, frame_rate, nans_in_out, start_end_frames, cameras2_use = (
        synchronizeVideos(
            cam_dirs,
            trial_relative_path,
            None,
            undistortPoints=True,
            CamParamDict=cam_params,
            filtFreqs=filt_freqs,
            confidenceThreshold=0.4,
            cams2Use=["all"],
            poseDetector="OpenPose",
            trialName=trial_name,
            resolutionPoseDetection="default",
            syncVer=sync_ver,
        )
    )

    if swap_cameras:
        keypoints2D, confidence = swap_camera_keypoints(keypoints2D, confidence)

    keypoints3D, _ = triangulateMultiviewVideo(
        copy.deepcopy(cam_param_dict),
        keypoints2D,
        cams2Use=cameras2_use,
        confidenceDict=confidence,
        spline3dZeros=True,
        splineMaxFrames=int(frame_rate / 5),
        nansInOut=nans_in_out,
        CameraDirectories=camera_directories,
        trialName=trial_name,
        startEndFrames=start_end_frames,
        trialID=trial_id,
    )

    suffix = trial_id if trial_id != trial_name else trial_name
    pre_trc = os.path.join(out_dir, f"{suffix}_openpose_swapped_pre.trc")
    post_trc = os.path.join(out_dir, f"{suffix}_openpose_swapped_post.trc")
    writeTRCfrom3DKeypoints(
        keypoints3D, pre_trc, keypoint_names, frameRate=frame_rate, rotationAngles=rotation_angles
    )

    augmenter_model_name = metadata["markerAugmentationSettings"]["markerAugmenterModel"]
    augmenter_dir = os.path.join(BASE, "MarkerAugmenter")
    vertical_offset = augmentTRC(
        pre_trc,
        metadata["mass_kg"],
        metadata["height_m"],
        post_trc,
        augmenter_dir,
        augmenterModelName=augmenter_model_name,
        augmenter_model=metadata.get("augmentermodel", "v0.3"),
        offset=True,
    )
    vertical_offset_settings = float(np.copy(vertical_offset) - 0.01)
    vertical_offset_vis = 0.01

    result = {
        "pre_trc": pre_trc,
        "post_trc": post_trc,
        "vertical_offset": vertical_offset_vis,
        "vertical_offset_settings": vertical_offset_settings,
    }

    open_sim_pipeline_dir = os.path.join(BASE, "opensimPipeline")
    suffix_model = "_shoulder" if "shoulder" in metadata["openSimModel"] else ""

    if run_scaling:
        model_dir = os.path.join(out_dir, "Model")
        os.makedirs(model_dir, exist_ok=True)
        setup_scaling = os.path.join(open_sim_pipeline_dir, "Scaling", "Setup_scaling_LaiUhlrich2022.xml")
        generic_model = os.path.join(open_sim_pipeline_dir, "Models", metadata["openSimModel"] + ".osim")

        threshold_position = 0.003
        max_threshold = 0.015
        increment = 0.001
        time_range_scaling = None
        while threshold_position <= max_threshold:
            try:
                time_range_scaling = getScaleTimeRange(
                    post_trc,
                    thresholdPosition=threshold_position,
                    thresholdTime=0.1,
                    removeRoot=True,
                )
                break
            except Exception as exc:
                logging.info(
                    "Scale time range attempt at threshold %.4f failed: %s",
                    threshold_position,
                    exc,
                )
                threshold_position += increment
        if time_range_scaling is None:
            raise RuntimeError("Could not determine scaling time range")

        scaled_model_path = runScaleTool(
            setup_scaling,
            generic_model,
            metadata["mass_kg"],
            post_trc,
            time_range_scaling,
            model_dir,
            subjectHeight=metadata["height_m"],
            suffix_model=suffix_model,
        )
        result["scaled_model"] = scaled_model_path
        result["scaled_motion"] = scaled_model_path[:-5] + ".mot"
        result["scale_time_range"] = time_range_scaling

    if run_ik:
        if scaled_model_path is None or not os.path.exists(scaled_model_path):
            raise ValueError("run_ik requires a valid scaled_model_path")
        ik_dir = os.path.join(out_dir, "Kinematics")
        os.makedirs(ik_dir, exist_ok=True)
        setup_ik = os.path.join(open_sim_pipeline_dir, "IK", f"Setup_IK{suffix_model}.xml")
        ik_motion, model_for_vis = runIKTool(
            setup_ik,
            scaled_model_path,
            post_trc,
            ik_dir,
            IKFileName=f"{suffix}_openpose_swapped",
        )
        vis_dir = os.path.join(out_dir, "VisualizerJsons")
        os.makedirs(vis_dir, exist_ok=True)
        vis_path = os.path.join(vis_dir, f"{suffix}_openpose_swapped.json")
        generateVisualizerJson(
            model_for_vis,
            ik_motion,
            vis_path,
            vertical_offset=vertical_offset_vis,
            roundToRotations=4,
            roundToTranslations=4,
        )
        result["ik_motion"] = ik_motion
        result["visualizer_json"] = vis_path

    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="sync_2-cameras")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--sync-ver", default="1.1")
    parser.add_argument("--no-swap", action="store_true")
    parser.add_argument("--scale-neutral", action="store_true")
    parser.add_argument("--ik-trial", default="squats")
    parser.add_argument("--ik-trial-id", default="squats")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    session_dir, metadata, camera_directories, cam_param_dict = load_session(
        args.session, args.data_dir
    )

    swap = not args.no_swap
    scaled_model = None

    if args.scale_neutral:
        logging.info("Processing neutral trial with swap=%s", swap)
        neutral_result = process_trial(
            session_dir,
            metadata,
            camera_directories,
            cam_param_dict,
            trial_name="neutral",
            trial_id="neutral",
            out_dir=args.out_dir,
            swap_cameras=swap,
            run_scaling=True,
            sync_ver=args.sync_ver,
        )
        scaled_model = neutral_result["scaled_model"]
        logging.info("Scaled model: %s", scaled_model)

    if args.ik_trial:
        logging.info("Processing %s IK with swap=%s", args.ik_trial, swap)
        ik_result = process_trial(
            session_dir,
            metadata,
            camera_directories,
            cam_param_dict,
            trial_name=args.ik_trial,
            trial_id=args.ik_trial_id,
            out_dir=args.out_dir,
            swap_cameras=swap,
            run_ik=True,
            scaled_model_path=scaled_model
            or os.path.join(
                args.out_dir,
                "Model",
                metadata["openSimModel"] + "_scaled.osim",
            ),
            sync_ver=args.sync_ver,
        )
        logging.info("IK motion: %s", ik_result["ik_motion"])


if __name__ == "__main__":
    main()
