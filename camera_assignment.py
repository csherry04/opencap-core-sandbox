import copy

import numpy as np

from utilsChecker import calcReprojectionError, triangulateMultiview
from utilsCameraPy3 import Camera


def _ordered_cameras(camera_param_dict, cams_to_use):
    if cams_to_use is None or cams_to_use == ['all']:
        return list(camera_param_dict.keys())
    return list(cams_to_use)


def _camera_objects(camera_param_dict, ordered_cams):
    camera_list = []
    for cam_name in ordered_cams:
        cam_params = camera_param_dict[cam_name]
        camera = Camera()
        camera.set_K(cam_params['intrinsicMat'])
        camera.set_R(cam_params['rotation'])
        camera.set_t(np.reshape(cam_params['translation'], (3, 1)))
        camera_list.append(camera)
    return camera_list


def _sample_frame_indices(num_frames, max_frames=40):
    if num_frames <= max_frames:
        return list(range(num_frames))
    return list(np.linspace(0, num_frames - 1, max_frames, dtype=int))


def score_camera_assignment(
    camera_param_dict,
    keypoint_dict,
    confidence_dict,
    cams_to_use=None,
    max_frames=40,
):
    ordered_cams = _ordered_cameras(camera_param_dict, cams_to_use)
    keypoint_list = [keypoint_dict[cam] for cam in ordered_cams]
    confidence_list = [confidence_dict[cam] for cam in ordered_cams]
    camera_param_list = [camera_param_dict[cam] for cam in ordered_cams]
    camera_list = _camera_objects(camera_param_dict, ordered_cams)

    frame_indices = _sample_frame_indices(keypoint_list[0].shape[1], max_frames=max_frames)
    frame_errors = []
    for frame_index in frame_indices:
        points2d = [keypoints[:, frame_index:frame_index + 1, :] for keypoints in keypoint_list]
        confidence = [conf[:, frame_index] for conf in confidence_list]
        points3d, _ = triangulateMultiview(
            camera_param_list,
            points2d,
            confidence=confidence,
        )

        stacked_points = np.stack(points2d)
        points_input = [
            stacked_points[:, marker_index, 0, :].T
            for marker_index in range(stacked_points.shape[1])
        ]
        reprojection_error = calcReprojectionError(
            camera_list,
            points_input,
            points3d,
            weights=confidence,
        )
        frame_errors.append(float(np.nanmean(reprojection_error)))

    return float(np.nanmean(frame_errors))


def swap_camera_keypoints(keypoint_dict, confidence_dict, cam_a, cam_b):
    swapped_keypoints = copy.deepcopy(keypoint_dict)
    swapped_confidence = copy.deepcopy(confidence_dict)
    swapped_keypoints[cam_a], swapped_keypoints[cam_b] = (
        keypoint_dict[cam_b].copy(),
        keypoint_dict[cam_a].copy(),
    )
    swapped_confidence[cam_a], swapped_confidence[cam_b] = (
        confidence_dict[cam_b].copy(),
        confidence_dict[cam_a].copy(),
    )
    return swapped_keypoints, swapped_confidence


def detect_and_optionally_fix_full_camera_swap(
    camera_param_dict,
    keypoint_dict,
    confidence_dict,
    cams_to_use=None,
    max_frames=40,
    improvement_ratio_threshold=0.85,
    auto_fix=False,
):
    ordered_cams = _ordered_cameras(camera_param_dict, cams_to_use)
    if len(ordered_cams) != 2:
        raise ValueError(
            "Full camera swap detection currently supports exactly 2 cameras."
        )

    original_error = score_camera_assignment(
        camera_param_dict,
        keypoint_dict,
        confidence_dict,
        cams_to_use=ordered_cams,
        max_frames=max_frames,
    )

    swapped_keypoints, swapped_confidence = swap_camera_keypoints(
        keypoint_dict,
        confidence_dict,
        ordered_cams[0],
        ordered_cams[1],
    )
    swapped_error = score_camera_assignment(
        camera_param_dict,
        swapped_keypoints,
        swapped_confidence,
        cams_to_use=ordered_cams,
        max_frames=max_frames,
    )

    swap_is_better = swapped_error < original_error * improvement_ratio_threshold
    result = {
        'ordered_cams': ordered_cams,
        'original_error': original_error,
        'swapped_error': swapped_error,
        'swap_is_better': swap_is_better,
    }

    if auto_fix and swap_is_better:
        result['keypoint_dict'] = swapped_keypoints
        result['confidence_dict'] = swapped_confidence
    else:
        result['keypoint_dict'] = keypoint_dict
        result['confidence_dict'] = confidence_dict

    return result
