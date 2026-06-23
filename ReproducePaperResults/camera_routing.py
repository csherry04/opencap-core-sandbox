def select_camera_setup_for_trial(full_camera_setup, requested_camera_setup,
                                  is_extrinsics_trial=False,
                                  scale_model=False):
    if is_extrinsics_trial or scale_model:
        return full_camera_setup
    return requested_camera_setup


def select_cameras_for_trial(recorded_cameras, requested_cameras,
                             is_extrinsics_trial=False, scale_model=False):
    recorded_cameras = list(recorded_cameras)
    requested_cameras = list(requested_cameras)

    if is_extrinsics_trial or scale_model:
        return recorded_cameras

    if len(requested_cameras) < 2:
        raise ValueError("At least two cameras are required for dynamic trials.")

    missing_cameras = [
        camera for camera in requested_cameras if camera not in recorded_cameras
    ]
    if missing_cameras:
        raise ValueError(
            "Requested cameras must be a subset of recorded cameras: "
            + ", ".join(missing_cameras)
        )

    return requested_cameras
