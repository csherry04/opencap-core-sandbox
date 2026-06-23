import os
import sys

thisDir = os.path.dirname(os.path.realpath(__file__))
repoDir = os.path.abspath(os.path.join(thisDir, '../'))
sys.path.append(repoDir)

from utils import (
    build_iphone_model_mapping,
    get_mapping_path,
    reconcile_camera_mapping,
)


def test_build_iphone_model_mapping_uses_device_mapping_not_trial_order():
    trial_videos = [
        {
            'device_id': 'device-b',
            'parameters': {'model': 'iPhoneB'},
        },
        {
            'device_id': 'device-a',
            'parameters': {'model': 'iPhoneA'},
        },
    ]
    mapping = {
        'DEVICEA': 0,
        'DEVICEB': 1,
    }

    iphone_model = build_iphone_model_mapping(trial_videos, mapping)

    assert iphone_model == {
        'Cam0': 'iPhoneA',
        'Cam1': 'iPhoneB',
    }


def test_reconcile_camera_mapping_clears_mapping_dependent_cache_on_drift(tmp_path):
    session_path = tmp_path / 'session'
    videos_dir = session_path / 'Videos' / 'Cam0' / 'InputMedia' / 'trial'
    videos_dir.mkdir(parents=True)
    (videos_dir / 'trial.mov').write_text('video')
    (session_path / 'MarkerData').mkdir()
    (session_path / 'sessionMetadata.yaml').write_text('meta')

    reconcile_camera_mapping(str(session_path), {'DEVICEA': 0})
    assert os.path.exists(get_mapping_path(str(session_path)))

    reconcile_camera_mapping(str(session_path), {'DEVICEA': 1})

    assert not (session_path / 'MarkerData').exists()
    assert not (session_path / 'sessionMetadata.yaml').exists()
    assert not (session_path / 'Videos' / 'Cam0').exists()
    assert os.path.exists(get_mapping_path(str(session_path)))
