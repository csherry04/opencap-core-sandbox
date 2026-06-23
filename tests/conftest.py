import os
import shutil

import pytest


THIS_DIR = os.path.dirname(os.path.realpath(__file__))
SOURCE_TEST_DATA_DIR = os.path.join(THIS_DIR, 'opencap-test-data')
SESSION_NAME = 'sync_2-cameras'


@pytest.fixture
def test_data_dir(tmp_path):
    copied_data_dir = tmp_path / 'opencap-test-data'
    shutil.copytree(
        SOURCE_TEST_DATA_DIR,
        copied_data_dir,
        ignore=shutil.ignore_patterns('.git'),
    )
    return str(copied_data_dir)


@pytest.fixture
def session_dir(test_data_dir):
    return os.path.join(test_data_dir, 'Data', SESSION_NAME)
