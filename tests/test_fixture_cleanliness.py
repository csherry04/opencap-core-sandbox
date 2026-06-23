import os
import subprocess


def test_opencap_test_data_fixture_is_clean():
    this_dir = os.path.dirname(os.path.realpath(__file__))
    fixture_dir = os.path.join(this_dir, 'opencap-test-data')

    result = subprocess.run(
        ['git', '-C', fixture_dir, 'status', '--short'],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == '', (
        "tests/opencap-test-data has local generated or modified files. "
        "The E2E tests compare against golden references, so dirty fixture "
        "state can hide or create downstream failures.\n\n"
        f"{result.stdout}"
    )
