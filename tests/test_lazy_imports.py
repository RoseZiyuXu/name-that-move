import os
import subprocess
import sys
from pathlib import Path


def test_package_import_does_not_eagerly_load_model_libraries():
    source_dir = Path(__file__).parents[1] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_dir)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import name_that_move; "
                "assert 'torch' not in sys.modules; "
                "assert 'tsai' not in sys.modules"
            ),
        ],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )
    assert result.returncode == 0, result.stderr
