"""Static checks on the deploy files: the unit reads the env file, the backup script parses."""

import re
import subprocess
from pathlib import Path

DEPLOY = Path(__file__).resolve().parents[1] / "deploy"


def test_unit_reads_the_env_file_and_scripts_parse():
    assert "EnvironmentFile=/etc/clawdia/env" in (DEPLOY / "clawdia.service").read_text()
    for script in ("backup.sh",):
        subprocess.run(["bash", "-n", str(DEPLOY / script)], check=True)
