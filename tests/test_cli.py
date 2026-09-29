"""Basic smoke tests for the phailogeny CLI."""

from __future__ import annotations

import subprocess
import sys


def test_cli_help() -> None:
    """The CLI should expose a help page without errors."""
    result = subprocess.run(
        [sys.executable, "-m", "phailogeny.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
