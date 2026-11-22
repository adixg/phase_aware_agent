"""Prepare an isolated workspace for reproducing the saved Oracle experiments."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
REPLAY_COMMIT = "a3dbeae41b7dac6c06f3fe1863ad1300e348f405"
REPLAY_FILES = (
    "analysis/__init__.py",
    "analysis/replay/__init__.py",
    "analysis/replay/extract.py",
    "analysis/replay/run_gate.py",
    "analysis/replay/sim.py",
    "tests/test_replay.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path, help="new directory for scripts and outputs")
    parser.add_argument("--data", type=Path, default=REPO / "data/freeinference")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    data = args.data.resolve()
    if workspace.is_relative_to(data):
        parser.error("choose a workspace outside the FreeInference input tree")

    for name in ("numpy", "orjson", "zstandard"):
        __import__(name)
    inputs = ("oracle-sensitivity/compact", "milestone-preprocessed/validation", "release")
    for path in inputs:
        if not (data / path).is_dir():
            parser.error(f"missing input directory: {data / path}")

    try:
        snapshot = {
            path: subprocess.check_output(["git", "show", f"{REPLAY_COMMIT}:{path}"], cwd=REPO)
            for path in REPLAY_FILES
        }
    except subprocess.CalledProcessError:
        parser.exit(1, f"Replay commit {REPLAY_COMMIT} is required. See README.md.\n")

    workspace.mkdir()
    source = workspace / "source"
    for path, contents in snapshot.items():
        target = source / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)

    ablation_source = workspace / "ablation-source"
    shutil.copytree(source, ablation_source)
    subprocess.run(
        ["patch", "-p1", "-i", str(PACKAGE / "demotion-hooks.patch")],
        cwd=ablation_source, check=True,
    )

    for name in ("oracle-sensitivity", "oracle-ablation", "dag-replay"):
        shutil.copytree(PACKAGE / name, workspace / name, ignore=shutil.ignore_patterns("__pycache__"))
    sensitivity = workspace / "oracle-sensitivity"
    ablation = workspace / "oracle-ablation"
    followup = ablation / "policy-followup"
    replay = workspace / "dag-replay"
    for directory in (sensitivity, replay, replay / "all-turns"):
        (directory / "source").symlink_to(source, target_is_directory=True)
    for directory in (ablation, followup):
        (directory / "source").symlink_to(ablation_source, target_is_directory=True)
    shutil.copyfile(ablation / "sensitivity_helpers.py", followup / "sensitivity_helpers.py")
    for directory in (sensitivity, ablation, followup):
        (directory / "compact").symlink_to(data / inputs[0], target_is_directory=True)
    for name in ("milestone-preprocessed", "release"):
        (workspace / name).symlink_to(data / name, target_is_directory=True)

    print(f"Workspace ready: {workspace}", file=sys.stderr)


if __name__ == "__main__":
    main()
