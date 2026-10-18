#!/usr/bin/env python3
"""Offline regression for plate boundary-candidate grid recovery."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


EXPECTED_IDS = ("00741", "00742", "00743", "00744", "00745")
BOUNDARY_RECT = [94, 91, 1197, 91, 1197, 667, 94, 667]


def detect_repeated(
        runner: Path, library: Path, color_path: Path, depth_path: Path,
        det_type: str, rect, repeat: int):
    command = [
        str(runner),
        str(color_path),
        str(depth_path),
        str(library),
        det_type,
        json.dumps(rect, separators=(",", ":")),
        str(max(1, repeat)),
    ]
    completed = subprocess.run(
        command, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"offline runner returned {completed.returncode}\n"
            f"{completed.stdout}")
    results = [
        json.loads(line.removeprefix("RESULT_JSON="))
        for line in completed.stdout.splitlines()
        if line.startswith("RESULT_JSON=")
    ]
    if len(results) != max(1, repeat):
        raise RuntimeError(
            f"offline runner returned {len(results)} JSON results, "
            f"expected {max(1, repeat)}")
    return results


def validate(result):
    expected = {
        "empty_count": 8,
        "wheel_count": 0,
        "unknown_count": 0,
        "tray_complete": True,
    }
    failures = [
        f"{key}={result.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if result.get(key) != value
    ]
    slots = result.get("slots", [])
    if len(slots) != 8 or any(
        slot.get("index") != index or slot.get("state") != "empty"
        for index, slot in enumerate(slots)
    ):
        failures.append("slots are not ordered empty positions 0..7")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument(
        "--library", type=Path, default=Path("build-debug/libai-det.so"))
    parser.add_argument(
        "--runner", type=Path,
        default=Path("build-debug/ai-det-offline-runner"))
    parser.add_argument("--det-type", default="plate_empty")
    parser.add_argument(
        "--rect", default=json.dumps(BOUNDARY_RECT),
        help="rect JSON; use [] to exercise the no-rect path")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--ids", nargs="+", default=list(EXPECTED_IDS))
    args = parser.parse_args()

    try:
        rect = json.loads(args.rect)
    except json.JSONDecodeError as error:
        parser.error(f"invalid --rect JSON: {error}")
    if not isinstance(rect, list) or len(rect) not in (0, 8):
        parser.error("--rect must be [] or an array of eight numbers")

    missing = []
    for sample_id in args.ids:
        for suffix in (".jpg", ".png"):
            path = args.sample_dir / f"{sample_id}{suffix}"
            if not path.is_file():
                missing.append(str(path))
    if missing:
        print("missing sample files:\n" + "\n".join(missing), file=sys.stderr)
        return 2

    if not args.library.is_file() or not args.runner.is_file():
        print(
            "library or runner is missing; configure with "
            "-DBUILD_OFFLINE_TESTS=ON and build first",
            file=sys.stderr)
        return 3

    failed = False
    for sample_id in args.ids:
        color_path = (args.sample_dir / f"{sample_id}.jpg").resolve()
        depth_path = (args.sample_dir / f"{sample_id}.png").resolve()
        baseline = None
        try:
            results = detect_repeated(
                args.runner.resolve(), args.library.resolve(),
                color_path, depth_path, args.det_type, rect, args.repeat)
            for iteration, result in enumerate(results):
                failures = validate(result)
                state = tuple(
                    slot.get("state") for slot in result.get("slots", []))
                signature = (
                    result.get("empty_count"),
                    result.get("wheel_count"),
                    result.get("unknown_count"),
                    result.get("tray_complete"),
                    state,
                )
                if baseline is None:
                    baseline = signature
                elif signature != baseline:
                    failures.append("result changed across repeated calls")
                if failures:
                    failed = True
                    print(
                        f"FAIL {sample_id} iteration={iteration}: "
                        + "; ".join(failures))
                    break
            else:
                print(f"PASS {sample_id} repeats={max(1, args.repeat)}")
        except (OSError, RuntimeError, json.JSONDecodeError) as error:
            failed = True
            print(f"FAIL {sample_id}: {error}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
