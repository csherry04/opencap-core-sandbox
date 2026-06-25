#!/usr/bin/env python3
"""Benchmark checkerboard calibration fallback modes on local fixtures.

This script discovers calibration videos from the local repository data and
benchmarks two routes through utilsChecker.calcExtrinsicsFromVideo:

1. primary_only: disables SB fallback, leaving only findChessboardCorners
2. current: the fallback flags currently used by utilsChecker.py
3. exhaustive: the same route, but with CALIB_CB_EXHAUSTIVE added to the SB
   fallback detector flags

The script stages videos into a temporary OpenCap-like folder so source fixture
directories are not modified by generated extrinsic images/pickles.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utilsChecker import calcExtrinsicsFromVideo  # noqa: E402

try:
    import yaml
except ImportError:  # pragma: no cover - dependency is expected in this repo.
    yaml = None


VIDEO_EXTENSIONS = {".avi", ".mov", ".mp4", ".qt"}
DEFAULT_ACL_BIG_TRIALS = {
    "0d524b54-92ca-4a4c-9951-b4b7d3bfdc92",
    "c3634e77-c544-435a-bfb9-bd03de3b2c48",
}


@dataclass(frozen=True)
class CalibrationCase:
    dataset: str
    group: str
    case_id: str
    video_path: Path
    checkerboard: dict[str, Any]
    camera_params_path: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark current vs exhaustive SB calibration fallback.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root used to resolve relative roots.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["utah", "acl", "labvalidation"],
        choices=["utah", "acl", "labvalidation"],
        help="Datasets to discover.",
    )
    parser.add_argument(
        "--utah-root",
        type=Path,
        default=Path("utah_calibration_tests"),
        help="Utah calibration fixture root.",
    )
    parser.add_argument(
        "--acl-root",
        type=Path,
        default=Path("calibration_acl_5240d028-8a8a-4e6e-955d-5223866fefa6"),
        help="ACL raw calibration folder root.",
    )
    parser.add_argument(
        "--labvalidation-root",
        type=Path,
        default=Path("Data/LabValidation"),
        help="LabValidation dataset root.",
    )
    parser.add_argument(
        "--acl-big-trials",
        default=",".join(sorted(DEFAULT_ACL_BIG_TRIALS)),
        help="Comma-separated ACL trial folders using the 11x8 / 60mm board.",
    )
    parser.add_argument(
        "--default-board",
        default="5x4:35",
        help="Default checkerboard as WIDTHxHEIGHT:SQUARE_MM.",
    )
    parser.add_argument(
        "--extra-video",
        action="append",
        type=Path,
        default=[],
        help=(
            "Additional standalone calibration video to benchmark. Can be "
            "specified multiple times. Uses --extra-board unless overridden "
            "by future metadata support."
        ),
    )
    parser.add_argument(
        "--extra-board",
        default=None,
        help=(
            "Checkerboard for --extra-video as WIDTHxHEIGHT:SQUARE_MM. "
            "Defaults to --default-board."
        ),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["current", "exhaustive"],
        choices=["primary_only", "current", "exhaustive"],
        help="Fallback modes to benchmark.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of discovered cases for quick smoke tests.",
    )
    parser.add_argument(
        "--case-filter",
        default=None,
        help="Only include cases whose path or case_id contains this substring.",
    )
    parser.add_argument(
        "--stage-method",
        choices=["auto", "symlink", "copy"],
        default="auto",
        help="How to stage videos into the temporary benchmark directory.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory for staged videos. Uses a temporary directory if omitted.",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Do not delete the temporary staged-video directory.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV path for per-case results.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON path for per-case results and summaries.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print summary tables.",
    )
    return parser.parse_args()


def resolve_root(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def parse_board(value: str) -> dict[str, Any]:
    try:
        dims_part, square_part = value.split(":", 1)
        width, height = dims_part.lower().split("x", 1)
        return {
            "dimensions": (int(width), int(height)),
            "squareSize": float(square_part),
        }
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid board specification {value!r}; expected WIDTHxHEIGHT:SQUARE_MM"
        ) from exc


def checkerboard_from_metadata(metadata_path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if yaml is None or not metadata_path.exists():
        return dict(default)
    with metadata_path.open("r") as f:
        metadata = yaml.safe_load(f) or {}
    checkerboard = metadata.get("checkerBoard") or {}
    width = checkerboard.get("black2BlackCornersWidth_n")
    height = checkerboard.get("black2BlackCornersHeight_n")
    square = checkerboard.get("squareSideLength_mm")
    if width is None or height is None or square is None:
        return dict(default)
    return {"dimensions": (int(width), int(height)), "squareSize": float(square)}


def video_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def find_nearest_camera_params(video_path: Path) -> Path | None:
    for parent in [video_path.parent, *video_path.parents]:
        candidate = parent / "cameraIntrinsicsExtrinsics.pickle"
        if candidate.exists():
            return candidate
    return None


def discover_utah(root: Path, default_board: dict[str, Any]) -> list[CalibrationCase]:
    cases: list[CalibrationCase] = []
    if not root.exists():
        return cases
    for session_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        board = checkerboard_from_metadata(session_dir / "sessionMetadata.yaml", default_board)
        for video_path in video_files(session_dir / "Videos"):
            if "InputMedia" not in video_path.parts or "calibration" not in video_path.parts:
                continue
            camera_params_path = find_nearest_camera_params(video_path)
            case_id = f"{session_dir.name}/{video_path.parent.parent.parent.name}/{video_path.name}"
            cases.append(
                CalibrationCase(
                    dataset="utah",
                    group=session_dir.name,
                    case_id=case_id,
                    video_path=video_path,
                    checkerboard=board,
                    camera_params_path=camera_params_path,
                )
            )
    return cases


def discover_acl(
    root: Path,
    default_board: dict[str, Any],
    acl_big_trials: set[str],
) -> list[CalibrationCase]:
    cases: list[CalibrationCase] = []
    if not root.exists():
        return cases
    big_board = {"dimensions": (11, 8), "squareSize": 60.0}
    for trial_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        board = big_board if trial_dir.name in acl_big_trials else default_board
        for video_path in video_files(trial_dir):
            case_id = f"{trial_dir.name}/{video_path.name}"
            cases.append(
                CalibrationCase(
                    dataset="acl",
                    group=trial_dir.name,
                    case_id=case_id,
                    video_path=video_path,
                    checkerboard=dict(board),
                )
            )
    return cases


def discover_labvalidation(root: Path, default_board: dict[str, Any]) -> list[CalibrationCase]:
    cases: list[CalibrationCase] = []
    if not root.exists():
        return cases
    for video_path in sorted(root.glob("subject*/VideoData/Session*/Cam*/extrinsics/extrinsics.*")):
        if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        subject_dir = next(
            (parent for parent in video_path.parents if parent.name.startswith("subject")),
            None,
        )
        board = dict(default_board)
        if subject_dir is not None:
            board = checkerboard_from_metadata(subject_dir / "sessionMetadata.yaml", board)
        session = video_path.parents[2].name
        cam = video_path.parents[0].name
        subject = subject_dir.name if subject_dir is not None else "unknown_subject"
        camera_params_path = find_nearest_camera_params(video_path)
        case_id = f"{subject}/{session}/{cam}/{video_path.name}"
        cases.append(
            CalibrationCase(
                dataset="labvalidation",
                group=f"{subject}/{session}",
                case_id=case_id,
                video_path=video_path,
                checkerboard=board,
                camera_params_path=camera_params_path,
            )
        )
    return cases


def discover_extra_videos(
    paths: list[Path],
    repo_root: Path,
    board: dict[str, Any],
) -> list[CalibrationCase]:
    cases: list[CalibrationCase] = []
    for raw_path in paths:
        video_path = resolve_root(repo_root, raw_path).resolve()
        if not video_path.exists():
            print(f"Skipping missing --extra-video: {video_path}", file=sys.stderr)
            continue
        if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            print(f"Skipping non-video --extra-video: {video_path}", file=sys.stderr)
            continue
        case_id = video_path.name
        cases.append(
            CalibrationCase(
                dataset="extra",
                group=video_path.parent.name,
                case_id=case_id,
                video_path=video_path,
                checkerboard=dict(board),
                camera_params_path=find_nearest_camera_params(video_path),
            )
        )
    return cases


def load_camera_params(path: Path | None, video_path: Path) -> dict[str, Any]:
    if path is not None and path.exists():
        with path.open("rb") as f:
            params = pickle.load(f)
        if params is not None:
            return params

    capture = cv2.VideoCapture(str(video_path))
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"Could not read video dimensions from {video_path}")
    focal_length = max(width, height)
    return {
        "intrinsicMat": np.array(
            [
                [focal_length, 0.0, width / 2],
                [0.0, focal_length, height / 2],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        "distortion": np.zeros((5, 1), dtype=float),
        "imageSize": np.array([height, width]),
    }


def valid_extrinsics(camera_params: dict[str, Any] | None) -> bool:
    if camera_params is None:
        return False
    for key in ("rotation", "translation", "rotation_EulerAngles"):
        if key not in camera_params:
            return False
        if not np.all(np.isfinite(camera_params[key])):
            return False
    translation_norm = float(np.linalg.norm(camera_params["translation"]))
    return 0 < translation_norm < 100000


def make_stage_path(work_dir: Path, case: CalibrationCase, mode: str) -> Path:
    safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in case.case_id)
    cam_name = "Cam0"
    for part in case.video_path.parts:
        if part.startswith("Cam"):
            cam_name = part
    return (
        work_dir
        / mode
        / "Data"
        / f"{case.dataset}_{safe_id}"
        / "Videos"
        / cam_name
        / "InputMedia"
        / "calibration"
        / case.video_path.name
    )


def stage_video(source: Path, destination: Path, method: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if method in ("auto", "symlink"):
        try:
            os.symlink(source.resolve(), destination)
            return
        except OSError:
            if method == "symlink":
                raise
    shutil.copy2(source, destination)


def run_case(
    case: CalibrationCase,
    mode: str,
    work_dir: Path,
    stage_method: str,
) -> dict[str, Any]:
    stage_path = make_stage_path(work_dir, case, mode)
    stage_video(case.video_path, stage_path, stage_method)
    camera_params = load_camera_params(case.camera_params_path, case.video_path)

    original_primary = cv2.findChessboardCorners
    original_sb = cv2.findChessboardCornersSBWithMeta
    detector_counts = {"primary_calls": 0, "sb_calls": 0}

    def primary_wrapper(*args: Any, **kwargs: Any):
        detector_counts["primary_calls"] += 1
        return original_primary(*args, **kwargs)

    def sb_wrapper(image: Any, pattern_size: Any, flags: int = 0):
        detector_counts["sb_calls"] += 1
        if mode == "primary_only":
            return False, None, None
        if mode == "exhaustive":
            flags = (
                flags
                | cv2.CALIB_CB_EXHAUSTIVE
                | cv2.CALIB_CB_ACCURACY
                | cv2.CALIB_CB_LARGER
            )
        return original_sb(image, pattern_size, flags)

    cv2.findChessboardCorners = primary_wrapper
    cv2.findChessboardCornersSBWithMeta = sb_wrapper

    start = time.perf_counter()
    error = ""
    result = None
    try:
        result = calcExtrinsicsFromVideo(
            str(stage_path),
            camera_params,
            case.checkerboard,
            visualize=False,
            imageUpsampleFactor=2,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark records failure reason.
        error = f"{type(exc).__name__}: {exc}"
    finally:
        elapsed = time.perf_counter() - start
        cv2.findChessboardCorners = original_primary
        cv2.findChessboardCornersSBWithMeta = original_sb

    passed = valid_extrinsics(result)
    return {
        "dataset": case.dataset,
        "group": case.group,
        "case_id": case.case_id,
        "video_path": str(case.video_path),
        "camera_params_path": str(case.camera_params_path) if case.camera_params_path else "",
        "board_dimensions": f"{case.checkerboard['dimensions'][0]}x{case.checkerboard['dimensions'][1]}",
        "square_size_mm": case.checkerboard["squareSize"],
        "mode": mode,
        "status": "PASS" if passed else "FAIL",
        "seconds": elapsed,
        "primary_calls": detector_counts["primary_calls"],
        "sb_calls": detector_counts["sb_calls"],
        "error": "" if passed else error,
    }


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    keys = sorted({(row["dataset"], row["mode"]) for row in results})
    for dataset, mode in keys:
        rows = [row for row in results if row["dataset"] == dataset and row["mode"] == mode]
        summaries.append(
            {
                "dataset": dataset,
                "mode": mode,
                "cases": len(rows),
                "passed": sum(row["status"] == "PASS" for row in rows),
                "failed": sum(row["status"] == "FAIL" for row in rows),
                "seconds": sum(float(row["seconds"]) for row in rows),
                "primary_calls": sum(int(row["primary_calls"]) for row in rows),
                "sb_calls": sum(int(row["sb_calls"]) for row in rows),
            }
        )

    for mode in sorted({row["mode"] for row in results}):
        rows = [row for row in results if row["mode"] == mode]
        summaries.append(
            {
                "dataset": "ALL",
                "mode": mode,
                "cases": len(rows),
                "passed": sum(row["status"] == "PASS" for row in rows),
                "failed": sum(row["status"] == "FAIL" for row in rows),
                "seconds": sum(float(row["seconds"]) for row in rows),
                "primary_calls": sum(int(row["primary_calls"]) for row in rows),
                "sb_calls": sum(int(row["sb_calls"]) for row in rows),
            }
        )
    return summaries


def timing_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p90": 0.0,
        }
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


def timing_summaries(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    keys = sorted({(row["dataset"], row["mode"]) for row in results})
    keys.extend(("ALL", mode) for mode in sorted({row["mode"] for row in results}))

    for dataset, mode in keys:
        if dataset == "ALL":
            rows = [row for row in results if row["mode"] == mode]
        else:
            rows = [
                row
                for row in results
                if row["dataset"] == dataset and row["mode"] == mode
            ]
        all_times = [float(row["seconds"]) for row in rows]
        pass_times = [float(row["seconds"]) for row in rows if row["status"] == "PASS"]
        fail_times = [float(row["seconds"]) for row in rows if row["status"] == "FAIL"]
        all_stats = timing_stats(all_times)
        pass_stats = timing_stats(pass_times)
        fail_stats = timing_stats(fail_times)
        summaries.append(
            {
                "dataset": dataset,
                "mode": mode,
                "cases": len(rows),
                "mean": all_stats["mean"],
                "median": all_stats["median"],
                "min": all_stats["min"],
                "max": all_stats["max"],
                "p90": all_stats["p90"],
                "pass_mean": pass_stats["mean"],
                "fail_mean": fail_stats["mean"],
            }
        )
    return summaries


def print_summary(summaries: list[dict[str, Any]]) -> None:
    rows = [
        [
            row["dataset"],
            row["mode"],
            row["cases"],
            row["passed"],
            row["failed"],
            f"{row['seconds']:.3f}s",
            row["primary_calls"],
            row["sb_calls"],
        ]
        for row in summaries
    ]
    print_table(
        "\nPass/Fail Summary",
        ["Dataset", "Mode", "Cases", "Pass", "Fail", "Total Time", "Primary", "SB"],
        rows,
    )


def print_timing_summary(summaries: list[dict[str, Any]]) -> None:
    rows = [
        [
            row["dataset"],
            row["mode"],
            row["cases"],
            f"{row['mean']:.3f}s",
            f"{row['median']:.3f}s",
            f"{row['min']:.3f}s",
            f"{row['max']:.3f}s",
            f"{row['p90']:.3f}s",
            f"{row['pass_mean']:.3f}s" if row["pass_mean"] else "-",
            f"{row['fail_mean']:.3f}s" if row["fail_mean"] else "-",
        ]
        for row in summaries
    ]
    print_table(
        "\nPer-Case Timing Stats",
        [
            "Dataset",
            "Mode",
            "Cases",
            "Mean",
            "Median",
            "Min",
            "Max",
            "P90",
            "Pass Mean",
            "Fail Mean",
        ],
        rows,
    )


def print_slowest_cases(results: list[dict[str, Any]], top_n: int = 10) -> None:
    rows = [
        [
            f"{float(row['seconds']):.3f}s",
            row["status"],
            row["dataset"],
            row["mode"],
            shorten_case_id(row["case_id"]),
            row["primary_calls"],
            row["sb_calls"],
        ]
        for row in sorted(results, key=lambda item: float(item["seconds"]), reverse=True)[
            :top_n
        ]
    ]
    print_table(
        f"\nSlowest Cases (Top {top_n})",
        ["Time", "Status", "Dataset", "Mode", "Case", "Primary", "SB"],
        rows,
    )


def print_mode_deltas(results: list[dict[str, Any]]) -> None:
    by_case: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in results:
        key = (row["dataset"], row["case_id"])
        by_case.setdefault(key, {})[row["mode"]] = row

    comparisons = [
        ("current", "exhaustive"),
        ("primary_only", "current"),
        ("primary_only", "exhaustive"),
    ]
    for left, right in comparisons:
        rows = []
        for (dataset, case_id), modes in by_case.items():
            if left not in modes or right not in modes:
                continue
            left_row = modes[left]
            right_row = modes[right]
            rows.append(
                {
                    "dataset": dataset,
                    "case_id": case_id,
                    "left": left,
                    "right": right,
                    "delta": float(right_row["seconds"]) - float(left_row["seconds"]),
                    "left_status": left_row["status"],
                    "right_status": right_row["status"],
                    "left_seconds": float(left_row["seconds"]),
                    "right_seconds": float(right_row["seconds"]),
                }
            )
        if not rows:
            continue
        deltas = [row["delta"] for row in rows]
        stats = timing_stats(deltas)
        print_table(
            f"\nMode Timing Delta: {right} minus {left}",
            [
                "Cases",
                "Mean",
                "Median",
                "Min",
                "Max",
                "P90",
                f"{right} Faster",
                f"{right} Slower",
            ],
            [
                [
                    len(rows),
                    format_delta(stats["mean"]),
                    format_delta(stats["median"]),
                    format_delta(stats["min"]),
                    format_delta(stats["max"]),
                    format_delta(stats["p90"]),
                    sum(row["delta"] < 0 for row in rows),
                    sum(row["delta"] > 0 for row in rows),
                ]
            ],
        )
        speedups = [row for row in sorted(rows, key=lambda item: item["delta"]) if row["delta"] < 0][:5]
        slowdowns = [
            row for row in sorted(rows, key=lambda item: item["delta"], reverse=True) if row["delta"] > 0
        ][:5]
        if speedups:
            print_delta_cases("Largest speedups", left, right, speedups)
        if slowdowns:
            print_delta_cases("Largest slowdowns", left, right, slowdowns)


def print_failures(results: list[dict[str, Any]]) -> None:
    failures = [row for row in results if row["status"] == "FAIL"]
    if not failures:
        print("\nFAILURES: none")
        return
    rows = [
        [
            row["dataset"],
            row["mode"],
            shorten_case_id(row["case_id"]),
            f"{row['board_dimensions']}:{row['square_size_mm']}mm",
            f"{row['seconds']:.3f}s",
            shorten_error(row["error"]),
        ]
        for row in failures
    ]
    print_table(
        "\nFailures",
        ["Dataset", "Mode", "Case", "Board", "Time", "Error"],
        rows,
    )


def print_table(title: str, headers: list[str], rows: list[list[Any]]) -> None:
    print(title)
    if not rows:
        print("  none")
        return
    rendered = [[str(value) for value in row] for row in rows]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rendered))
        for index in range(len(headers))
    ]
    header = "  " + "  ".join(
        headers[index].ljust(widths[index]) for index in range(len(headers))
    )
    divider = "  " + "  ".join("-" * widths[index] for index in range(len(headers)))
    print(header)
    print(divider)
    for row in rendered:
        print("  " + "  ".join(row[index].ljust(widths[index]) for index in range(len(headers))))


def format_delta(seconds: float) -> str:
    if abs(seconds) < 0.0005:
        return "0.000s"
    sign = "+" if seconds > 0 else "-"
    return f"{sign}{abs(seconds):.3f}s"


def shorten_case_id(case_id: str, max_len: int = 72) -> str:
    if len(case_id) <= max_len:
        return case_id
    return "..." + case_id[-(max_len - 3):]


def shorten_error(error: str, max_len: int = 90) -> str:
    if not error:
        return ""
    if "checkerboard was not detected" in error:
        return "checkerboard not detected"
    if len(error) <= max_len:
        return error
    return error[: max_len - 3] + "..."


def print_delta_cases(
    title: str,
    left: str,
    right: str,
    rows: list[dict[str, Any]],
) -> None:
    table_rows = [
        [
            format_delta(row["delta"]),
            row["dataset"],
            shorten_case_id(row["case_id"]),
            f"{row['left_seconds']:.3f}s/{row['left_status']}",
            f"{row['right_seconds']:.3f}s/{row['right_status']}",
        ]
        for row in rows
    ]
    print_table(
        title,
        ["Delta", "Dataset", "Case", left, right],
        table_rows,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "group",
        "case_id",
        "video_path",
        "camera_params_path",
        "board_dimensions",
        "square_size_mm",
        "mode",
        "status",
        "seconds",
        "primary_calls",
        "sb_calls",
        "error",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    default_board = parse_board(args.default_board)
    extra_board = parse_board(args.extra_board) if args.extra_board else default_board
    acl_big_trials = {value for value in args.acl_big_trials.split(",") if value}

    cases: list[CalibrationCase] = []
    if "utah" in args.datasets:
        cases.extend(discover_utah(resolve_root(repo_root, args.utah_root), default_board))
    if "acl" in args.datasets:
        cases.extend(discover_acl(resolve_root(repo_root, args.acl_root), default_board, acl_big_trials))
    if "labvalidation" in args.datasets:
        cases.extend(discover_labvalidation(resolve_root(repo_root, args.labvalidation_root), default_board))
    if args.extra_video:
        cases.extend(discover_extra_videos(args.extra_video, repo_root, extra_board))

    if args.case_filter:
        cases = [
            case
            for case in cases
            if args.case_filter in case.case_id or args.case_filter in str(case.video_path)
        ]
    if args.limit is not None:
        cases = cases[: args.limit]

    if not cases:
        print("No calibration videos discovered.", file=sys.stderr)
        return 1

    created_temp = args.work_dir is None
    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="calibration_fallback_benchmark_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"Discovered {len(cases)} calibration videos.")
    print(f"Work directory: {work_dir}")
    print(f"Modes: {', '.join(args.modes)}")

    results: list[dict[str, Any]] = []
    try:
        for mode in args.modes:
            print(f"\nRunning mode: {mode}")
            for index, case in enumerate(cases, start=1):
                row = run_case(case, mode, work_dir, args.stage_method)
                results.append(row)
                if not args.quiet:
                    print(
                        f"[{index}/{len(cases)}] {row['status']} {row['seconds']:.3f}s "
                        f"{row['dataset']} {row['case_id']} "
                        f"primary={row['primary_calls']} sb={row['sb_calls']}"
                    )
    finally:
        if created_temp and not args.keep_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)

    summaries = summarize(results)
    timing = timing_summaries(results)
    print_summary(summaries)
    print_timing_summary(timing)
    print_slowest_cases(results)
    print_mode_deltas(results)
    print_failures(results)

    if args.output_csv:
        write_csv(resolve_root(repo_root, args.output_csv), results)
        print(f"\nWrote CSV: {resolve_root(repo_root, args.output_csv)}")
    if args.output_json:
        output_path = resolve_root(repo_root, args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(
                {"results": results, "summary": summaries, "timing": timing},
                f,
                indent=2,
            )
        print(f"Wrote JSON: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
