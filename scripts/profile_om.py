#!/usr/bin/env python3
"""OM 性能分析工具：执行推理、采集 msprof 数据并生成 HTML 报告。

用于定位 NPU 算子耗时和性能瓶颈，不参与摄像头实时推理流水线。
"""

import argparse
import csv
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="runs/predict/result.jpg")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--profile-output", default="runs/msprof")
    parser.add_argument("--report", help="HTML report path")
    parser.add_argument("--metrics-output", help="Normal inference metrics JSON path")
    parser.add_argument("--top", type=int, default=10, help="Number of slow operators printed to terminal")
    parser.add_argument("--msprof", help="Path to msprof executable")
    return parser.parse_args()


def absolute(path):
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def infer_arguments(args, metrics_output=None):
    values = [sys.executable, str(PROJECT_ROOT / "scripts/infer_om.py"), "--model", str(absolute(args.model)), "--image", str(absolute(args.image)), "--output", str(absolute(args.output)), "--device", str(args.device), "--imgsz", str(args.imgsz), "--conf", str(args.conf), "--iou", str(args.iou)]
    if metrics_output is not None:
        values.extend(["--metrics-output", str(metrics_output)])
    return values


def find_msprof(requested):
    if requested:
        return str(absolute(requested))
    discovered = shutil.which("msprof")
    if discovered:
        return discovered
    fallback = Path("/mnt/p3/usr_extend/Ascend/cann-8.5.0/bin/msprof")
    if fallback.exists():
        return str(fallback)
    raise FileNotFoundError("Cannot find msprof; activate the CANN environment or pass --msprof")


def latest_profile(output_dir, previous):
    candidates = [path for path in output_dir.glob("PROF_*") if path.is_dir() and path not in previous]
    if not candidates:
        candidates = [path for path in output_dir.glob("PROF_*") if path.is_dir()]
    if not candidates:
        raise RuntimeError(f"msprof did not create a PROF_* directory under {output_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def print_top_operators(profile_dir, limit):
    files = sorted((profile_dir / "mindstudio_profiler_output").glob("op_summary_*.csv"))
    if not files:
        raise RuntimeError("op_summary CSV was not generated. Check /dev/prof_drv and /dev/ts_aisle permissions.")
    with files[-1].open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    rows.sort(key=lambda row: float(row.get("Task Duration(us)", 0) or 0), reverse=True)
    print(f"top {min(limit, len(rows))} operators:")
    for index, row in enumerate(rows[:limit], 1):
        print(f"{index:02d}. {row.get('OP Type','')} | {float(row.get('Task Duration(us)',0) or 0):.3f} us | {row.get('Op Name','')}")


def main():
    args = parse_args()
    output = absolute(args.output)
    report = absolute(args.report) if args.report else absolute(f"runs/reports/{output.stem}_report.html")
    metrics = absolute(args.metrics_output) if args.metrics_output else absolute(f"runs/reports/{output.stem}_metrics.json")
    profile_output = absolute(args.profile_output)
    profile_output.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    previous = {path.resolve() for path in profile_output.glob("PROF_*") if path.is_dir()}

    print("[1/3] Running normal inference...")
    subprocess.run(infer_arguments(args, metrics), cwd=PROJECT_ROOT, check=True)

    print("[2/3] Collecting Ascend operator profiling data...")
    application = shlex.join(infer_arguments(args))
    msprof_command = [find_msprof(args.msprof), f"--output={profile_output}", f"--application={application}", "--model-execution=on", "--task-time=on", "--ai-core=on", "--aic-metrics=PipeUtilization"]
    subprocess.run(msprof_command, cwd=PROJECT_ROOT, check=True)
    profile_dir = latest_profile(profile_output, previous)
    print_top_operators(profile_dir, args.top)

    print("[3/3] Generating standalone HTML report...")
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts/generate_report.py"), "--metrics", str(metrics), "--profile-dir", str(profile_dir), "--report", str(report)], cwd=PROJECT_ROOT, check=True)
    print(f"profile: {profile_dir}")
    print(f"report: {report}")


if __name__ == "__main__":
    main()
