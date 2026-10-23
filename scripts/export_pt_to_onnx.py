#!/usr/bin/env python3
"""Export the legacy YOLOv8n PT weight to an ATC-compatible ONNX model."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

# PyTorch 2.6+ defaults to the restricted weights-only loader. The official
# Ultralytics v0.0.0 YOLOv8 weight contains trusted model class objects and is
# loaded through the pinned Ultralytics 8.0.29 package below.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import onnx
import torch
import ultralytics
from ultralytics import YOLO


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", type=Path, required=True, help="source .pt weight")
    parser.add_argument("--output", type=Path, required=True, help="target .onnx file")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=11)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if ultralytics.__version__ != "8.0.29":
        raise RuntimeError(
            f"Ultralytics 8.0.29 is required, found {ultralytics.__version__}"
        )
    torch_version = torch.__version__.split("+")[0]
    if torch_version != "2.9.0":
        raise RuntimeError(f"PyTorch 2.9.0 is required, found {torch.__version__}")

    pt_path = args.pt.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not pt_path.is_file():
        raise FileNotFoundError(f"PT file not found: {pt_path}")

    # torch.onnx.export defaults to dynamo=True in recent PyTorch releases.
    # Ultralytics 8.0.29 predates that exporter, so force the legacy path used
    # during the verified Ascend 310P conversion.
    original_torch_export = torch.onnx.export

    def legacy_torch_export(*export_args, **export_kwargs):
        export_kwargs.setdefault("dynamo", False)
        return original_torch_export(*export_args, **export_kwargs)

    torch.onnx.export = legacy_torch_export

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="yolov8-export-") as temp_dir:
        temp_pt = Path(temp_dir) / "yolov8n.pt"
        shutil.copy2(pt_path, temp_pt)
        exported = Path(
            YOLO(str(temp_pt)).export(
                format="onnx",
                imgsz=args.imgsz,
                opset=args.opset,
                simplify=True,
                dynamic=True,
            )
        )
        if not exported.is_file():
            exported = temp_pt.with_suffix(".onnx")
        if not exported.is_file():
            raise FileNotFoundError(f"Ultralytics did not produce ONNX: {exported}")
        shutil.copy2(exported, output_path)

    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)
    inputs = {
        item.name: [dim.dim_param or dim.dim_value for dim in item.type.tensor_type.shape.dim]
        for item in model.graph.input
    }
    outputs = {
        item.name: [dim.dim_param or dim.dim_value for dim in item.type.tensor_type.shape.dim]
        for item in model.graph.output
    }
    if "images" not in inputs:
        raise RuntimeError(f"Expected ONNX input 'images', found: {list(inputs)}")

    print("PT:", pt_path)
    print("PT SHA-256:", sha256(pt_path))
    print("ONNX:", output_path)
    print("ONNX SHA-256:", sha256(output_path))
    print("ONNX inputs:", inputs)
    print("ONNX outputs:", outputs)
    print("ONNX check passed")


if __name__ == "__main__":
    main()
