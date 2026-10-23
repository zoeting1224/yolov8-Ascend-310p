#!/usr/bin/env python3
"""离线验证工具：测试 YUYV -> DVPP RGB letterbox -> AIPP OM 设备直连。

用于检查 stride、ROI、设备缓冲区和检测结果是否正确，不负责启动网站。
"""

import argparse
import time
from pathlib import Path

import aclruntime
import cv2
import numpy as np
from ais_bench.infer.interface import InferSession

from dvpp_yuyv import DvppYuyvLetterbox
from infer_om import letterbox, postprocess


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw", default="runs/dvpp_camera_yuyv/camera_yuyv422.raw"
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--reference-model", default="models/weights/om/yolov8n_bs1.om")
    parser.add_argument(
        "--aipp-model", default="models/weights/om/yolov8n_bs1_aipp_rgb.om"
    )
    parser.add_argument("--output", default="runs/dvpp_aipp_direct/rgb_letterbox.jpg")
    return parser.parse_args()


def main():
    args = parse_args()
    raw = np.fromfile(args.raw, dtype=np.uint8)
    expected = args.width * args.height * 2
    if raw.size != expected:
        raise RuntimeError(f"raw size={raw.size}, expected={expected}")
    yuyv = raw.reshape(args.height, args.width, 2)
    bgr = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)

    cpu_input, scale, left, top = letterbox(bgr, args.imgsz)
    reference_session = InferSession(args.device, args.reference_model)
    try:
        reference_output = reference_session.infer([cpu_input])[0]
    finally:
        reference_session.free_resource()

    session = InferSession(args.device, args.aipp_model)
    preprocessor = None
    input_tensor = None
    try:
        session.set_context()
        preprocessor = DvppYuyvLetterbox(
            args.width,
            args.height,
            args.imgsz,
            args.imgsz,
            args.device,
            preserve_current_context=True,
        )
        start = time.perf_counter()
        preprocessor.preprocess(yuyv)
        preprocess_ms = (time.perf_counter() - start) * 1000
        input_tensor = aclruntime.BaseTensor(
            preprocessor.output_dev, preprocessor.output_size
        )
        start = time.perf_counter()
        direct_output = session.run([input_tensor], out_array=True)[0]
        inference_ms = (time.perf_counter() - start) * 1000

        rgb = preprocessor.copy_rgb_to_host()
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"failed to write {output}")

        reference_boxes, reference_scores, reference_classes = postprocess(
            reference_output, scale, left, top, bgr.shape, args.conf, args.iou
        )
        direct_boxes, direct_scores, direct_classes = postprocess(
            direct_output,
            preprocessor.scale,
            preprocessor.left,
            preprocessor.top,
            bgr.shape,
            args.conf,
            args.iou,
        )
        diff = np.abs(
            np.asarray(reference_output, dtype=np.float32)
            - np.asarray(direct_output, dtype=np.float32)
        )
        print(
            "letterbox:",
            f"scale={preprocessor.scale}",
            f"resize={preprocessor.resized_width}x{preprocessor.resized_height}",
            f"left={preprocessor.left}",
            f"top={preprocessor.top}",
        )
        print(
            "device buffer:",
            f"stride={preprocessor.output_width_stride}x{preprocessor.output_height_stride}",
            f"size={preprocessor.output_size}",
        )
        print(f"DVPP preprocess: {preprocess_ms:.3f} ms")
        print(f"AIPP OM inference: {inference_ms:.3f} ms")
        print(f"output mean_abs_diff: {float(diff.mean()):.6f}")
        print(f"output max_abs_diff: {float(diff.max()):.6f}")
        print(
            "reference detections:",
            len(reference_boxes),
            reference_classes.tolist(),
            np.round(reference_scores, 3).tolist(),
        )
        print(
            "direct detections:",
            len(direct_boxes),
            direct_classes.tolist(),
            np.round(direct_scores, 3).tolist(),
        )
        print("debug image:", output)
    finally:
        input_tensor = None
        if preprocessor is not None:
            preprocessor.close()
        session.free_resource()


if __name__ == "__main__":
    main()
