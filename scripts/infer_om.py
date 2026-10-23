#!/usr/bin/env python3
"""单张图片 OM 推理工具，同时提供实时网站使用的公共检测函数。

可独立执行图片推理；camera_dashboard.py 还会复用其中的 COCO 类别名称、
画框函数等公共代码。本文件不是摄像头网站入口。
"""
import time
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from ais_bench.infer.interface import InferSession


COCO_NAMES = (
    "person bicycle car motorcycle airplane bus train truck boat traffic-light fire-hydrant "
    "stop-sign parking-meter bench bird cat dog horse sheep cow elephant bear zebra giraffe "
    "backpack umbrella handbag tie suitcase frisbee skis snowboard sports-ball kite baseball-bat "
    "baseball-glove skateboard surfboard tennis-racket bottle wine-glass cup fork knife spoon bowl "
    "banana apple sandwich orange broccoli carrot hot-dog pizza donut cake chair couch potted-plant "
    "bed dining-table toilet tv laptop mouse remote keyboard cell-phone microwave oven toaster sink "
    "refrigerator book clock vase scissors teddy-bear hair-drier toothbrush"
).split()


def parse_args():
    # 定义命令行参数，例如模型路径、图片路径和置信度阈值。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to the .om model")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--output", default="runs/predict/result.jpg", help="Output image path")
    parser.add_argument("--device", type=int, default=0, help="Ascend device ID")
    parser.add_argument("--imgsz", type=int, default=640, help="Square model input size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--metrics-output", help="Optional JSON path for structured inference metrics")
    return parser.parse_args()


def letterbox(image, size):
    """等比例缩放图片并填充到正方形，同时转换成模型需要的 NCHW 输入。"""
    # 记录原图尺寸，OpenCV 图片的 shape 顺序是 (高度, 宽度, 通道数)。
    height, width = image.shape[:2]
    # 使用同一个缩放比例，防止图片被拉伸变形。
    scale = min(size / width, size / height)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    # 计算图片在 640×640（或 --imgsz 指定尺寸）画布中的左、上填充量。
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    # 114 是 YOLO 系列 Letterbox 预处理常用的灰色填充值。
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = resized

    # BGR 转 RGB，HWC 转 CHW，并将像素值从 0~255 归一化到 0~1。
    tensor = canvas[:, :, ::-1].transpose(2, 0, 1)
    tensor = np.ascontiguousarray(tensor, dtype=np.float32) / 255.0
    # tensor[None] 增加 batch 维度，最终形状为 (1, 3, size, size)。
    # scale、left、top 会在后处理中用于把检测框还原到原图坐标。
    return tensor[None], scale, left, top


def postprocess(prediction, scale, left, top, image_shape, conf_threshold, iou_threshold):
    """解析模型输出，执行置信度过滤和 NMS，并把检测框还原到原图。"""
    # OM 输出通常为 (1, 84, 8400)，去掉 batch 维并转置成 (8400, 84)。
    prediction = np.asarray(prediction, dtype=np.float32).squeeze(0).T
    # 前 4 个数是中心点坐标和宽高 (cx, cy, w, h)，其余 80 个数是类别分数。
    boxes_xywh = prediction[:, :4]
    class_scores = prediction[:, 4:]
    # 对每个候选框选择分数最高的类别及对应分数。
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
    # 删除置信度低于 --conf 的候选框。
    keep = scores >= conf_threshold
    boxes_xywh, scores, class_ids = boxes_xywh[keep], scores[keep], class_ids[keep]
    if not len(scores):
        # 没有检测结果时返回空数组，避免后续 NMS 报错。
        return np.empty((0, 4)), scores, class_ids

    # 把 (中心点 x, 中心点 y, 宽, 高) 转换成 (左上 x1, y1, 右下 x2, y2)。
    boxes = np.empty_like(boxes_xywh)
    boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    boxes_tensor = torch.from_numpy(boxes)
    scores_tensor = torch.from_numpy(scores)
    # 不同类别的框增加不同坐标偏移，防止 NMS 在不同类别之间相互抑制。
    offsets = torch.from_numpy(class_ids.astype(np.float32))[:, None] * 7680.0
    # NMS 删除高度重叠的重复候选框，只保留置信度较高的框。
    selected = torchvision.ops.nms(boxes_tensor + offsets, scores_tensor, iou_threshold).numpy()
    boxes, scores, class_ids = boxes[selected], scores[selected], class_ids[selected]

    # 去除 Letterbox 填充并除以缩放比例，将检测框坐标映射回原始图片。
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / scale
    height, width = image_shape[:2]
    # 将坐标限制在原图范围内，避免画框位置越界。
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height - 1)
    return boxes, scores, class_ids


def draw_detections(image, boxes, scores, class_ids):
    """在图片上绘制最终检测框、类别名称和置信度。"""
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.round().astype(int)
        name = COCO_NAMES[class_id] if class_id < len(COCO_NAMES) else str(class_id)
        label = f"{name} {score:.2f}"
        color = (0, 200, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, max(y1 - 7, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return image


def main():
    args = parse_args()
    total_start = time.perf_counter()
    # 使用 OpenCV 读取待检测图片；读取失败时 image 为 None。
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")

    # 预处理原图，得到 OM 模型需要的输入张量以及坐标还原参数。
    preprocess_start = time.perf_counter()
    input_tensor, scale, left, top = letterbox(image, args.imgsz)
    preprocess_time = time.perf_counter() - preprocess_start
    # 创建 Ascend OM 推理会话：第一个参数是 NPU 设备号，第二个参数是模型路径。
    model_load_start = time.perf_counter()
    session = InferSession(args.device, args.model)
    model_load_time = time.perf_counter() - model_load_start

    # 只统计 session.infer() 的单次模型推理时间，不包含预处理和后处理。
    start = time.perf_counter()
    outputs = session.infer([input_tensor])
    inference_time = time.perf_counter() - start

    # inference_time 的单位是秒，乘 1000 后转换为毫秒；FPS 表示每秒理论推理次数。
    print(f"inference time: {inference_time * 1000:.2f} ms")
    print(f"FPS: {1.0 / inference_time:.2f}")

    # 对模型输出进行置信度过滤、NMS 和检测框坐标还原。
    postprocess_start = time.perf_counter()
    boxes, scores, class_ids = postprocess(
        outputs[0], scale, left, top, image.shape, args.conf, args.iou
    )
    # 复制原图后画框，避免直接修改 image 本身。
    result = draw_detections(image.copy(), boxes, scores, class_ids)
    postprocess_time = time.perf_counter() - postprocess_start

    # 创建输出目录并保存画好检测框的结果图片。
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_start = time.perf_counter()
    if not cv2.imwrite(str(output), result):
        raise RuntimeError(f"Failed to write result: {output}")
    save_time = time.perf_counter() - save_start
    total_time = time.perf_counter() - total_start
    # len(boxes) 是经过置信度过滤和 NMS 后最终保留的目标数量。
    print(f"detections: {len(boxes)}")
    print(f"output: {output.resolve()}")

    if args.metrics_output:
        detections = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            class_index = int(class_id)
            detections.append(
                {
                    "class_id": class_index,
                    "class_name": COCO_NAMES[class_index] if class_index < len(COCO_NAMES) else str(class_index),
                    "confidence": float(score),
                    "box_xyxy": [float(value) for value in box],
                }
            )
        metrics = {
            "model": str(Path(args.model).resolve()),
            "image": str(Path(args.image).resolve()),
            "output": str(output.resolve()),
            "device": args.device,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "image_shape": list(image.shape),
            "preprocess_time_ms": preprocess_time * 1000,
            "model_load_time_ms": model_load_time * 1000,
            "inference_time_ms": inference_time * 1000,
            "postprocess_time_ms": postprocess_time * 1000,
            "save_time_ms": save_time * 1000,
            "total_time_ms": total_time * 1000,
            "fps": 1.0 / inference_time,
            "detections_count": len(detections),
            "detections": detections,
        }
        metrics_output = Path(args.metrics_output)
        metrics_output.parent.mkdir(parents=True, exist_ok=True)
        metrics_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"metrics: {metrics_output.resolve()}")


if __name__ == "__main__":
    # 只有直接运行该脚本时才执行 main()。
    main()
