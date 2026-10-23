# 模型文件

| 文件 | 用途 | SHA-256 |
| --- | --- | --- |
| `pretrained/yolov8n.pt` | PT 转 ONNX 的起点；Ultralytics assets `v0.0.0` 版本 | `31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95` |
| `onnx/yolov8n.onnx` | ATC 转换输入；动态 batch/height/width，opset 11 | `da0169c2ec585974591a3115cdfcace4267fef112d8bb1827f57d33d441a3140` |
| `om/yolov8n_bs1_aipp_rgb.om` | 网页在线/离线推理，RGB888 uint8 NHWC 输入 | `0b94b1e8b2cbe1e8e4e511578cf4a04c3d8b73099125ed6d188dc42ecda8e00a` |
| `om/yolov8n_bs1.om` | 单图推理与 AIPP 对照测试，FP32 NCHW 输入 | `26b90f354e2c3f21e6da61ec8d7e8bc8ea615c9f34f56d4d44719b0335d84daa` |

仓库中的 PT 已与官方发布文件逐字节比较，SHA-256 完全一致。以上 OM 按 Ascend 310P3 编译并完成验证。完整转换方法见 `docs/MODEL_CONVERSION.md`。
