# 模型转换

仓库已提供在 Ascend 310P3 上验证过的两个 BS=1 模型：

- `models/weights/om/yolov8n_bs1.om`：普通 FP32/NCHW Host 输入模型，供单图推理与对照测试使用。
- `models/weights/om/yolov8n_bs1_aipp_rgb.om`：静态 AIPP 模型，物理输入为 `uint8 NHWC [1,640,640,3]`，供网页在线/离线模式使用。

## PT 导出 ONNX

该模型源于 Ultralytics YOLOv8n。建议在独立转换环境中安装与导出模型一致的版本：

```bash
python -m pip install "ultralytics==8.0.29" onnx onnxsim
python - <<'PY'
from ultralytics import YOLO
YOLO("models/weights/pretrained/yolov8n.pt").export(
    format="onnx", imgsz=640, opset=11, simplify=True, dynamic=False
)
PY
```

## ONNX 转普通 OM

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh

atc \
  --framework=5 \
  --model=models/weights/onnx/yolov8n.onnx \
  --input_format=NCHW \
  --input_shape="images:1,3,640,640" \
  --output_type=FP16 \
  --output=models/weights/om/yolov8n_bs1 \
  --soc_version=Ascend310P3 \
  --log=error
```

## 生成静态 AIPP OM

AIPP 配置位于 `configs/yolov8n_rgb_aipp.cfg`，输入 RGB888 uint8，设备侧执行 1/255 归一化：

```bash
atc \
  --framework=5 \
  --model=models/weights/onnx/yolov8n.onnx \
  --input_format=NCHW \
  --input_shape="images:1,3,640,640" \
  --insert_op_conf=configs/yolov8n_rgb_aipp.cfg \
  --output_type=FP16 \
  --output=models/weights/om/yolov8n_bs1_aipp_rgb \
  --soc_version=Ascend310P3 \
  --log=error
```

ATC 对插入 AIPP 后的物理输入布局可能随 CANN/ONNX 图而变化。生成后必须用 `scripts/check_env.py` 或 `InferSession.get_inputs()` 确认结果为：

```text
images [1, 640, 640, 3] dtype.uint8 1228800
```

不同 CANN 版本重新编译 OM 后，应重新验证输入 shape、颜色顺序和检测结果。OM 与目标 SoC/CANN 组合有关，不能把 310P3 的 OM 当作跨芯片通用文件。
