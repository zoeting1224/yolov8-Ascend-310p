# 模型转换

仓库已提供在 Ascend 310P3 上验证过的两个 BS=1 模型：

- `models/weights/om/yolov8n_bs1.om`：普通 FP32/NCHW Host 输入模型，供单图推理与对照测试使用。
- `models/weights/om/yolov8n_bs1_aipp_rgb.om`：静态 AIPP 模型，物理输入为 `uint8 NHWC [1,640,640,3]`，供网页在线/离线模式使用。

完整转换链路是：

```text
yolov8n.pt → yolov8n.onnx → yolov8n_bs1.om
                            → yolov8n_bs1_aipp_rgb.om
```

## 一条命令完成转换

在仓库根目录执行：

```bash
conda activate yolov8-310p
python -m pip install "ultralytics==8.0.29" onnx onnxsim
bash scripts/convert_pt_to_om.sh
```

脚本默认读取 `models/weights/pretrained/yolov8n.pt`，自动加载 `/usr/local/Ascend/cann-9.0.0/set_env.sh`，生成 ONNX 和两个 OM，并在 NPU 0 上完成一次推理自检。

若 CANN 安装位置或目标 SoC 不同，可以显式指定：

```bash
CANN_ENV=/usr/local/Ascend/cann-9.0.0/set_env.sh \
SOC_VERSION=Ascend310P3 \
bash scripts/convert_pt_to_om.sh
```

以下各节是该脚本执行的具体步骤。

## PT 导出 ONNX

该模型源于 Ultralytics YOLOv8n。建议在独立转换环境中安装与导出模型一致的版本：

```bash
python -m pip install "ultralytics==8.0.29" onnx onnxsim
python - <<'PY'
import shutil
from pathlib import Path

import onnx
from ultralytics import YOLO

pt_path = Path("models/weights/pretrained/yolov8n.pt")
exported = Path(YOLO(str(pt_path)).export(
    format="onnx", imgsz=640, opset=11, simplify=True, dynamic=False
))
onnx_path = Path("models/weights/onnx/yolov8n.onnx")
onnx_path.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(exported, onnx_path)
onnx.checker.check_model(onnx.load(str(onnx_path)))
print("ONNX check passed:", onnx_path)
PY
```

Ultralytics 8.0.29 默认把 ONNX 导出到 PT 文件旁边，上面的 `copy2` 是必要步骤，确保后续 ATC 使用固定路径 `models/weights/onnx/yolov8n.onnx`。

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

执行完整推理自检：

```bash
python -u scripts/check_env.py
```

预期输出张量为 `(1, 84, 8400) float16`，且所有数值有限。自检脚本会按 `session.free_resource()`、`InferSession.finalize()` 的顺序释放资源，避免进程退出阶段出现 ACLRuntime 段错误。

不同 CANN 版本重新编译 OM 后，应重新验证输入 shape、颜色顺序和检测结果。OM 与目标 SoC/CANN 组合有关，不能把 310P3 的 OM 当作跨芯片通用文件。
