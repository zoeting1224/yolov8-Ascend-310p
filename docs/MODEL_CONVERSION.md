# 从 PT 复现 Ascend 310P OM 模型

本文从一个刚克隆的仓库开始，完整生成并验证以下文件：

```text
models/weights/pretrained/yolov8n.pt
                    │ Ultralytics 8.0.29 / legacy ONNX exporter
                    ▼
models/weights/onnx/yolov8n.onnx
                    │ ATC，BS=1，640×640，Ascend310P3
                    ├──► models/weights/om/yolov8n_bs1.om
                    └──► models/weights/om/yolov8n_bs1_aipp_rgb.om
```

普通 OM 接收 `float32 NCHW [1,3,640,640]` Host 输入，用于单图对照。AIPP OM 接收 `uint8 NHWC [1,640,640,3]` RGB 输入，设备侧完成 `1/255` 归一化，用于在线摄像头和离线视频网页。

## 1. 前提条件

- Linux aarch64 服务器；
- Ascend 310P3；
- 驱动、固件和 CANN 9.0.0 已由管理员安装；
- CANN 环境脚本位于 `/usr/local/Ascend/cann-9.0.0/set_env.sh`；
- Miniconda 已安装。

先确认设备和 ATC：

```bash
npu-smi info
source /usr/local/Ascend/cann-9.0.0/set_env.sh
atc --version
python -c "import acl; print(acl.__file__)"
```

OM 与目标 SoC/CANN 相关。若硬件不是 310P3，请根据设备修改 `SOC_VERSION`，不要把仓库中的 310P3 OM 当作跨芯片通用文件。

## 2. 克隆仓库并创建环境

```bash
git clone https://github.com/zoeting1224/yolov8-Ascend-310p.git
cd yolov8-Ascend-310p

source ~/miniconda3/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate yolov8-310p
```

安装转换依赖。Ultralytics 必须用 `--no-deps` 单独安装，避免 pip 用 GUI OpenCV 覆盖 `opencv-python-headless`，或替换已经固定的 CPU PyTorch/TorchVision：

```bash
python -m pip install -r requirements-convert.txt
python -m pip install --no-deps "ultralytics==8.0.29"
python -m pip install -r requirements-ascend.txt
```

检查关键版本：

```bash
python - <<'PY'
import onnx, torch, ultralytics
print("torch:", torch.__version__)
print("ultralytics:", ultralytics.__version__)
print("onnx:", onnx.__version__)
PY
```

基线应使用 Python 3.10、PyTorch 2.9.0、Ultralytics 8.0.29 和 ONNX 1.22.0。

## 3. 确认起点 PT

仓库已经包含经过验证的原始权重：

```text
models/weights/pretrained/yolov8n.pt
```

它来自 Ultralytics 官方 assets 的 `v0.0.0` 版本，而不是新版 `v8.3.0` 权重。旧版 Ultralytics 8.0.29 与新版 PT 内部模块路径不兼容。

检查仓库文件：

```bash
sha256sum models/weights/pretrained/yolov8n.pt
```

应得到：

```text
31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95
```

如果仓库没有该文件，可以从官方发布页重新下载并再次校验：

```bash
mkdir -p models/weights/pretrained
curl -L \
  -o models/weights/pretrained/yolov8n.pt \
  https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
sha256sum models/weights/pretrained/yolov8n.pt
```

## 4. 一条命令完成 PT → ONNX → OM → NPU 自检

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
bash scripts/convert_pt_to_om.sh
```

脚本执行顺序：

1. `scripts/export_pt_to_onnx.py` 加载 PT；
2. 设置 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`，兼容 PyTorch 2.6 以后默认的安全加载策略；
3. 为 `torch.onnx.export` 强制指定 `dynamo=False`，使用与 Ultralytics 8.0.29 兼容的 legacy exporter；
4. 以 opset 11、动态 batch/height/width 导出 ONNX；
5. 使用 `onnx.checker` 校验模型和输入名 `images`；
6. ATC 生成普通 BS=1 OM；
7. ATC 结合 `configs/yolov8n_rgb_aipp.cfg` 生成静态 AIPP OM；
8. 加载 AIPP OM，在 NPU 0 上执行一次真实推理并安全释放 ACLRuntime。

若要从其他兼容 PT 开始：

```bash
bash scripts/convert_pt_to_om.sh /absolute/path/to/your_yolov8n.pt
```

若 CANN 路径或 SoC 不同：

```bash
CANN_ENV=/usr/local/Ascend/cann-9.0.0/set_env.sh \
SOC_VERSION=Ascend310P3 \
bash scripts/convert_pt_to_om.sh
```

只转换、不执行 NPU 自检：

```bash
SKIP_RUNTIME_CHECK=1 bash scripts/convert_pt_to_om.sh
```

## 5. 分步执行：PT 转 ONNX

一键脚本内部实际执行：

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python -u scripts/export_pt_to_onnx.py \
  --pt models/weights/pretrained/yolov8n.pt \
  --output models/weights/onnx/yolov8n.onnx \
  --imgsz 640 \
  --opset 11
```

导出器会自动处理 PyTorch 2.9 的 Dynamo 默认行为，无需手工修改 site-packages。成功输出应包含：

```text
ONNX inputs: {'images': ['batch', 3, 'height', 'width']}
ONNX check passed
```

可再次独立检查：

```bash
python - <<'PY'
import onnx
model = onnx.load("models/weights/onnx/yolov8n.onnx")
onnx.checker.check_model(model)
print("opset:", [(item.domain, item.version) for item in model.opset_import])
print("inputs:", [item.name for item in model.graph.input])
print("outputs:", [item.name for item in model.graph.output])
PY
```

## 6. 分步执行：ONNX 转普通 OM

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

检查输出：

```bash
ls -lh models/weights/om/yolov8n_bs1.om
```

## 7. 分步执行：生成静态 AIPP OM

AIPP 配置位于 `configs/yolov8n_rgb_aipp.cfg`，规定 RGB888 U8 输入和设备侧 `1/255` 归一化：

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

## 8. 验证最终模型

```bash
python -u scripts/check_env.py
```

关键结果应为：

```text
input: images [1, 640, 640, 3] dtype.uint8 1228800
output: (1, 84, 8400) float16 finite= True
environment check passed
```

再用普通 OM 对测试图片执行端到端验证：

```bash
python -u scripts/infer_om.py \
  --model models/weights/om/yolov8n_bs1.om \
  --image datasets/images/test/bus.jpg \
  --output runs/predict/bus_result.jpg \
  --metrics-output runs/reports/bus_metrics.json \
  --device 0
```

`check_env.py` 会先执行 `session.free_resource()`，再调用 `InferSession.finalize()`，避免进程退出时在 ACLRuntime 中出现段错误。

## 9. 参考基线

- Ultralytics YOLOv8 8.0.29 对应源码提交：`7a7c8dc7b70cf4bc0be18763a6b66805974ecbe6`；
- Ascend-SACT YOLOv8 参考实现：<https://gitcode.com/Ascend-SACT/YOLOv8>；
- 仓库内四个模型文件的 SHA-256：`models/weights/README.md`。

不同版本的 PyTorch、Ultralytics、CANN 或目标芯片可能产生不同的 ONNX/OM 二进制。复现目标是获得相同的输入输出契约并通过端到端推理验证，不要求不同机器生成的文件逐字节一致。
