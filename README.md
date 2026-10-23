# YOLOv8 on Ascend 310P：在线摄像头与离线视频分析

这是一个面向 **Ascend 310P3** 的 YOLOv8n 推理项目。网页提供两个入口：

| 模式 | 是否需要摄像头 | 输入 | 输出 |
| --- | --- | --- | --- |
| 离线视频分析 | 否 | 浏览器上传 MP4/AVI/MOV/MKV/WebM 等视频 | 实时进度、检测预览、可播放/下载的 WebM 结果 |
| 在线摄像头检测 | 是 | `/dev/video0` 的 YUYV USB 摄像头 | 浏览器 MJPEG 原始画面与检测画面、流水线耗时与 FPS |

没有摄像头时可以完整使用离线功能。在线模式采用 `YUYV → H2D → DVPP RGB letterbox → AIPP OM → NPU → CPU NMS/画框 → MJPEG`；离线模式用 OpenCV 解码视频，再将 `RGB888 uint8 NHWC` 输入静态 AIPP OM。

## 已验证环境

最终迁移记录对应以下环境：

- 服务器：双卡 Ascend 310P3（程序通过 `--device 0` 使用一张卡）
- 架构：Linux aarch64
- Python：3.10.20
- CANN：9.0.0，路径 `/usr/local/Ascend/cann-9.0.0`
- `aclruntime`：0.0.2
- `ais-bench`：0.0.2
- NumPy：1.26.4
- OpenCV：4.11.0
- PyTorch：2.9.0（CPU，仅用于后处理）
- TorchVision：0.24.0（CPU NMS）
- 模型输入：`images [1,640,640,3] uint8`，输出 `(1,84,8400) float16`

项目也曾在 CANN 8.5.0 的 310P 环境运行，但 README 以最终验证的 CANN 9.0.0 环境为复现基线。CANN 与驱动必须由管理员预先正确安装；不同版本请使用与之配套的 ACLRuntime/AISBench。

## 目录

```text
configs/                         AIPP 配置
models/weights/                  PT、ONNX、普通 OM、AIPP OM
requirements-convert.txt         PT → ONNX 的固定版本依赖
scripts/camera_dashboard.py      在线 + 离线网页入口
scripts/dvpp_yuyv.py             摄像头 YUYV 的 DVPP 预处理
scripts/infer_om.py              普通 OM 单图推理与公共后处理
scripts/check_env.py             环境、模型输入与一次推理自检
scripts/export_pt_to_onnx.py     兼容 PyTorch 2.9 的 PT → ONNX 导出器
scripts/convert_pt_to_om.sh      PT → ONNX → 普通 OM/AIPP OM 完整转换
datasets/images/test/bus.jpg     自检图片
runs/                            运行结果（默认不提交）
```

## 核心模型转换：PT → ONNX → OM

仓库不仅提供已经验证的模型文件，也包含从原始 `yolov8n.pt` 重新生成部署模型的完整链路：

```text
models/weights/pretrained/yolov8n.pt
                │ Ultralytics 8.0.29，opset 11，动态 ONNX
                ▼
models/weights/onnx/yolov8n.onnx
                │ ATC / Ascend310P3
                ├────────► models/weights/om/yolov8n_bs1.om
                │           FP32 NCHW Host 输入，用于单图对照
                └────────► models/weights/om/yolov8n_bs1_aipp_rgb.om
                            RGB888 uint8 NHWC 物理输入，用于网页推理
```

仓库内的 PT 与 Ultralytics 官方 `v0.0.0/yolov8n.pt` 完全一致，SHA-256 为：

```text
31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95
```

从一个刚克隆的仓库开始，在装有 CANN 9.0.0 的 Ascend 310P3 服务器上依次执行：

```bash
git clone https://github.com/zoeting1224/yolov8-Ascend-310p.git
cd yolov8-Ascend-310p

source ~/miniconda3/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate yolov8-310p

python -m pip install -r requirements-convert.txt
python -m pip install --no-deps "ultralytics==8.0.29"
python -m pip install -r requirements-ascend.txt

source /usr/local/Ascend/cann-9.0.0/set_env.sh
bash scripts/convert_pt_to_om.sh
```

脚本会依次完成 PT 加载、旧版 ONNX exporter 兼容处理、ONNX 合法性检查、ATC 生成普通 OM、插入静态 AIPP 生成网页使用的 OM，最后在 NPU 0 上执行一次 AIPP OM 自检。也可以把自己的兼容 PT 作为第一个参数：

```bash
bash scripts/convert_pt_to_om.sh /absolute/path/to/your_yolov8n.pt
```

项目固定使用 Ultralytics 8.0.29。较新 Ultralytics 训练出的 PT 可能包含不同的 Python 模块路径，不能保证被旧版代码直接加载。要复现仓库的完整基线，应先使用仓库自带的 PT。详细的分步命令、输入输出规格和版本注意事项见[模型转换说明](docs/MODEL_CONVERSION.md)。

## 1. 克隆并创建环境

```bash
git clone https://github.com/zoeting1224/yolov8-Ascend-310p.git
cd yolov8-Ascend-310p

source ~/miniconda3/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate yolov8-310p
```

如果已经有 Python 3.10 的 Conda 环境，也可以执行：

```bash
python -m pip install -r requirements.txt
```

## 2. 加载 CANN 并安装 AISBench

每次打开新终端都要先加载 CANN：

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
```

确认下面两个命令正常：

```bash
npu-smi info
python -c "import acl; print(acl.__file__)"
```

安装已验证的 Python 3.10/aarch64 推理轮子：

```bash
python -m pip install -r requirements-ascend.txt
```

> 如果你的 CANN、Python 或架构不同，不要强装这里的 `cp310-linux_aarch64` 轮子，请从昇腾官方渠道选择对应版本。

## 3. 一次性自检

```bash
python -u scripts/check_env.py
```

预期关键信息：

```text
input: images [1, 640, 640, 3] dtype.uint8 1228800
output: (1, 84, 8400) float16 finite= True
environment check passed
```

自检脚本和网页程序都按以下顺序清理资源：先 `session.free_resource()`，最后 `InferSession.finalize()`。不要在同一进程中重复手工调用 ACL finalize。

## 4A. 无摄像头：离线视频分析

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolov8-310p
source /usr/local/Ascend/cann-9.0.0/set_env.sh

cd ~/yolov8-Ascend-310p
python -u scripts/camera_dashboard.py \
  --offline-only \
  --device 0 \
  --host 0.0.0.0 \
  --port 8000
```

浏览器打开：

```text
http://<服务器IP>:8000
```

选择视频后点击“上传并开始检测”。页面会显示处理进度、已处理帧数、离线 FPS、最新检测帧，并在完成后提供结果视频的播放与下载。

## 4B. 有摄像头：在线检测，同时保留离线功能

先检查摄像头：

```bash
v4l2-ctl --list-devices
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

验证配置使用 YUYV、1024×576、15 FPS。启动：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolov8-310p
source /usr/local/Ascend/cann-9.0.0/set_env.sh

cd ~/yolov8-Ascend-310p
python -u scripts/camera_dashboard.py \
  --device 0 \
  --camera 0 \
  --camera-width 1024 \
  --camera-height 576 \
  --camera-fps 15 \
  --host 0.0.0.0 \
  --port 8000
```

网页默认显示离线页签；点击“实时检测”切换到摄像头画面。启动日志中的 `http://0.0.0.0:8000` 是监听地址，浏览器应使用服务器实际 IP。

如果没接摄像头却使用了这个命令，程序会提示 `/dev/video0` 不存在，离线功能仍可使用；无摄像头时建议明确加 `--offline-only`。

## 5. 单图验证普通 OM

```bash
python -u scripts/infer_om.py \
  --model models/weights/om/yolov8n_bs1.om \
  --image datasets/images/test/bus.jpg \
  --output runs/predict/bus_result.jpg \
  --metrics-output runs/reports/bus_metrics.json \
  --device 0
```

## 常用参数

```text
--device 0           NPU 逻辑设备号
--model PATH         AIPP OM 路径
--conf 0.25          置信度阈值
--iou 0.45           NMS IoU 阈值
--warmup 5           首次正式推理前的预热次数
--jpeg-quality 80    在线 MJPEG 和离线预览质量
--offline-only       不打开 V4L2 摄像头
```

## 常见问题

### `ModuleNotFoundError: No module named 'acl'`

当前终端没有加载 CANN：

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
```

### `OSError: [Errno 98] Address already in use`

```bash
ss -ltnp | grep ':8000'
```

停止旧进程，或换端口启动：

```bash
python -u scripts/camera_dashboard.py --offline-only --port 8001
```

### `libGL.so.1` 缺失

项目使用 `opencv-python-headless`。请卸载非 headless 的 OpenCV 后重装：

```bash
python -m pip uninstall -y opencv-python opencv-contrib-python
python -m pip install --force-reinstall "opencv-python-headless==4.11.0.86"
```

### 离线结果视频无法创建

离线输出使用 WebM/VP8。确认 OpenCV 的 FFmpeg/VP8 编码能力可用；必要时安装系统 FFmpeg，并重新安装支持视频编解码的 OpenCV 包。

### 摄像头无法打开

确认设备节点和权限：

```bash
ls -l /dev/video*
groups
```

同时确认摄像头支持启动参数中的 YUYV 分辨率和帧率。

## 模型与兼容性

默认网页模型是 `models/weights/om/yolov8n_bs1_aipp_rgb.om`。该文件针对 Ascend 310P3 编译，SHA-256 和普通 OM 的用途见 `models/weights/README.md`。重新导出 ONNX、ATC 转换及 AIPP 配置见 [模型转换说明](docs/MODEL_CONVERSION.md)。

历史实测中，DVPP 预处理约 1.2–2.4 ms、NPU 推理约 29.3 ms；这些是特定服务器、摄像头与软件栈上的记录，不是通用性能承诺。

## 依赖与许可提示

本仓库未复制完整 Ultralytics 源码。PT/ONNX/OM 来自 YOLOv8n 预训练模型链路；CANN、pyACL、AISBench、Ultralytics、PyTorch、TorchVision 与 OpenCV 均适用各自许可。详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
