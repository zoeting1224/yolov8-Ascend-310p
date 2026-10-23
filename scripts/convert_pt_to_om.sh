#!/usr/bin/env bash
set -euo pipefail

# Rebuild the repository's ONNX, ordinary OM and static-AIPP OM from yolov8n.pt.
# Run from any directory. Override CANN_ENV or SOC_VERSION when needed.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PT_PATH="${1:-${PT_PATH:-${ROOT_DIR}/models/weights/pretrained/yolov8n.pt}}"
ONNX_PATH="${ROOT_DIR}/models/weights/onnx/yolov8n.onnx"
OM_DIR="${ROOT_DIR}/models/weights/om"
AIPP_CONFIG="${ROOT_DIR}/configs/yolov8n_rgb_aipp.cfg"
CANN_ENV="${CANN_ENV:-/usr/local/Ascend/cann-9.0.0/set_env.sh}"
SOC_VERSION="${SOC_VERSION:-Ascend310P3}"

if [[ ! -f "${PT_PATH}" ]]; then
  echo "PT file not found: ${PT_PATH}" >&2
  exit 1
fi
if [[ ! -f "${CANN_ENV}" ]]; then
  echo "CANN environment script not found: ${CANN_ENV}" >&2
  exit 1
fi

mkdir -p "$(dirname "${ONNX_PATH}")" "${OM_DIR}"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python -u "${ROOT_DIR}/scripts/export_pt_to_onnx.py" \
  --pt "${PT_PATH}" \
  --output "${ONNX_PATH}" \
  --imgsz 640 \
  --opset 11

# Vendor environment scripts can reference unset variables, so temporarily
# disable nounset while sourcing CANN.
set +u
# shellcheck disable=SC1090
source "${CANN_ENV}"
set -u

if ! command -v atc >/dev/null 2>&1; then
  echo "atc was not found after sourcing ${CANN_ENV}" >&2
  exit 1
fi

echo "Building ordinary FP32/NCHW-input OM for ${SOC_VERSION} ..."
atc \
  --framework=5 \
  --model="${ONNX_PATH}" \
  --input_format=NCHW \
  --input_shape="images:1,3,640,640" \
  --output_type=FP16 \
  --output="${OM_DIR}/yolov8n_bs1" \
  --soc_version="${SOC_VERSION}" \
  --log=error

echo "Building RGB888/NHWC static-AIPP OM for ${SOC_VERSION} ..."
atc \
  --framework=5 \
  --model="${ONNX_PATH}" \
  --input_format=NCHW \
  --input_shape="images:1,3,640,640" \
  --insert_op_conf="${AIPP_CONFIG}" \
  --output_type=FP16 \
  --output="${OM_DIR}/yolov8n_bs1_aipp_rgb" \
  --soc_version="${SOC_VERSION}" \
  --log=error

ls -lh \
  "${PT_PATH}" \
  "${ONNX_PATH}" \
  "${OM_DIR}/yolov8n_bs1.om" \
  "${OM_DIR}/yolov8n_bs1_aipp_rgb.om"

if [[ "${SKIP_RUNTIME_CHECK:-0}" == "1" ]]; then
  echo "Conversion finished; runtime check skipped (SKIP_RUNTIME_CHECK=1)."
else
  echo "Running AIPP OM input and inference check on NPU 0 ..."
  python -u "${ROOT_DIR}/scripts/check_env.py"
fi
