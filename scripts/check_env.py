#!/usr/bin/env python3
"""Validate host packages, CANN Python bindings and the bundled AIPP OM."""

from pathlib import Path
import platform


def main():
    import acl
    import aclruntime
    import cv2
    import numpy as np
    import torch
    import torchvision
    from ais_bench.infer.interface import InferSession

    root = Path(__file__).resolve().parents[1]
    model = root / "models/weights/om/yolov8n_bs1_aipp_rgb.om"
    print("architecture:", platform.machine())
    print("python:", platform.python_version())
    print("numpy:", np.__version__)
    print("opencv:", cv2.__version__)
    print("torch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    print("acl:", acl.__file__)
    print("aclruntime:", aclruntime.__file__)
    print("model:", model)

    session = InferSession(0, str(model))
    try:
        for item in session.get_inputs():
            print("input:", item.name, item.shape, item.datatype, item.realsize)
        image = np.full((1, 640, 640, 3), 114, dtype=np.uint8)
        output = np.asarray(session.infer([image])[0])
        print("output:", output.shape, output.dtype, "finite=", bool(np.isfinite(output).all()))
    finally:
        session.free_resource()
        InferSession.finalize()
    print("environment check passed")


if __name__ == "__main__":
    main()
