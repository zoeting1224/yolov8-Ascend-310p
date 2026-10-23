"""当前正式 DVPP 模块：处理 V4L2 摄像头的 YUYV 帧。

主要提供设备侧 YUYV -> RGB888、resize 和 letterbox，并保留输出设备指针，
供 camera_dashboard.py 通过 BaseTensor 直接送入 AIPP OM，避免图像结果 D2H。
"""

import ctypes
import threading

import acl
import numpy as np


PIXEL_FORMAT_YUYV_PACKED_422 = 7
PIXEL_FORMAT_RGB_888 = 12
PIXEL_FORMAT_BGR_888 = 13
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2

_ACL_LOCK = threading.Lock()
_ACL_INITIALIZED = False


def _require_success(ret, operation):
    if ret != 0:
        raise RuntimeError(f"{operation} failed: {ret}")


def _initialize_acl(device, set_device=True):
    global _ACL_INITIALIZED
    with _ACL_LOCK:
        if not _ACL_INITIALIZED:
            ret = acl.init()
            # ais_bench InferSession may initialize ACL before the camera thread.
            # 100002 means repeat initialization and is safe to reuse here.
            if ret not in (0, 100002):
                raise RuntimeError(f"acl.init failed: {ret}")
            _ACL_INITIALIZED = True
    if set_device:
        _require_success(acl.rt.set_device(device), "acl.rt.set_device")


class DvppYuyvConverter:
    """Convert fixed-size YUYV422 frames to BGR888 using one persistent DVPP channel."""

    def __init__(self, width, height, device=0):
        self.width = width
        self.height = height
        self.device = device
        # CANN widthStride is measured in bytes for packed formats.
        self.input_width_stride = (width * 2 + 15) // 16 * 16
        self.input_size = self.input_width_stride * height
        self.output_width_stride = (width * 3 + 15) // 16 * 16
        self.output_height_stride = (height + 1) // 2 * 2
        self.output_size = self.output_width_stride * self.output_height_stride
        self.channel_desc = None
        self.stream = None
        self.input_desc = None
        self.output_desc = None
        self.input_dev = None
        self.output_dev = None
        self.host_output = None
        self._create()

    def _create(self):
        _initialize_acl(self.device)
        try:
            self.channel_desc = acl.media.dvpp_create_channel_desc()
            _require_success(
                acl.media.dvpp_create_channel(self.channel_desc),
                "dvpp_create_channel",
            )
            self.stream, ret = acl.rt.create_stream()
            _require_success(ret, "acl.rt.create_stream")

            self.input_dev, ret = acl.media.dvpp_malloc(self.input_size)
            _require_success(ret, "dvpp_malloc input")
            self.output_dev, ret = acl.media.dvpp_malloc(self.output_size)
            _require_success(ret, "dvpp_malloc output")

            self.input_desc = acl.media.dvpp_create_pic_desc()
            acl.media.dvpp_set_pic_desc_data(self.input_desc, self.input_dev)
            acl.media.dvpp_set_pic_desc_size(self.input_desc, self.input_size)
            acl.media.dvpp_set_pic_desc_format(
                self.input_desc, PIXEL_FORMAT_YUYV_PACKED_422
            )
            acl.media.dvpp_set_pic_desc_width(self.input_desc, self.width)
            acl.media.dvpp_set_pic_desc_height(self.input_desc, self.height)
            acl.media.dvpp_set_pic_desc_width_stride(
                self.input_desc, self.input_width_stride
            )
            acl.media.dvpp_set_pic_desc_height_stride(self.input_desc, self.height)

            self.output_desc = acl.media.dvpp_create_pic_desc()
            acl.media.dvpp_set_pic_desc_data(self.output_desc, self.output_dev)
            acl.media.dvpp_set_pic_desc_size(self.output_desc, self.output_size)
            acl.media.dvpp_set_pic_desc_format(self.output_desc, PIXEL_FORMAT_BGR_888)
            # CANN 8.5 ConvertColor requires output geometry fields to start at zero.
            acl.media.dvpp_set_pic_desc_width(self.output_desc, 0)
            acl.media.dvpp_set_pic_desc_height(self.output_desc, 0)
            acl.media.dvpp_set_pic_desc_width_stride(self.output_desc, 0)
            acl.media.dvpp_set_pic_desc_height_stride(self.output_desc, 0)
            self.host_output = ctypes.create_string_buffer(self.output_size)
        except Exception:
            self.close()
            raise

    def convert(self, yuyv_frame):
        frame = np.ascontiguousarray(yuyv_frame, dtype=np.uint8)
        if frame.shape != (self.height, self.width, 2):
            raise ValueError(
                f"Unexpected YUYV shape {frame.shape}; "
                f"expected {(self.height, self.width, 2)}"
            )
        frame_bytes = frame.tobytes()
        _require_success(
            acl.rt.memcpy(
                self.input_dev,
                self.input_size,
                acl.util.bytes_to_ptr(frame_bytes),
                self.input_size,
                ACL_MEMCPY_HOST_TO_DEVICE,
            ),
            "YUYV host_to_device memcpy",
        )
        _require_success(
            acl.media.dvpp_vpc_convert_color_async(
                self.channel_desc,
                self.input_desc,
                self.output_desc,
                self.stream,
            ),
            "dvpp_vpc_convert_color_async",
        )
        _require_success(
            acl.rt.synchronize_stream(self.stream),
            "acl.rt.synchronize_stream",
        )
        _require_success(
            acl.media.dvpp_get_pic_desc_ret_code(self.output_desc),
            "VPC output ret_code",
        )
        _require_success(
            acl.rt.memcpy(
                ctypes.addressof(self.host_output),
                self.output_size,
                self.output_dev,
                self.output_size,
                ACL_MEMCPY_DEVICE_TO_HOST,
            ),
            "BGR device_to_host memcpy",
        )
        bgr = np.frombuffer(self.host_output.raw, dtype=np.uint8).reshape(
            self.output_height_stride,
            self.output_width_stride,
        )
        return bgr[: self.height, : self.width * 3].reshape(
            self.height, self.width, 3
        ).copy()

    def close(self):
        if self.output_desc is not None:
            acl.media.dvpp_destroy_pic_desc(self.output_desc)
            self.output_desc = None
        if self.input_desc is not None:
            acl.media.dvpp_destroy_pic_desc(self.input_desc)
            self.input_desc = None
        if self.output_dev is not None:
            acl.media.dvpp_free(self.output_dev)
            self.output_dev = None
        if self.input_dev is not None:
            acl.media.dvpp_free(self.input_dev)
            self.input_dev = None
        if self.stream is not None:
            acl.rt.destroy_stream(self.stream)
            self.stream = None
        if self.channel_desc is not None:
            acl.media.dvpp_destroy_channel(self.channel_desc)
            acl.media.dvpp_destroy_channel_desc(self.channel_desc)
            self.channel_desc = None


class DvppYuyvLetterbox:
    """Prepare one YUYV422 frame as a device-resident RGB888 letterbox image.

    The output buffer stays in DVPP/device memory so it can be wrapped by
    ``aclruntime.BaseTensor`` and passed directly to an AIPP-enabled OM model.
    """

    def __init__(
        self,
        width,
        height,
        output_width=640,
        output_height=640,
        device=0,
        preserve_current_context=False,
    ):
        if width % 2 or height % 2:
            raise ValueError("YUYV input width and height must be even")
        if output_width % 2 or output_height % 2:
            raise ValueError("DVPP output width and height must be even")

        self.width = width
        self.height = height
        self.output_width = output_width
        self.output_height = output_height
        self.device = device
        self.input_width_stride = (width * 2 + 15) // 16 * 16
        self.input_height_stride = (height + 1) // 2 * 2
        self.input_size = self.input_width_stride * self.input_height_stride
        self.output_width_stride = (output_width * 3 + 15) // 16 * 16
        self.output_height_stride = (output_height + 1) // 2 * 2
        self.output_size = self.output_width_stride * self.output_height_stride

        self.scale = min(output_width / width, output_height / height)
        self.resized_width = int(round(width * self.scale))
        self.resized_height = int(round(height * self.scale))
        # DVPP YUV ROI coordinates require even left/top and odd right/bottom,
        # which means the resized width and height must both be even.
        self.resized_width -= self.resized_width % 2
        self.resized_height -= self.resized_height % 2
        self.left = ((output_width - self.resized_width) // 2) // 2 * 2
        self.top = ((output_height - self.resized_height) // 2) // 2 * 2
        self.right = self.left + self.resized_width - 1
        self.bottom = self.top + self.resized_height - 1
        self.scale_x = self.resized_width / width
        self.scale_y = self.resized_height / height

        self.channel_desc = None
        self.stream = None
        self.input_desc = None
        self.output_desc = None
        self.crop_roi = None
        self.paste_roi = None
        self.resize_config = None
        self.input_dev = None
        self.output_dev = None
        self._create(preserve_current_context)

    def _create(self, preserve_current_context):
        _initialize_acl(self.device, set_device=not preserve_current_context)
        try:
            self.channel_desc = acl.media.dvpp_create_channel_desc()
            _require_success(
                acl.media.dvpp_create_channel(self.channel_desc),
                "dvpp_create_channel",
            )
            self.stream, ret = acl.rt.create_stream()
            _require_success(ret, "acl.rt.create_stream")

            self.input_dev, ret = acl.media.dvpp_malloc(self.input_size)
            _require_success(ret, "dvpp_malloc input")
            self.output_dev, ret = acl.media.dvpp_malloc(self.output_size)
            _require_success(ret, "dvpp_malloc output")

            self.input_desc = acl.media.dvpp_create_pic_desc()
            acl.media.dvpp_set_pic_desc_data(self.input_desc, self.input_dev)
            acl.media.dvpp_set_pic_desc_size(self.input_desc, self.input_size)
            acl.media.dvpp_set_pic_desc_format(
                self.input_desc, PIXEL_FORMAT_YUYV_PACKED_422
            )
            acl.media.dvpp_set_pic_desc_width(self.input_desc, self.width)
            acl.media.dvpp_set_pic_desc_height(self.input_desc, self.height)
            acl.media.dvpp_set_pic_desc_width_stride(
                self.input_desc, self.input_width_stride
            )
            acl.media.dvpp_set_pic_desc_height_stride(
                self.input_desc, self.input_height_stride
            )

            self.output_desc = acl.media.dvpp_create_pic_desc()
            acl.media.dvpp_set_pic_desc_data(self.output_desc, self.output_dev)
            acl.media.dvpp_set_pic_desc_size(self.output_desc, self.output_size)
            acl.media.dvpp_set_pic_desc_format(self.output_desc, PIXEL_FORMAT_RGB_888)
            acl.media.dvpp_set_pic_desc_width(self.output_desc, self.output_width)
            acl.media.dvpp_set_pic_desc_height(self.output_desc, self.output_height)
            acl.media.dvpp_set_pic_desc_width_stride(
                self.output_desc, self.output_width_stride
            )
            acl.media.dvpp_set_pic_desc_height_stride(
                self.output_desc, self.output_height_stride
            )

            self.crop_roi = acl.media.dvpp_create_roi_config(
                0, self.width - 1, 0, self.height - 1
            )
            self.paste_roi = acl.media.dvpp_create_roi_config(
                self.left, self.right, self.top, self.bottom
            )
            self.resize_config = acl.media.dvpp_create_resize_config()
        except Exception:
            self.close()
            raise

    def preprocess(self, yuyv_frame):
        """Run H2D and DVPP crop/resize/paste, returning the output pointer."""
        frame = np.ascontiguousarray(yuyv_frame, dtype=np.uint8)
        if frame.shape != (self.height, self.width, 2):
            raise ValueError(
                f"Unexpected YUYV shape {frame.shape}; "
                f"expected {(self.height, self.width, 2)}"
            )

        row_bytes = self.width * 2
        if self.input_width_stride == row_bytes:
            frame_bytes = frame.tobytes()
        else:
            padded = np.zeros(
                (self.input_height_stride, self.input_width_stride), dtype=np.uint8
            )
            padded[: self.height, :row_bytes] = frame.reshape(self.height, row_bytes)
            frame_bytes = padded.tobytes()

        _require_success(
            acl.rt.memcpy(
                self.input_dev,
                self.input_size,
                acl.util.bytes_to_ptr(frame_bytes),
                self.input_size,
                ACL_MEMCPY_HOST_TO_DEVICE,
            ),
            "YUYV host_to_device memcpy",
        )
        _require_success(
            acl.rt.memset(
                self.output_dev,
                self.output_size,
                114,
                self.output_size,
            ),
            "RGB letterbox memset",
        )
        _require_success(
            acl.media.dvpp_vpc_crop_resize_paste_async(
                self.channel_desc,
                self.input_desc,
                self.output_desc,
                self.crop_roi,
                self.paste_roi,
                self.resize_config,
                self.stream,
            ),
            "dvpp_vpc_crop_resize_paste_async",
        )
        _require_success(
            acl.rt.synchronize_stream(self.stream),
            "acl.rt.synchronize_stream",
        )
        _require_success(
            acl.media.dvpp_get_pic_desc_ret_code(self.output_desc),
            "VPC output ret_code",
        )
        return self.output_dev, self.output_size

    def copy_rgb_to_host(self):
        """Copy the prepared RGB image to the host for diagnostics only."""
        host_output = ctypes.create_string_buffer(self.output_size)
        _require_success(
            acl.rt.memcpy(
                ctypes.addressof(host_output),
                self.output_size,
                self.output_dev,
                self.output_size,
                ACL_MEMCPY_DEVICE_TO_HOST,
            ),
            "RGB device_to_host memcpy",
        )
        return np.frombuffer(host_output.raw, dtype=np.uint8).reshape(
            self.output_height_stride,
            self.output_width_stride,
        )[: self.output_height, : self.output_width * 3].reshape(
            self.output_height, self.output_width, 3
        ).copy()

    def close(self):
        if self.resize_config is not None:
            acl.media.dvpp_destroy_resize_config(self.resize_config)
            self.resize_config = None
        if self.paste_roi is not None:
            acl.media.dvpp_destroy_roi_config(self.paste_roi)
            self.paste_roi = None
        if self.crop_roi is not None:
            acl.media.dvpp_destroy_roi_config(self.crop_roi)
            self.crop_roi = None
        if self.output_desc is not None:
            acl.media.dvpp_destroy_pic_desc(self.output_desc)
            self.output_desc = None
        if self.input_desc is not None:
            acl.media.dvpp_destroy_pic_desc(self.input_desc)
            self.input_desc = None
        if self.output_dev is not None:
            acl.media.dvpp_free(self.output_dev)
            self.output_dev = None
        if self.input_dev is not None:
            acl.media.dvpp_free(self.input_dev)
            self.input_dev = None
        if self.stream is not None:
            acl.rt.destroy_stream(self.stream)
            self.stream = None
        if self.channel_desc is not None:
            acl.media.dvpp_destroy_channel(self.channel_desc)
            acl.media.dvpp_destroy_channel_desc(self.channel_desc)
            self.channel_desc = None
