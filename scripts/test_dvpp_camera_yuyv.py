"""DVPP 摄像头格式测试工具：抓取一帧 YUYV 并转换为 BGR。

用于排查 V4L2 格式、widthStride、颜色转换和 D2H 输出问题，
不参与当前设备直连网站的正常运行。
"""

import argparse
import ctypes
import json
import os
import subprocess

import acl
import cv2
import numpy as np


PIXEL_FORMAT_YUYV_PACKED_422 = 7
PIXEL_FORMAT_BGR_888 = 13
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


def require_success(ret, operation):
    if ret != 0:
        raise RuntimeError(f"{operation} failed: {ret}")


def capture_yuyv(device, width, height, fps, raw_path):
    command = [
        "v4l2-ctl",
        f"--device={device}",
        f"--set-fmt-video=width={width},height={height},pixelformat=YUYV",
        f"--set-parm={fps}",
        "--stream-mmap=4",
        "--stream-skip=5",
        "--stream-count=1",
        f"--stream-to={raw_path}",
    ]
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--npu-device", type=int, default=0)
    parser.add_argument("--output-dir", default="runs/dvpp_camera_yuyv")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    raw_path = os.path.join(args.output_dir, "camera_yuyv422.raw")
    cpu_path = os.path.join(args.output_dir, "camera_cpu_bgr.jpg")
    dvpp_path = os.path.join(args.output_dir, "camera_dvpp_bgr.jpg")
    info_path = os.path.join(args.output_dir, "camera_info.json")

    print(f"[1/8] 配置摄像头 YUYV: {args.width}x{args.height} @ {args.fps}fps")
    capture_yuyv(args.device, args.width, args.height, args.fps, raw_path)

    raw = np.fromfile(raw_path, dtype=np.uint8)
    expected_size = args.width * args.height * 2
    if raw.size != expected_size:
        raise RuntimeError(
            f"YUYV 帧大小不正确: actual={raw.size}, expected={expected_size}. "
            "请检查 V4L2 bytesperline。"
        )
    print(f"[2/8] 捕获原始 YUYV 帧: {raw_path} ({raw.size} bytes)")

    yuyv = raw.reshape(args.height, args.width, 2)
    cpu_bgr = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
    cv2.imwrite(cpu_path, cpu_bgr)
    print(f"[3/8] CPU YUYV 转 BGR: {cpu_path}")

    channel_desc = None
    stream = None
    input_desc = None
    output_desc = None
    input_dev = None
    output_dev = None
    acl_initialized = False
    device_set = False

    try:
        require_success(acl.init(), "acl.init")
        acl_initialized = True
        require_success(acl.rt.set_device(args.npu_device), "acl.rt.set_device")
        device_set = True

        channel_desc = acl.media.dvpp_create_channel_desc()
        require_success(acl.media.dvpp_create_channel(channel_desc), "dvpp_create_channel")
        stream, ret = acl.rt.create_stream()
        require_success(ret, "acl.rt.create_stream")

        input_size = expected_size
        raw_bytes = raw.tobytes()
        input_dev, ret = acl.media.dvpp_malloc(input_size)
        require_success(ret, "dvpp_malloc input")
        require_success(
            acl.rt.memcpy(
                input_dev,
                input_size,
                acl.util.bytes_to_ptr(raw_bytes),
                input_size,
                ACL_MEMCPY_HOST_TO_DEVICE,
            ),
            "YUYV host_to_device memcpy",
        )
        print("[4/8] YUYV 帧拷贝到 DVPP 内存")

        input_desc = acl.media.dvpp_create_pic_desc()
        acl.media.dvpp_set_pic_desc_data(input_desc, input_dev)
        acl.media.dvpp_set_pic_desc_size(input_desc, input_size)
        acl.media.dvpp_set_pic_desc_format(input_desc, PIXEL_FORMAT_YUYV_PACKED_422)
        acl.media.dvpp_set_pic_desc_width(input_desc, args.width)
        acl.media.dvpp_set_pic_desc_height(input_desc, args.height)
        input_width_stride = (args.width * 2 + 15) // 16 * 16
        acl.media.dvpp_set_pic_desc_width_stride(input_desc, input_width_stride)
        acl.media.dvpp_set_pic_desc_height_stride(input_desc, args.height)

        output_width_stride = (args.width * 3 + 15) // 16 * 16
        output_height_stride = (args.height + 1) // 2 * 2
        output_size = output_width_stride * output_height_stride
        output_dev, ret = acl.media.dvpp_malloc(output_size)
        require_success(ret, "dvpp_malloc output")

        output_desc = acl.media.dvpp_create_pic_desc()
        acl.media.dvpp_set_pic_desc_data(output_desc, output_dev)
        acl.media.dvpp_set_pic_desc_size(output_desc, output_size)
        acl.media.dvpp_set_pic_desc_format(output_desc, PIXEL_FORMAT_BGR_888)
        # CANN 8.5 要求 VPC ConvertColor 的输出宽高和 stride 初始为 0，由接口回填。
        acl.media.dvpp_set_pic_desc_width(output_desc, 0)
        acl.media.dvpp_set_pic_desc_height(output_desc, 0)
        acl.media.dvpp_set_pic_desc_width_stride(output_desc, 0)
        acl.media.dvpp_set_pic_desc_height_stride(output_desc, 0)

        print("[5/8] 提交 DVPP VPC: YUYV422 -> BGR888")
        ret = acl.media.dvpp_vpc_convert_color_async(
            channel_desc, input_desc, output_desc, stream
        )
        require_success(ret, "dvpp_vpc_convert_color_async")
        require_success(acl.rt.synchronize_stream(stream), "acl.rt.synchronize_stream")
        require_success(
            acl.media.dvpp_get_pic_desc_ret_code(output_desc),
            "VPC output ret_code",
        )

        actual_format = acl.media.dvpp_get_pic_desc_format(output_desc)
        actual_width = acl.media.dvpp_get_pic_desc_width(output_desc)
        actual_height = acl.media.dvpp_get_pic_desc_height(output_desc)
        actual_width_stride = acl.media.dvpp_get_pic_desc_width_stride(output_desc)
        actual_height_stride = acl.media.dvpp_get_pic_desc_height_stride(output_desc)
        actual_size = acl.media.dvpp_get_pic_desc_size(output_desc)
        actual_data = acl.media.dvpp_get_pic_desc_data(output_desc)
        print(
            "[6/8] VPC 输出:",
            f"format={actual_format}",
            f"shape={actual_width}x{actual_height}",
            f"stride={actual_width_stride}x{actual_height_stride}",
            f"size={actual_size}",
            f"data_match={actual_data == output_dev}",
        )

        if actual_width_stride <= 0:
            actual_width_stride = output_width_stride
        if actual_height_stride <= 0:
            actual_height_stride = output_height_stride
        copy_size = actual_width_stride * actual_height_stride
        if copy_size > output_size:
            raise RuntimeError(
                f"VPC 输出超过已分配缓冲区: actual={copy_size}, allocated={output_size}"
            )

        host_output = ctypes.create_string_buffer(copy_size)
        require_success(
            acl.rt.memcpy(
                ctypes.addressof(host_output),
                copy_size,
                actual_data,
                copy_size,
                ACL_MEMCPY_DEVICE_TO_HOST,
            ),
            "BGR device_to_host memcpy",
        )
        dvpp_bgr = np.frombuffer(host_output.raw, dtype=np.uint8).reshape(
            actual_height_stride, actual_width_stride
        )[: args.height, : args.width * 3].reshape(args.height, args.width, 3)
        cv2.imwrite(dvpp_path, dvpp_bgr)
        print(f"[7/8] DVPP BGR 输出: {dvpp_path}")

        diff = np.abs(dvpp_bgr.astype(np.int16) - cpu_bgr.astype(np.int16))
        mean_abs_diff = float(diff.mean())
        max_abs_diff = int(diff.max())
        same_pixel_ratio = float(np.all(diff == 0, axis=2).mean())
        print("[8/8] 与 CPU YUYV 转换结果比较")
        print("mean_abs_diff:", mean_abs_diff)
        print("max_abs_diff:", max_abs_diff)
        print("same_pixel_ratio:", same_pixel_ratio)

        info = {
            "device": args.device,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "input_format": "YUYV 4:2:2",
            "input_format_value": PIXEL_FORMAT_YUYV_PACKED_422,
            "output_format": "BGR888",
            "output_format_value": actual_format,
            "output_width_stride": actual_width_stride,
            "output_height_stride": actual_height_stride,
            "mean_abs_diff": mean_abs_diff,
            "max_abs_diff": max_abs_diff,
            "same_pixel_ratio": same_pixel_ratio,
            "raw_file": raw_path,
            "cpu_bgr_file": cpu_path,
            "dvpp_bgr_file": dvpp_path,
        }
        with open(info_path, "w", encoding="utf-8") as file:
            json.dump(info, file, ensure_ascii=False, indent=2)
        print("测试信息:", info_path)
    finally:
        if output_desc is not None:
            acl.media.dvpp_destroy_pic_desc(output_desc)
        if input_desc is not None:
            acl.media.dvpp_destroy_pic_desc(input_desc)
        if output_dev is not None:
            acl.media.dvpp_free(output_dev)
        if input_dev is not None:
            acl.media.dvpp_free(input_dev)
        if stream is not None:
            acl.rt.destroy_stream(stream)
        if channel_desc is not None:
            acl.media.dvpp_destroy_channel(channel_desc)
            acl.media.dvpp_destroy_channel_desc(channel_desc)
        if device_set:
            acl.rt.reset_device(args.npu_device)
        if acl_initialized:
            acl.finalize()


if __name__ == "__main__":
    main()
