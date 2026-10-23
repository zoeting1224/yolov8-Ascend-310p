#!/usr/bin/env python3
"""正式入口：USB 摄像头实时检测与离线视频检测网站。

实时推理：YUYV -> H2D -> DVPP RGB letterbox -> AIPP OM -> NPU。
实时显示：Host YUYV -> BGR -> 画框/JPEG -> 双路 MJPEG。
离线检测：上传视频 -> Host 解码/letterbox/RGB -> AIPP OM -> NPU -> WebM。
"""

import argparse
import mimetypes
import queue
import re
import signal
import threading
import time
import uuid
from datetime import datetime
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import aclruntime
import numpy as np
import torch
import torchvision
from ais_bench.infer.interface import InferSession

import dashboard
from dvpp_yuyv import DvppYuyvLetterbox
from infer_om import COCO_NAMES, draw_detections


TIMING_CARDS = '<div class="cards" style="margin-top:13px"><div class="card"><span>摄像头采集</span><strong id="captureTime">--</strong></div><div class="card"><span>DVPP 前处理</span><strong id="preprocessTime">--</strong></div><div class="card"><span>CPU 后处理</span><strong id="postprocessTime">--</strong></div><div class="card"><span>画框</span><strong id="drawTime">--</strong></div><div class="card"><span>JPEG 编码</span><strong id="jpegTime">--</strong></div><div class="card"><span>端到端延迟</span><strong id="endToEnd">--</strong></div><div class="card"><span>流水线 FPS</span><strong id="pipelineFps">--</strong></div></div>'
PIPELINE_VISUAL = r'''<style>
.pipeline-grid{display:grid;grid-template-columns:minmax(210px,1fr) 44px minmax(390px,1.65fr) 44px minmax(230px,1fr);gap:12px;align-items:stretch}
.thread-card{position:relative;border:1px solid var(--border);background:#0a1728;border-radius:15px;padding:17px;transition:.2s}
.thread-card.bottleneck{border-color:var(--orange);box-shadow:0 0 0 1px var(--orange),0 0 24px rgba(255,189,89,.16)}
.thread-card.bottleneck:after{content:"当前瓶颈";position:absolute;right:12px;top:11px;color:#07111f;background:var(--orange);border-radius:999px;padding:3px 8px;font-size:11px;font-weight:800}
.thread-head{display:flex;align-items:center;gap:10px;margin-bottom:14px}.thread-number{display:grid;place-items:center;width:30px;height:30px;border-radius:9px;background:linear-gradient(135deg,#138c78,#2276bc);font-weight:800}.thread-title{font-size:17px;font-weight:800}.thread-time{font-size:29px;font-weight:850;margin:8px 0 4px}.thread-code{color:var(--green);font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px}.thread-note{color:var(--muted);font-size:12px;margin-top:9px;line-height:1.6}
.flow-arrow{display:grid;place-items:center;color:var(--blue);font-size:30px;font-weight:800}.infer-stages{display:grid;grid-template-columns:repeat(3,1fr);gap:9px;margin-top:13px}.infer-stage,.encode-stage{background:#12243b;border:1px solid #274562;border-radius:11px;padding:12px}.infer-stage span,.encode-stage span{display:block;color:var(--muted);font-size:12px}.infer-stage strong,.encode-stage strong{display:block;font-size:19px;margin:7px 0}.infer-stage code,.encode-stage code{font-size:10px;color:var(--green);white-space:normal}.encode-stages{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-top:13px}
.pipeline-summary{display:grid;grid-template-columns:1.3fr 1fr 1fr;gap:12px;margin-top:14px}.pipeline-summary>div{background:#0a1728;border:1px solid var(--border);border-radius:12px;padding:14px}.pipeline-summary span{display:block;color:var(--muted);font-size:12px}.pipeline-summary strong{display:block;font-size:20px;margin-top:7px}.pipeline-formula{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;color:var(--green);font-size:13px;margin-top:7px}.pipeline-help{color:var(--muted);line-height:1.65;margin:13px 0 0;font-size:13px}
@media(max-width:1100px){.pipeline-grid{grid-template-columns:1fr}.flow-arrow{transform:rotate(90deg);height:30px}.infer-stages{grid-template-columns:1fr}.pipeline-summary{grid-template-columns:1fr}}
</style><section class="section"><h2>三级并行流水线</h2><div class="muted" style="margin:-8px 0 17px">三个线程并行处理不同帧；线程 2 内部的三个阶段按顺序执行。</div><div class="pipeline-grid">
<div class="thread-card" id="captureThreadCard"><div class="thread-head"><span class="thread-number">1</span><span class="thread-title">摄像头采集线程</span></div><div class="thread-time" id="pipelineCaptureThread">--</div><div class="thread-code">capture.read() + cv2.cvtColor()</div><div class="thread-note">从 V4L2 等待并取得一帧 YUYV，同时生成网页显示使用的 BGR 帧，再把最新帧送入推理队列。</div></div>
<div class="flow-arrow">→</div>
<div class="thread-card" id="inferenceThreadCard"><div class="thread-head"><span class="thread-number">2</span><span class="thread-title">YOLO 推理线程</span></div><div class="thread-time" id="pipelineInferenceThread">--</div><div class="thread-code">前处理 → NPU 推理 → 后处理</div><div class="infer-stages"><div class="infer-stage"><span>① DVPP 前处理</span><strong id="pipelinePreprocess">--</strong><code>preprocessor.preprocess()</code></div><div class="infer-stage"><span>② NPU 模型推理</span><strong id="pipelineNpu">--</strong><code>session.run([BaseTensor])</code></div><div class="infer-stage"><span>③ CPU 后处理</span><strong id="pipelinePostprocess">--</strong><code>postprocess_timed()</code></div></div><div class="thread-note">这三个阶段属于同一个线程，必须依次完成，线程耗时为三者相加。</div></div>
<div class="flow-arrow">→</div>
<div class="thread-card" id="encodeThreadCard"><div class="thread-head"><span class="thread-number">3</span><span class="thread-title">显示编码线程</span></div><div class="thread-time" id="pipelineEncodeThread">--</div><div class="thread-code">draw_detections() + cv2.imencode()</div><div class="encode-stages"><div class="encode-stage"><span>CPU 画框</span><strong id="pipelineDraw">--</strong><code>draw_detections()</code></div><div class="encode-stage"><span>CPU JPEG 编码</span><strong id="pipelineJpeg">--</strong><code>cv2.imencode()</code></div></div><div class="thread-note">生成原始流和检测结果流，供浏览器通过 HTTP MJPEG 观看。</div></div>
</div><div class="pipeline-summary"><div><span>流水线瓶颈 = 三个线程中的最大耗时</span><strong id="pipelineBottleneck">--</strong><div class="pipeline-formula" id="pipelineFormula">max(--, --, --)</div></div><div><span>实际流水线吞吐量</span><strong id="pipelineSummaryFps">--</strong></div><div><span>单帧端到端延迟</span><strong id="pipelineSummaryLatency">--</strong></div></div><p class="pipeline-help">吞吐量看最慢线程：稳定运行时，线程 1 可以采集第 3 帧、线程 2 推理第 2 帧、线程 3 编码第 1 帧。端到端延迟则是一帧从采集开始到编码完成的实际总等待时间。</p></section>'''
OFFLINE_STYLE = r'''
.mode-switch{display:flex;gap:10px;margin:20px 0}.mode-switch button{border:1px solid var(--border);background:var(--panel2);color:var(--text);padding:12px 22px;border-radius:11px;font-weight:800;cursor:pointer}.mode-switch button.active{background:linear-gradient(90deg,#138c78,#2276bc);border-color:#52a8ff}.offline-upload{display:grid;grid-template-columns:minmax(250px,1fr) auto;gap:12px;align-items:center}.offline-upload input{width:100%;background:#07111f;border:1px solid var(--border);color:var(--text);border-radius:10px;padding:11px}.offline-upload button,.offline-download{border:0;background:linear-gradient(90deg,#138c78,#2276bc);color:white;padding:12px 18px;border-radius:10px;font-weight:800;cursor:pointer;text-decoration:none;text-align:center}.offline-upload button:disabled{opacity:.45;cursor:not-allowed}.offline-note{margin-top:13px;padding:12px 14px;border:1px solid #355574;background:#0a1728;border-radius:10px;color:var(--muted);line-height:1.7}.offline-progress{height:18px;margin:18px 0 9px;background:#07111f;border:1px solid var(--border);border-radius:999px;overflow:hidden}.offline-progress>div{height:100%;width:0;background:linear-gradient(90deg,var(--green),var(--blue));transition:width .25s}.offline-status-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-top:13px}.offline-status-grid>div{background:#0a1728;border:1px solid var(--border);border-radius:11px;padding:13px}.offline-status-grid span{display:block;color:var(--muted);font-size:12px}.offline-status-grid strong{display:block;margin-top:7px;font-size:18px}.offline-media{display:grid;grid-template-columns:1fr 1fr;gap:17px;margin-top:18px}.offline-media>div{background:#050c15;border:1px solid var(--border);border-radius:13px;padding:12px}.offline-media h3{margin:0 0 10px}.offline-media img,.offline-media video{width:100%;height:430px;display:block;object-fit:contain;background:#02060b;border-radius:9px}.offline-actions{display:flex;gap:10px;align-items:center;margin-top:14px}.offline-error{color:#ffc3cb;margin-top:11px;display:none}.offline-file{word-break:break-all;color:var(--muted);font-size:13px;margin-top:10px}@media(max-width:1000px){.offline-upload{grid-template-columns:1fr}.offline-status-grid{grid-template-columns:repeat(2,1fr)}.offline-media{grid-template-columns:1fr}.offline-media img,.offline-media video{height:auto}}
'''
MODE_SWITCH_HTML = r'''<div class="mode-switch"><button id="realtimeModeButton" onclick="switchDashboardMode('realtime')">实时检测</button><button id="offlineModeButton" class="active" onclick="switchDashboardMode('offline')">离线视频检测</button></div><div id="realtimePanel" style="display:none">'''
OFFLINE_HTML = r'''</div><div id="offlinePanel"><section class="section"><h2>离线视频检测</h2><div class="offline-upload"><input id="offlineFile" type="file" accept="video/*,.mp4,.avi,.mov,.mkv,.webm,.m4v,.ts,.mpeg,.mpg"><button id="offlineUploadButton" onclick="uploadOfflineVideo()">上传并开始检测</button></div><div class="offline-note">上传视频后，由 OpenCV 在 Host 端逐帧解码并准备 RGB888 输入，再交给 AIPP OM 和 NPU 完成检测。离线模式不需要连接摄像头。</div><div class="offline-file" id="offlineFilename">尚未选择视频</div><div class="offline-progress"><div id="offlineProgressBar"></div></div><div class="muted" id="offlineStatusText">等待上传视频</div><div class="offline-error" id="offlineError"></div><div class="offline-status-grid"><div><span>任务状态</span><strong id="offlineState">空闲</strong></div><div><span>处理进度</span><strong id="offlineProgress">0.0%</strong></div><div><span>已处理帧</span><strong id="offlineFrames">0 / 0</strong></div><div><span>离线处理 FPS</span><strong id="offlineFps">0.00</strong></div><div><span>已用时间</span><strong id="offlineElapsed">0.00 s</strong></div></div><div class="offline-media"><div><h3>最新检测帧</h3><img id="offlinePreview" alt="离线检测预览" style="display:none"><div id="offlinePreviewEmpty" class="muted">任务开始后显示最新画框结果。</div></div><div><h3>检测结果视频</h3><video id="offlineResultVideo" controls preload="metadata" style="display:none"></video><div id="offlineResultEmpty" class="muted">任务完成后可在线播放结果视频。</div></div></div><div class="offline-actions"><a id="offlineDownload" class="offline-download" href="#" style="display:none">下载画框结果</a><span class="muted" id="offlineResolution"></span></div></section></div>'''
OFFLINE_SCRIPT = r'''
var offlineBusy=false,offlinePreviewSequence=-1,offlineResultJob='';
function switchDashboardMode(mode){const realtime=mode==='realtime';document.getElementById('realtimePanel').style.display=realtime?'block':'none';document.getElementById('offlinePanel').style.display=realtime?'none':'block';document.getElementById('realtimeModeButton').classList.toggle('active',realtime);document.getElementById('offlineModeButton').classList.toggle('active',!realtime);const input=document.getElementById('inputImage'),output=document.getElementById('outputImage');if(realtime){if(!input.getAttribute('src'))input.src='/stream/input.mjpg?t='+Date.now();if(!output.getAttribute('src'))output.src='/stream/output.mjpg?t='+Date.now()}else{input.removeAttribute('src');output.removeAttribute('src')}}
async function uploadOfflineVideo(){const input=document.getElementById('offlineFile'),file=input.files&&input.files[0];if(!file){alert('请先选择一个视频文件');return}const button=document.getElementById('offlineUploadButton');button.disabled=true;document.getElementById('offlineStatusText').textContent='正在上传 '+file.name+' ...';document.getElementById('offlineError').style.display='none';try{const response=await fetch('/api/offline/upload?filename='+encodeURIComponent(file.name),{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:file});const data=await response.json();if(!response.ok||!data.accepted)throw new Error(data.error||'上传失败');offlineResultJob='';offlinePreviewSequence=-1;await loadOfflineStatus()}catch(error){const box=document.getElementById('offlineError');box.textContent=error.message;box.style.display='block';button.disabled=false}}
function renderOffline(o){const labels={idle:'空闲',uploading:'上传中',queued:'等待推理',running:'检测中',completed:'已完成',error:'失败'},state=o.state||'idle',busy=['uploading','queued','running'].includes(state),progress=Math.max(0,Math.min(100,Number(o.progress||0)));offlineBusy=busy;document.getElementById('offlineState').textContent=labels[state]||state;document.getElementById('offlineProgress').textContent=progress.toFixed(1)+'%';document.getElementById('offlineProgressBar').style.width=progress+'%';document.getElementById('offlineFrames').textContent=(o.processed_frames||0)+' / '+(o.total_frames||0);document.getElementById('offlineFps').textContent=Number(o.processing_fps||0).toFixed(2);document.getElementById('offlineElapsed').textContent=(Number(o.elapsed_ms||0)/1000).toFixed(2)+' s';document.getElementById('offlineFilename').textContent=o.filename?'当前文件：'+o.filename:'尚未选择视频';document.getElementById('offlineResolution').textContent=o.input_resolution?('输入 '+o.input_resolution+' ｜ 输出 '+(o.output_resolution||o.input_resolution)+' ｜ '+Number(o.source_fps||0).toFixed(2)+' FPS'):'';const statusText={idle:'等待上传视频',uploading:'视频正在上传',queued:'视频已上传，等待现有推理线程接管',running:'正在逐帧检测；实时检测会在任务完成后自动恢复',completed:'检测完成，可以播放或下载画框结果',error:'离线检测失败'};document.getElementById('offlineStatusText').textContent=statusText[state]||state;const error=document.getElementById('offlineError');error.textContent=o.error||'';error.style.display=o.error?'block':'none';document.getElementById('offlineUploadButton').disabled=busy;document.getElementById('offlineFile').disabled=busy;document.getElementById('inferButton').disabled=busy||document.getElementById('inferButton').disabled;document.getElementById('profileButton').disabled=busy||document.getElementById('profileButton').disabled;const preview=document.getElementById('offlinePreview'),previewEmpty=document.getElementById('offlinePreviewEmpty');if(Number(o.preview_sequence||0)>0){if(offlinePreviewSequence!==o.preview_sequence){preview.src='/api/offline/preview?job_id='+encodeURIComponent(o.job_id||'')+'&v='+o.preview_sequence;offlinePreviewSequence=o.preview_sequence}preview.style.display='block';previewEmpty.style.display='none'}else{preview.style.display='none';previewEmpty.style.display='block'}const video=document.getElementById('offlineResultVideo'),empty=document.getElementById('offlineResultEmpty'),download=document.getElementById('offlineDownload');if(state==='completed'&&o.job_id){if(offlineResultJob!==o.job_id){video.src='/api/offline/result?job_id='+encodeURIComponent(o.job_id);video.load();offlineResultJob=o.job_id}video.style.display='block';empty.style.display='none';download.href='/api/offline/result?download=1&job_id='+encodeURIComponent(o.job_id);download.style.display='inline-block'}else{video.style.display='none';empty.style.display='block';download.style.display='none'}}
async function loadOfflineStatus(){try{const response=await fetch('/api/offline/status',{cache:'no-store'});if(response.ok)renderOffline(await response.json())}catch(error){}}
loadOfflineStatus();setInterval(loadOfflineStatus,1000);
'''
OLD_IMAGE_BLOCK = '<div class="images"><div class="image-box"><h3>输入</h3><img id="inputImage" alt="输入图片"></div><div class="image-box"><h3>检测结果</h3><img id="outputImage" alt="检测结果"></div></div>'
NEW_IMAGE_BLOCK = '<div class="images"><div class="image-box"><h3>DVPP 原始视频流</h3><img id="inputImage" src="/stream/input.mjpg" alt="DVPP 原始视频流"></div><div class="image-box"><h3>YOLO 实时检测结果</h3><img id="outputImage" src="/stream/output.mjpg" alt="YOLO 实时检测结果"></div></div>'

CAMERA_HTML = dashboard.DASHBOARD_HTML.replace(
    "页面每 2 秒刷新数据；普通推理和完整 Profiling 分开执行。",
    "采集、推理、编码三线程流水线；统计每 2 秒刷新，完整 Profiling 按需执行。",
).replace(
    '</div><section class="section"><h2>最新检测画面</h2>',
    f'</div>{TIMING_CARDS}{PIPELINE_VISUAL}<section class="section"><h2>USB 摄像头实时检测画面</h2>',
).replace(
    OLD_IMAGE_BLOCK,
    NEW_IMAGE_BLOCK,
).replace(
    "const stamp=Date.now();document.getElementById('inputImage').src='/api/image/input?t='+stamp;document.getElementById('outputImage').src='/api/image/output?t='+stamp;",
    "if(!data.camera_enabled){const stamp=Date.now();document.getElementById('outputImage').src='/api/image/output?t='+stamp;}",
).replace(
    "document.getElementById('device').textContent='Device '+(m.device??0);document.getElementById('updatedAt').textContent='页面刷新：'+data.updated_at;",
    "document.getElementById('device').textContent='Device '+(m.device??0);const capture=Number(m.capture_thread_time_ms||m.capture_read_ms||0),pre=Number(m.preprocess_time_ms||0),npu=Number(m.inference_time_ms||0),post=Number(m.postprocess_time_ms||0),draw=Number(m.draw_time_ms||0),jpeg=Number(m.jpeg_encode_time_ms||0),inferThread=Number(m.inference_thread_time_ms||pre+npu+post),encodeThread=Number(m.encode_thread_time_ms||draw+jpeg),critical=Math.max(capture,inferThread,encodeThread),ms=v=>fmt(v)+' ms';document.getElementById('captureTime').textContent=ms(capture);document.getElementById('preprocessTime').textContent=ms(pre);document.getElementById('postprocessTime').textContent=ms(post);document.getElementById('drawTime').textContent=ms(draw);document.getElementById('jpegTime').textContent=ms(jpeg);document.getElementById('endToEnd').textContent=ms(m.end_to_end_time_ms);document.getElementById('pipelineFps').textContent=fmt(m.pipeline_fps);document.getElementById('pipelineCaptureThread').textContent=ms(capture);document.getElementById('pipelineInferenceThread').textContent=ms(inferThread);document.getElementById('pipelineEncodeThread').textContent=ms(encodeThread);document.getElementById('pipelinePreprocess').textContent=ms(pre);document.getElementById('pipelineNpu').textContent=ms(npu);document.getElementById('pipelinePostprocess').textContent=ms(post);document.getElementById('pipelineDraw').textContent=ms(draw);document.getElementById('pipelineJpeg').textContent=ms(jpeg);const threadRows=[['captureThreadCard','采集线程',capture],['inferenceThreadCard','推理线程',inferThread],['encodeThreadCard','显示编码线程',encodeThread]],bottleneck=threadRows.reduce((a,b)=>b[2]>a[2]?b:a,threadRows[0]);threadRows.forEach(r=>document.getElementById(r[0]).classList.toggle('bottleneck',r[0]===bottleneck[0]));document.getElementById('pipelineBottleneck').textContent=bottleneck[1]+' · '+ms(bottleneck[2]);document.getElementById('pipelineFormula').textContent='max('+fmt(capture)+', '+fmt(inferThread)+', '+fmt(encodeThread)+') = '+fmt(critical)+' ms';document.getElementById('pipelineSummaryFps').textContent=fmt(m.pipeline_fps)+' FPS';document.getElementById('pipelineSummaryLatency').textContent=ms(m.end_to_end_time_ms);document.getElementById('updatedAt').textContent='页面刷新：'+data.updated_at+'｜观看人数：'+(m.viewer_count??0)+'｜采集丢帧：'+(m.capture_dropped_frames??0)+'｜结果丢帧：'+(m.result_dropped_frames??0);",
)
CAMERA_HTML = CAMERA_HTML.replace("</style>", OFFLINE_STYLE + "</style>", 1)
CAMERA_HTML = CAMERA_HTML.replace('<div class="actions">', MODE_SWITCH_HTML + '<div class="actions">', 1)
CAMERA_HTML = CAMERA_HTML.replace("</main><script>", OFFLINE_HTML + "</main><script>", 1)
CAMERA_HTML = CAMERA_HTML.replace("</script></body>", OFFLINE_SCRIPT + "</script></body>", 1)
CAMERA_HTML = CAMERA_HTML.replace(
    "document.getElementById('inferButton').disabled=!!s.running;document.getElementById('profileButton').disabled=!!s.running;",
    "document.getElementById('inferButton').disabled=!!s.running||offlineBusy;document.getElementById('profileButton').disabled=!!s.running||offlineBusy;",
)


def put_latest(target_queue, item):
    """Put one item into a size-one queue, replacing a stale item if necessary."""
    dropped = 0
    while True:
        try:
            target_queue.put_nowait(item)
            return dropped
        except queue.Full:
            try:
                target_queue.get_nowait()
                dropped += 1
            except queue.Empty:
                continue


def postprocess_timed(prediction, scale, left, top, image_shape, conf_threshold, iou_threshold):
    """Run CPU YOLO postprocessing and return detailed stage timings."""
    timings = {"decode_filter_time_ms": 0.0, "nms_time_ms": 0.0, "restore_boxes_time_ms": 0.0}
    decode_start = time.perf_counter()
    prediction = np.asarray(prediction, dtype=np.float32).squeeze(0).T
    boxes_xywh = prediction[:, :4]
    class_scores = prediction[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
    keep = scores >= conf_threshold
    boxes_xywh, scores, class_ids = boxes_xywh[keep], scores[keep], class_ids[keep]
    if not len(scores):
        timings["decode_filter_time_ms"] = (time.perf_counter() - decode_start) * 1000
        return np.empty((0, 4)), scores, class_ids, timings
    boxes = np.empty_like(boxes_xywh)
    boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    timings["decode_filter_time_ms"] = (time.perf_counter() - decode_start) * 1000

    nms_start = time.perf_counter()
    boxes_tensor = torch.from_numpy(boxes)
    scores_tensor = torch.from_numpy(scores)
    offsets = torch.from_numpy(class_ids.astype(np.float32))[:, None] * 7680.0
    selected = torchvision.ops.nms(boxes_tensor + offsets, scores_tensor, iou_threshold).numpy()
    boxes, scores, class_ids = boxes[selected], scores[selected], class_ids[selected]
    timings["nms_time_ms"] = (time.perf_counter() - nms_start) * 1000

    restore_start = time.perf_counter()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / scale
    height, width = image_shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height - 1)
    timings["restore_boxes_time_ms"] = (time.perf_counter() - restore_start) * 1000
    return boxes, scores, class_ids, timings


class CameraWorker:
    def __init__(self, state, args):
        self.state = state
        self.args = args
        self.capture_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.offline_queue = queue.Queue(maxsize=1)
        self.condition = threading.Condition()
        self.input_sequence = 0
        self.input_jpeg = None
        self.sequence = 0
        self.output_jpeg = None
        self.metrics = {}
        self.error = ""
        self.started_at = ""
        self.viewer_count = 0
        self.capture_dropped_frames = 0
        self.result_dropped_frames = 0
        self.capture_frame_id = 0
        self.warmup_completed = False
        self.model_load_time_ms = 0.0
        self.offline_root = dashboard.absolute("runs/offline")
        self.offline_upload_dir = self.offline_root / "uploads"
        self.offline_result_dir = self.offline_root / "results"
        self.offline_upload_dir.mkdir(parents=True, exist_ok=True)
        self.offline_result_dir.mkdir(parents=True, exist_ok=True)
        self.offline_preview_jpeg = None
        self.offline_status = self._new_offline_status()
        self.stop_event = threading.Event()
        self.threads = []

    @staticmethod
    def _new_offline_status():
        return {
            "state": "idle",
            "job_id": "",
            "filename": "",
            "processed_frames": 0,
            "total_frames": 0,
            "progress": 0.0,
            "processing_fps": 0.0,
            "elapsed_ms": 0.0,
            "source_fps": 0.0,
            "input_resolution": "",
            "output_resolution": "",
            "preview_sequence": 0,
            "started_at": "",
            "finished_at": "",
            "error": "",
            "output_bytes": 0,
        }

    def start(self):
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.threads = [threading.Thread(target=self._inference_loop, name="camera-inference")]
        if not self.args.offline_only:
            self.threads.extend([
                threading.Thread(target=self._capture_loop, name="camera-capture"),
                threading.Thread(target=self._encode_loop, name="camera-encode"),
            ])
        for thread in self.threads:
            thread.start()

    def stop(self):
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        for thread in self.threads:
            thread.join(timeout=10.0)
        alive = [thread.name for thread in self.threads if thread.is_alive()]
        if alive:
            raise RuntimeError(f"工作线程未能停止: {', '.join(alive)}")

    def add_viewer(self):
        with self.condition:
            self.viewer_count += 1

    def remove_viewer(self):
        with self.condition:
            self.viewer_count = max(0, self.viewer_count - 1)

    def snapshot(self):
        with self.condition:
            return {
                "sequence": self.sequence,
                "input_sequence": self.input_sequence,
                "error": self.error,
                "started_at": self.started_at,
                "metrics": dict(self.metrics),
                "viewer_count": self.viewer_count,
                "capture_dropped_frames": self.capture_dropped_frames,
                "result_dropped_frames": self.result_dropped_frames,
                "warmup_completed": self.warmup_completed,
            }

    def offline_snapshot(self):
        with self.condition:
            return dict(self.offline_status)

    def offline_busy(self):
        with self.condition:
            return self.offline_status["state"] in {"uploading", "queued", "running"}

    def begin_offline_upload(self, filename, content_length):
        suffix = Path(filename).suffix.lower()
        allowed_suffixes = {
            ".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v",
            ".ts", ".mpeg", ".mpg",
        }
        if suffix not in allowed_suffixes:
            raise ValueError("不支持的视频格式，请上传 MP4、AVI、MOV、MKV 或 WebM")
        safe_stem = re.sub(r"[^0-9A-Za-z._-]+", "_", Path(filename).stem).strip("._")
        safe_stem = safe_stem[:80] or "video"
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        input_path = self.offline_upload_dir / f"{job_id}_{safe_stem}{suffix}"
        output_path = self.offline_result_dir / f"{job_id}_detected.webm"
        if self.state.service_status()["running"]:
            raise RuntimeError("普通推理或性能分析正在运行，请等待任务完成")
        with self.condition:
            if self.offline_status["state"] in {"uploading", "queued", "running"}:
                raise RuntimeError("已有离线视频正在上传或检测，请等待任务完成")
            self.offline_preview_jpeg = None
            self.offline_status = self._new_offline_status()
            self.offline_status.update({
                "state": "uploading",
                "job_id": job_id,
                "filename": Path(filename).name[:200],
            })
        return {
            "job_id": job_id,
            "filename": Path(filename).name[:200],
            "input_path": input_path,
            "output_path": output_path,
            "content_length": content_length,
        }

    def finish_offline_upload(self, task):
        try:
            self.offline_queue.put_nowait(task)
        except queue.Full as error:
            raise RuntimeError("离线任务队列已满") from error
        with self.condition:
            self.offline_status["state"] = "queued"
            self.offline_status["error"] = ""

    def fail_offline_upload(self, task, message):
        try:
            task["input_path"].unlink(missing_ok=True)
        except OSError:
            pass
        with self.condition:
            if self.offline_status["job_id"] == task["job_id"]:
                self.offline_status["state"] = "error"
                self.offline_status["error"] = message
                self.offline_status["finished_at"] = datetime.now().isoformat(timespec="seconds")

    def offline_preview(self, job_id):
        with self.condition:
            if self.offline_status["job_id"] != job_id:
                return None
            return self.offline_preview_jpeg

    def offline_result(self, job_id):
        with self.condition:
            if (
                self.offline_status["job_id"] != job_id
                or self.offline_status["state"] != "completed"
            ):
                return None
        path = self.offline_result_dir / f"{job_id}_detected.webm"
        return path if path.exists() and path.is_file() else None

    def wait_for_frame(self, previous_sequence, stream_name):
        with self.condition:
            if stream_name == "input":
                self.condition.wait_for(
                    lambda: self.input_sequence != previous_sequence, timeout=5.0
                )
                return self.input_sequence, self.input_jpeg
            self.condition.wait_for(lambda: self.sequence != previous_sequence, timeout=5.0)
            return self.sequence, self.output_jpeg

    def _set_error(self, message):
        with self.condition:
            self.error = message
            self.condition.notify_all()

    def _capture_loop(self):
        while not self.stop_event.is_set():
            camera_path = Path(f"/dev/video{self.args.camera}")
            if not camera_path.exists():
                self._set_error(
                    f"未检测到 {camera_path}；实时检测暂不可用，离线视频检测可正常使用"
                )
                self.stop_event.wait(1.0)
                continue
            capture = cv2.VideoCapture(self.args.camera, cv2.CAP_V4L2)
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.camera_width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.camera_height)
            capture.set(cv2.CAP_PROP_FPS, self.args.camera_fps)
            # Keep several V4L2 buffers queued. A single UVC/YUYV buffer can
            # starve the driver and make a nominal 15 FPS stream deliver
            # roughly every other frame (~7.5 FPS).
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 4)
            capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            if not capture.isOpened():
                capture.release()
                self._set_error(
                    f"无法打开 {camera_path}；实时检测暂不可用，离线视频检测可正常使用"
                )
                self.stop_event.wait(1.0)
                continue
            failed_reads = 0
            try:
                while failed_reads < 10 and not self.stop_event.is_set():
                    capture_started = time.perf_counter()
                    ok, yuyv_frame = capture.read()
                    raw_captured_at = time.perf_counter()
                    if not ok or yuyv_frame is None:
                        failed_reads += 1
                        self._set_error(f"Cannot read /dev/video{self.args.camera}; retry {failed_reads}/10")
                        self.stop_event.wait(0.1)
                        continue
                    yuyv_frame = np.ascontiguousarray(yuyv_frame, dtype=np.uint8)
                    frame = cv2.cvtColor(yuyv_frame, cv2.COLOR_YUV2BGR_YUY2)
                    captured_at = time.perf_counter()
                    failed_reads = 0
                    if self.error:
                        self._set_error("")
                    self.capture_frame_id += 1
                    item = {
                        "frame_id": self.capture_frame_id,
                        "yuyv_frame": yuyv_frame,
                        "frame": frame,
                        "capture_started": capture_started,
                        "captured_at": captured_at,
                        "capture_read_ms": (raw_captured_at - capture_started) * 1000,
                        "capture_thread_time_ms": (captured_at - capture_started) * 1000,
                        "v4l2_read_ms": (raw_captured_at - capture_started) * 1000,
                        "display_convert_ms": (captured_at - raw_captured_at) * 1000,
                    }
                    dropped = put_latest(self.capture_queue, item)
                    if dropped:
                        with self.condition:
                            self.capture_dropped_frames += dropped
            except Exception as error:
                self._set_error(f"capture {type(error).__name__}: {error}; retrying")
            finally:
                capture.release()
            self.stop_event.wait(1.0)

    def _prepare_offline_input(self, frame):
        height, width = frame.shape[:2]
        scale = min(self.args.imgsz / width, self.args.imgsz / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(
            frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
        )
        left = (self.args.imgsz - resized_width) // 2
        top = (self.args.imgsz - resized_height) // 2
        canvas = np.full(
            (self.args.imgsz, self.args.imgsz, 3), 114, dtype=np.uint8
        )
        canvas[top:top + resized_height, left:left + resized_width] = resized
        # The static-AIPP OM expects RGB888 uint8 NHWC. AIPP performs the
        # model-side layout conversion and normalization on the device.
        input_array = np.ascontiguousarray(canvas[:, :, ::-1][None])
        return input_array, scale, left, top

    def _update_offline_preview(self, job_id, result):
        ok, preview_buffer = cv2.imencode(
            ".jpg", result, [int(cv2.IMWRITE_JPEG_QUALITY), self.args.jpeg_quality]
        )
        if not ok:
            return
        with self.condition:
            if self.offline_status["job_id"] != job_id:
                return
            self.offline_preview_jpeg = preview_buffer.tobytes()
            self.offline_status["preview_sequence"] += 1

    def _process_offline_video(self, session, task):
        capture = None
        writer = None
        started = time.perf_counter()
        job_id = task["job_id"]
        try:
            with self.condition:
                self.offline_status.update({
                    "state": "running",
                    "started_at": datetime.now().isoformat(timespec="seconds"),
                    "finished_at": "",
                    "error": "",
                })
            capture = cv2.VideoCapture(str(task["input_path"]))
            if not capture.isOpened():
                raise RuntimeError("OpenCV/FFmpeg 无法打开上传的视频")
            source_fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not np.isfinite(source_fps) or source_fps <= 0.0:
                source_fps = 15.0
            total_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            width = max(0, int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            height = max(0, int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            if width <= 0 or height <= 0:
                raise RuntimeError("无法读取视频分辨率")
            output_width = width - (width % 2)
            output_height = height - (height % 2)
            if output_width <= 0 or output_height <= 0:
                raise RuntimeError("视频分辨率无效")
            writer = cv2.VideoWriter(
                str(task["output_path"]),
                cv2.VideoWriter_fourcc(*"VP80"),
                source_fps,
                (output_width, output_height),
            )
            if not writer.isOpened():
                raise RuntimeError("无法创建 WebM/VP8 结果视频")
            with self.condition:
                self.offline_status.update({
                    "total_frames": total_frames,
                    "source_fps": source_fps,
                    "input_resolution": f"{width}x{height}",
                    "output_resolution": f"{output_width}x{output_height}",
                })

            processed_frames = 0
            while not self.stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                input_array, scale, left, top = self._prepare_offline_input(frame)
                if not self.warmup_completed:
                    for _ in range(self.args.warmup):
                        session.infer([input_array])
                    with self.condition:
                        self.warmup_completed = True
                outputs = session.infer([input_array])
                boxes, scores, class_ids, _ = postprocess_timed(
                    outputs[0],
                    scale,
                    left,
                    top,
                    frame.shape,
                    self.args.conf,
                    self.args.iou,
                )
                result = draw_detections(
                    frame.copy(), boxes, scores, class_ids
                )
                if result.shape[1] != output_width or result.shape[0] != output_height:
                    result = cv2.resize(result, (output_width, output_height))
                writer.write(result)
                processed_frames += 1
                elapsed = time.perf_counter() - started
                if processed_frames == 1 or processed_frames % 10 == 0:
                    self._update_offline_preview(job_id, result)
                with self.condition:
                    if self.offline_status["job_id"] != job_id:
                        raise RuntimeError("离线任务状态已被替换")
                    self.offline_status.update({
                        "processed_frames": processed_frames,
                        "progress": (
                            min(100.0, processed_frames * 100.0 / total_frames)
                            if total_frames > 0 else 0.0
                        ),
                        "processing_fps": processed_frames / max(elapsed, 1e-9),
                        "elapsed_ms": elapsed * 1000,
                    })
            if processed_frames == 0:
                raise RuntimeError("视频中没有可解码的画面")
            writer.release()
            writer = None
            self._update_offline_preview(job_id, result)
            elapsed = time.perf_counter() - started
            output_bytes = task["output_path"].stat().st_size
            with self.condition:
                self.offline_status.update({
                    "state": "completed",
                    "processed_frames": processed_frames,
                    "total_frames": total_frames or processed_frames,
                    "progress": 100.0,
                    "processing_fps": processed_frames / max(elapsed, 1e-9),
                    "elapsed_ms": elapsed * 1000,
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "output_bytes": output_bytes,
                    "error": "",
                })
        except Exception as error:
            try:
                task["output_path"].unlink(missing_ok=True)
            except OSError:
                pass
            elapsed = time.perf_counter() - started
            with self.condition:
                if self.offline_status["job_id"] == job_id:
                    self.offline_status.update({
                        "state": "error",
                        "elapsed_ms": elapsed * 1000,
                        "finished_at": datetime.now().isoformat(timespec="seconds"),
                        "error": f"{type(error).__name__}: {error}",
                    })
        finally:
            if writer is not None:
                writer.release()
            if capture is not None:
                capture.release()

    def _inference_loop(self):
        session = None
        preprocessor = None
        input_tensor = None
        try:
            model_load_start = time.perf_counter()
            session = InferSession(self.args.device, str(self.state.model))
            self.model_load_time_ms = (time.perf_counter() - model_load_start) * 1000
            # Keep InferSession, DVPP buffers and BaseTensor in one ACL context.
            session.set_context()
            preprocessor = DvppYuyvLetterbox(
                self.args.camera_width,
                self.args.camera_height,
                self.args.imgsz,
                self.args.imgsz,
                self.args.device,
                preserve_current_context=True,
            )
            input_tensor = aclruntime.BaseTensor(
                preprocessor.output_dev, preprocessor.output_size
            )
            while not self.stop_event.is_set():
                if self.state.service_status()["running"]:
                    time.sleep(0.05)
                    continue
                try:
                    offline_task = self.offline_queue.get_nowait()
                except queue.Empty:
                    offline_task = None
                if offline_task is not None:
                    self._process_offline_video(session, offline_task)
                    continue
                try:
                    item = self.capture_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if self.state.service_status()["running"]:
                    time.sleep(0.05)
                    continue
                frame = item["frame"]
                preprocess_start = time.perf_counter()
                preprocessor.preprocess(item["yuyv_frame"])
                preprocess_ms = (time.perf_counter() - preprocess_start) * 1000
                if not self.warmup_completed:
                    for _ in range(self.args.warmup):
                        session.run([input_tensor], out_array=True)
                    with self.condition:
                        self.warmup_completed = True
                    continue
                inference_start = time.perf_counter()
                outputs = session.run([input_tensor], out_array=True)
                inference_ms = (time.perf_counter() - inference_start) * 1000
                boxes, scores, class_ids, stage_timings = postprocess_timed(
                    outputs[0],
                    preprocessor.scale,
                    preprocessor.left,
                    preprocessor.top,
                    frame.shape,
                    self.args.conf,
                    self.args.iou,
                )
                detections = []
                for box, score, class_id in zip(boxes, scores, class_ids):
                    class_index = int(class_id)
                    detections.append({
                        "class_id": class_index,
                        "class_name": COCO_NAMES[class_index] if class_index < len(COCO_NAMES) else str(class_index),
                        "confidence": float(score),
                        "box_xyxy": [float(value) for value in box],
                    })
                metrics = {
                    "model": str(self.state.model),
                    "image": f"/dev/video{self.args.camera}",
                    "camera_input_format": "YUYV 4:2:2",
                    "camera_resolution": f"{self.args.camera_width}x{self.args.camera_height}",
                    "camera_configured_fps": self.args.camera_fps,
                    "output": "YUYV -> H2D -> DVPP RGB letterbox -> AIPP OM -> detections only D2H",
                    "display_path": "host YUYV -> BGR -> draw boxes -> MJPEG",
                    "model_input": "RGB888 uint8 NHWC device buffer (AIPP)",
                    "device": self.args.device,
                    "imgsz": self.args.imgsz,
                    "conf": self.args.conf,
                    "iou": self.args.iou,
                    "image_shape": list(frame.shape),
                    "frame_id": item["frame_id"],
                    "capture_read_ms": item["capture_read_ms"],
                    "capture_thread_time_ms": item["capture_thread_time_ms"],
                    "v4l2_read_ms": item["v4l2_read_ms"],
                    "display_convert_ms": item["display_convert_ms"],
                    "dvpp_preprocess_ms": preprocess_ms,
                    "capture_queue_wait_ms": (preprocess_start - item["captured_at"]) * 1000,
                    "preprocess_time_ms": preprocess_ms,
                    "letterbox_scale": preprocessor.scale,
                    "letterbox_left": preprocessor.left,
                    "letterbox_top": preprocessor.top,
                    "letterbox_resized_width": preprocessor.resized_width,
                    "letterbox_resized_height": preprocessor.resized_height,
                    "device_input_bytes": preprocessor.output_size,
                    "model_load_time_ms": self.model_load_time_ms,
                    "inference_time_ms": inference_ms,
                    "decode_filter_time_ms": stage_timings["decode_filter_time_ms"],
                    "nms_time_ms": stage_timings["nms_time_ms"],
                    "restore_boxes_time_ms": stage_timings["restore_boxes_time_ms"],
                    "postprocess_time_ms": sum(stage_timings.values()),
                    "draw_time_ms": 0.0,
                    "jpeg_encode_time_ms": 0.0,
                    "total_time_ms": 0.0,
                    "end_to_end_time_ms": 0.0,
                    "fps": 1000.0 / max(inference_ms, 1e-9),
                    "camera_fps": 0.0,
                    "pipeline_fps": 0.0,
                    "published_fps": 0.0,
                    "detections_count": len(detections),
                    "detections": detections,
                }
                result_item = {
                    "frame": frame,
                    "boxes": boxes,
                    "scores": scores,
                    "class_ids": class_ids,
                    "capture_started": item["capture_started"],
                    "metrics": metrics,
                }
                dropped = put_latest(self.result_queue, result_item)
                if dropped:
                    with self.condition:
                        self.result_dropped_frames += dropped
        except Exception as error:
            self._set_error(f"inference {type(error).__name__}: {error}")
        finally:
            input_tensor = None
            if preprocessor is not None:
                preprocessor.close()
            if session is not None:
                session.free_resource()

    def _encode_loop(self):
        previous_finished = None
        previous_published = None
        while not self.stop_event.is_set():
            try:
                item = self.result_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            with self.condition:
                viewers = self.viewer_count
            draw_ms = 0.0
            jpeg_ms = 0.0
            input_bytes = None
            output_bytes = None
            if viewers > 0:
                input_encode_start = time.perf_counter()
                input_ok, input_buffer = cv2.imencode(
                    ".jpg", item["frame"], [int(cv2.IMWRITE_JPEG_QUALITY), self.args.jpeg_quality]
                )
                input_encode_ms = (time.perf_counter() - input_encode_start) * 1000
                if not input_ok:
                    self._set_error("Input JPEG encoding failed")
                    continue
                input_bytes = input_buffer.tobytes()
                draw_start = time.perf_counter()
                result = draw_detections(item["frame"].copy(), item["boxes"], item["scores"], item["class_ids"])
                draw_ms = (time.perf_counter() - draw_start) * 1000
                output_encode_start = time.perf_counter()
                ok, output_buffer = cv2.imencode(
                    ".jpg", result, [int(cv2.IMWRITE_JPEG_QUALITY), self.args.jpeg_quality]
                )
                output_encode_ms = (time.perf_counter() - output_encode_start) * 1000
                jpeg_ms = input_encode_ms + output_encode_ms
                if not ok:
                    self._set_error("JPEG encoding failed")
                    continue
                output_bytes = output_buffer.tobytes()
            finished = time.perf_counter()
            pipeline_fps = 0.0 if previous_finished is None else 1.0 / max(finished - previous_finished, 1e-9)
            previous_finished = finished
            published_fps = 0.0
            if output_bytes is not None:
                published_fps = 0.0 if previous_published is None else 1.0 / max(finished - previous_published, 1e-9)
                previous_published = finished
            metrics = dict(item["metrics"])
            metrics.update({
                "draw_time_ms": draw_ms,
                "jpeg_encode_time_ms": jpeg_ms,
                "yolo_three_stage_time_ms": metrics.get("preprocess_time_ms", 0.0) + metrics.get("inference_time_ms", 0.0) + metrics.get("postprocess_time_ms", 0.0),
                "inference_thread_time_ms": metrics.get("preprocess_time_ms", 0.0) + metrics.get("inference_time_ms", 0.0) + metrics.get("postprocess_time_ms", 0.0),
                "encode_thread_time_ms": draw_ms + jpeg_ms,
                "total_time_ms": (finished - item["capture_started"]) * 1000,
                "end_to_end_time_ms": (finished - item["capture_started"]) * 1000,
                "camera_fps": pipeline_fps,
                "pipeline_fps": pipeline_fps,
                "published_fps": published_fps,
                "viewer_count": viewers,
            })
            metrics["pipeline_critical_time_ms"] = max(metrics.get("capture_thread_time_ms", metrics.get("capture_read_ms", 0.0)), metrics["inference_thread_time_ms"], metrics["encode_thread_time_ms"])
            with self.condition:
                metrics["capture_dropped_frames"] = self.capture_dropped_frames
                metrics["result_dropped_frames"] = self.result_dropped_frames
                self.metrics = metrics
                self.error = ""
                if input_bytes is not None:
                    self.input_jpeg = input_bytes
                    self.input_sequence += 1
                if output_bytes is not None:
                    self.output_jpeg = output_bytes
                    self.sequence += 1
                if input_bytes is not None or output_bytes is not None:
                    self.condition.notify_all()


class CameraDashboardState(dashboard.DashboardState):
    def __init__(self, args):
        super().__init__(args)
        self.camera = CameraWorker(self, args)

    def start(self, task):
        if self.camera.offline_busy():
            return False
        return super().start(task)

    def data(self):
        payload = super().data()
        camera = self.camera.snapshot()
        payload["camera_enabled"] = True
        payload["camera"] = {
            "device": f"/dev/video{self.args.camera}",
            "sequence": camera["sequence"],
            "input_sequence": camera["input_sequence"],
            "error": camera["error"],
            "started_at": camera["started_at"],
            "viewer_count": camera["viewer_count"],
            "capture_dropped_frames": camera["capture_dropped_frames"],
            "result_dropped_frames": camera["result_dropped_frames"],
            "warmup_completed": camera["warmup_completed"],
        }
        payload["offline"] = self.camera.offline_snapshot()
        if camera["metrics"]:
            payload["metrics"] = camera["metrics"]
        if camera["error"] and not payload["service"]["last_error"]:
            payload["service"]["last_error"] = camera["error"]
        return payload


class CameraDashboardHandler(dashboard.DashboardHandler):
    server_version = "YOLOv8PipelinedCameraDashboard/2.0"

    def _stream(self, stream_name):
        self.state.camera.add_viewer()
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Connection", "close")
            self.end_headers()
            sequence = -1
            while True:
                sequence, frame = self.state.camera.wait_for_frame(sequence, stream_name)
                if frame is None:
                    continue
                self.wfile.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(frame)).encode()
                    + b"\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        finally:
            self.state.camera.remove_viewer()

    def _send_result_file(self, path, download=False):
        file_size = path.stat().st_size
        start = 0
        end = file_size - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("Range", "")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if match is None:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return
            first, last = match.groups()
            if first:
                start = int(first)
                end = int(last) if last else end
            elif last:
                length = min(int(last), file_size)
                start = file_size - length
            if start >= file_size or end < start:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return
            end = min(end, file_size - 1)
            status = HTTPStatus.PARTIAL_CONTENT
        content_length = max(0, end - start + 1)
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "video/webm")
        self.send_header("Content-Length", str(content_length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        if download:
            self.send_header(
                "Content-Disposition", f'attachment; filename="{path.name}"'
            )
        self.end_headers()
        try:
            with path.open("rb") as file:
                file.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk = file.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path == "/":
            self.send_bytes(CAMERA_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/stream/input.mjpg":
            self._stream("input")
        elif path == "/stream/output.mjpg":
            self._stream("output")
        elif path == "/api/offline/status":
            self.send_json(self.state.camera.offline_snapshot())
        elif path == "/api/offline/preview":
            job_id = query.get("job_id", [""])[0]
            preview = self.state.camera.offline_preview(job_id)
            if preview is None:
                self.send_json({"error": "preview not found"}, HTTPStatus.NOT_FOUND)
                return
            self.send_bytes(preview, "image/jpeg")
        elif path == "/api/offline/result":
            job_id = query.get("job_id", [""])[0]
            result_path = self.state.camera.offline_result(job_id)
            if result_path is None:
                self.send_json({"error": "result not found"}, HTTPStatus.NOT_FOUND)
                return
            self._send_result_file(
                result_path, download=query.get("download", ["0"])[0] == "1"
            )
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/offline/upload":
            super().do_POST()
            return
        query = parse_qs(parsed.query)
        filename = query.get("filename", [""])[0].strip()
        if not filename:
            self.send_json({"accepted": False, "error": "缺少视频文件名"}, HTTPStatus.BAD_REQUEST)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        max_upload_bytes = 1024 * 1024 * 1024
        if content_length <= 0:
            self.send_json({"accepted": False, "error": "上传内容为空"}, HTTPStatus.BAD_REQUEST)
            return
        if content_length > max_upload_bytes:
            self.send_json(
                {"accepted": False, "error": "视频不能超过 1 GiB"},
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
            return
        try:
            task = self.state.camera.begin_offline_upload(filename, content_length)
        except ValueError as error:
            self.send_json({"accepted": False, "error": str(error)}, HTTPStatus.BAD_REQUEST)
            return
        except RuntimeError as error:
            self.send_json({"accepted": False, "error": str(error)}, HTTPStatus.CONFLICT)
            return
        try:
            remaining = content_length
            with task["input_path"].open("xb") as file:
                while remaining > 0:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise RuntimeError("客户端在视频上传完成前断开连接")
                    file.write(chunk)
                    remaining -= len(chunk)
            self.state.camera.finish_offline_upload(task)
        except Exception as error:
            self.state.camera.fail_offline_upload(task, str(error))
            self.send_json(
                {"accepted": False, "error": str(error)},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return
        self.send_json(
            {"accepted": True, "job_id": task["job_id"]}, HTTPStatus.ACCEPTED
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model", default="models/weights/om/yolov8n_bs1_aipp_rgb.om"
    )
    parser.add_argument("--image", default="datasets/images/test/bus.jpg")
    parser.add_argument("--output", default="runs/predict/bus_result.jpg")
    parser.add_argument("--metrics", default="runs/reports/bus_result_metrics.json")
    parser.add_argument("--profile-output", default="runs/msprof")
    parser.add_argument("--report", default="runs/reports/bus_report.html")
    parser.add_argument("--log", default="logs/dashboard_tasks.log")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--interval", type=float, default=0)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=1024)
    parser.add_argument("--camera-height", type=int, default=576)
    parser.add_argument("--camera-fps", type=int, default=15)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="只启用离线视频检测，不打开摄像头",
    )
    return parser.parse_args()


def main():
    def handle_termination(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handle_termination)
    args = parse_args()
    state = CameraDashboardState(args)
    server = ThreadingHTTPServer((args.host, args.port), CameraDashboardHandler)
    server.daemon_threads = True
    server.state = state
    state.camera.start()
    if args.interval > 0:
        threading.Thread(target=dashboard.automatic_inference, args=(state, args.interval), daemon=True).start()
    print(f"pipelined camera dashboard: http://{args.host}:{args.port}", flush=True)
    print(f"camera: /dev/video{args.camera}", flush=True)
    print(f"model: {state.model}", flush=True)
    print(f"warmup: {args.warmup}", flush=True)
    print(f"mode: {'offline only' if args.offline_only else 'realtime + offline'}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("正在停止服务并释放 NPU 资源...", flush=True)
    finally:
        server.server_close()
        try:
            state.camera.stop()
        finally:
            InferSession.finalize()
        print("服务已安全停止", flush=True)


if __name__ == "__main__":
    main()
