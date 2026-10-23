#!/usr/bin/env python3
"""网站公共基础模块：提供页面、HTTP 接口、任务状态和监控数据。

camera_dashboard.py 会导入本模块并复用 DashboardState、请求处理器和页面模板。
它通常不作为摄像头实时检测网站的直接启动入口。
"""

import argparse
import csv
import json
import mimetypes
import subprocess
import sys
import threading
import time
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def absolute(path):
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def number(value, default=0.0):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def read_csv(path):
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def find_latest(directory, pattern):
    matches = sorted(directory.glob(pattern)) if directory.exists() else []
    return matches[-1] if matches else None


class DashboardState:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.running = False
        self.task = ""
        self.last_error = ""
        self.last_started = ""
        self.last_finished = ""
        self.log_path = absolute(args.log)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def model(self):
        return absolute(self.args.model)

    @property
    def image(self):
        return absolute(self.args.image)

    @property
    def output(self):
        return absolute(self.args.output)

    @property
    def metrics(self):
        return absolute(self.args.metrics)

    @property
    def profile_output(self):
        return absolute(self.args.profile_output)

    @property
    def report(self):
        return absolute(self.args.report)

    def service_status(self):
        with self.lock:
            return {"running": self.running, "task": self.task, "last_error": self.last_error, "last_started": self.last_started, "last_finished": self.last_finished}

    def start(self, task):
        with self.lock:
            if self.running:
                return False
            self.running = True
            self.task = task
            self.last_error = ""
            self.last_started = datetime.now().isoformat(timespec="seconds")
        threading.Thread(target=self._run, args=(task,), daemon=True).start()
        return True

    def _infer_command(self):
        return [sys.executable, str(PROJECT_ROOT / "scripts/infer_om.py"), f"--model={self.model}", f"--image={self.image}", f"--output={self.output}", f"--device={self.args.device}", f"--imgsz={self.args.imgsz}", f"--conf={self.args.conf}", f"--iou={self.args.iou}", f"--metrics-output={self.metrics}"]

    def _profile_command(self):
        return [sys.executable, str(PROJECT_ROOT / "scripts/profile_om.py"), f"--model={self.model}", f"--image={self.image}", f"--output={self.output}", f"--device={self.args.device}", f"--imgsz={self.args.imgsz}", f"--conf={self.args.conf}", f"--iou={self.args.iou}", f"--metrics-output={self.metrics}", f"--profile-output={self.profile_output}", f"--report={self.report}", f"--top={self.args.top}"]

    def _run(self, task):
        command = self._profile_command() if task == "profile" else self._infer_command()
        try:
            with self.log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] task={task}\n")
                log.flush()
                result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"{task} exited with status {result.returncode}; see {self.log_path}")
        except Exception as error:
            with self.lock:
                self.last_error = str(error)
        finally:
            with self.lock:
                self.running = False
                self.task = ""
                self.last_finished = datetime.now().isoformat(timespec="seconds")

    def latest_profile(self):
        profiles = [path for path in self.profile_output.glob("PROF_*") if path.is_dir()] if self.profile_output.exists() else []
        return max(profiles, key=lambda path: path.stat().st_mtime) if profiles else None

    def data(self):
        metrics = {}
        if self.metrics.exists():
            try:
                metrics = json.loads(self.metrics.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metrics = {}
        profile = self.latest_profile()
        profiler_output = profile / "mindstudio_profiler_output" if profile else None
        op_path = find_latest(profiler_output, "op_summary_*.csv") if profiler_output else None
        statistic_path = find_latest(profiler_output, "op_statistic_*.csv") if profiler_output else None
        api_path = find_latest(profiler_output, "api_statistic_*.csv") if profiler_output else None
        op_rows = read_csv(op_path)
        op_rows.sort(key=lambda row: number(row.get("Task Duration(us)")), reverse=True)
        statistic_rows = read_csv(statistic_path)
        statistic_rows.sort(key=lambda row: number(row.get("Total Time(us)")), reverse=True)
        api_rows = read_csv(api_path)
        model_execute_us = next((number(row.get("Time(us)")) for row in api_rows if row.get("API Name") == "ModelExecute"), 0.0)
        operators = []
        for row in op_rows:
            operators.append({"name": row.get("Op Name", ""), "type": row.get("OP Type", ""), "task_type": row.get("Task Type", ""), "duration_us": number(row.get("Task Duration(us)")), "wait_us": number(row.get("Task Wait Time(us)")), "aicore_us": number(row.get("aicore_time(us)")), "memory_bound": number(row.get("memory_bound")), "cube_utilization": number(row.get("cube_utilization(%)"))})
        statistics = []
        for row in statistic_rows:
            statistics.append({"type": row.get("OP Type", ""), "core": row.get("Core Type", ""), "count": int(number(row.get("Count"))), "total_us": number(row.get("Total Time(us)")), "avg_us": number(row.get("Avg Time(us)")), "max_us": number(row.get("Max Time(us)")), "ratio": number(row.get("Ratio(%)"))})
        return {"metrics": metrics, "service": self.service_status(), "profile_dir": str(profile) if profile else "", "model_execute_ms": model_execute_us / 1000, "operator_total_ms": sum(row["duration_us"] for row in operators) / 1000, "top_operators": operators[:20], "operators": operators, "statistics": statistics, "updated_at": datetime.now().isoformat(timespec="seconds")}


DASHBOARD_HTML = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>YOLOv8 OM 实时监控</title><style>:root{--bg:#07111f;--panel:#101d31;--panel2:#142641;--border:#29415f;--text:#eaf2ff;--muted:#94a9c2;--green:#22d3a6;--blue:#52a8ff;--orange:#ffbd59;--red:#ff6b7a}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 20% 0,#102643,#07111f 38%);color:var(--text);font-family:Inter,"Noto Sans SC",system-ui,sans-serif}main{width:min(1540px,95vw);margin:28px auto 70px}.header{display:flex;justify-content:space-between;align-items:flex-start;gap:18px}.header h1{margin:0 0 7px;font-size:32px}.muted{color:var(--muted)}.status{display:flex;align-items:center;gap:9px;background:var(--panel);border:1px solid var(--border);padding:10px 14px;border-radius:999px}.dot{width:10px;height:10px;border-radius:50%;background:var(--green);box-shadow:0 0 14px var(--green)}.dot.running{background:var(--orange);box-shadow:0 0 14px var(--orange);animation:pulse 1s infinite}.dot.error{background:var(--red);box-shadow:0 0 14px var(--red)}@keyframes pulse{50%{opacity:.35}}.actions{display:flex;gap:10px;margin:20px 0}.actions button{border:1px solid var(--border);background:var(--panel2);color:var(--text);padding:11px 17px;border-radius:10px;font-weight:700;cursor:pointer}.actions button.primary{background:linear-gradient(90deg,#138c78,#2276bc)}.actions button:disabled{opacity:.45;cursor:not-allowed}.cards{display:grid;grid-template-columns:repeat(6,minmax(130px,1fr));gap:13px}.card,.section{background:rgba(16,29,49,.94);border:1px solid var(--border);box-shadow:0 18px 45px rgba(0,0,0,.18)}.card{border-radius:14px;padding:16px}.card span{color:var(--muted);font-size:13px}.card strong{display:block;font-size:24px;margin-top:8px}.section{border-radius:18px;padding:21px;margin-top:20px}.section h2{margin:0 0 17px;font-size:20px}.images{display:grid;grid-template-columns:1fr 1fr;gap:17px}.image-box{border:1px solid var(--border);background:#050c15;border-radius:13px;padding:12px}.image-box h3{margin:0 0 10px}.image-box img{width:100%;height:520px;display:block;object-fit:contain;border-radius:9px}.bar{display:grid;grid-template-columns:minmax(240px,35%) 1fr 100px;gap:12px;align-items:center;margin:9px 0}.bar-name{font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.bar-name small{display:block;color:var(--muted);margin-top:3px}.bar-track{height:17px;background:#07111f;border-radius:999px;overflow:hidden}.bar-fill{height:100%;background:linear-gradient(90deg,var(--green),var(--blue));border-radius:999px}.bar-value{text-align:right;font-size:12px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.table-wrap{overflow:auto;max-height:560px;border:1px solid var(--border);border-radius:11px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:10px 12px;border-bottom:1px solid #223b59;text-align:left;white-space:nowrap}th{position:sticky;top:0;background:#19304f;z-index:1}tr:hover td{background:#142641}.search{width:min(420px,100%);margin-bottom:12px;background:#07111f;border:1px solid var(--border);color:var(--text);border-radius:9px;padding:10px 12px}.error-box{display:none;margin:15px 0;padding:12px 14px;border:1px solid #81404c;background:#351923;color:#ffc3cb;border-radius:10px}.path{font-size:12px;color:var(--muted);word-break:break-all;margin-top:15px}@media(max-width:1000px){.cards{grid-template-columns:repeat(2,1fr)}.images,.grid{grid-template-columns:1fr}.image-box img{height:auto}.bar{grid-template-columns:180px 1fr 85px}}</style></head><body><main><div class="header"><div><h1>YOLOv8 OM 实时监控</h1><div class="muted">页面每 2 秒刷新数据；普通推理和完整 Profiling 分开执行。</div></div><div class="status"><span id="statusDot" class="dot"></span><span id="statusText">正在连接</span></div></div><div id="errorBox" class="error-box"></div><div class="actions"><button id="inferButton" class="primary" onclick="runTask('infer')">立即推理</button><button id="profileButton" onclick="runTask('profile')">运行性能分析</button><span class="muted" id="updatedAt"></span></div><div class="cards"><div class="card"><span>普通推理时间</span><strong id="inference">--</strong></div><div class="card"><span>理论 FPS</span><strong id="fps">--</strong></div><div class="card"><span>检测数量</span><strong id="detections">--</strong></div><div class="card"><span>msprof ModelExecute</span><strong id="modelExecute">--</strong></div><div class="card"><span>算子耗时之和</span><strong id="operatorTotal">--</strong></div><div class="card"><span>NPU</span><strong id="device">--</strong></div></div><section class="section"><h2>最新检测画面</h2><div class="images"><div class="image-box"><h3>输入</h3><img id="inputImage" alt="输入图片"></div><div class="image-box"><h3>检测结果</h3><img id="outputImage" alt="检测结果"></div></div></section><section class="section"><h2>耗时最高的 20 个算子</h2><div id="operatorBars"></div></section><div class="grid"><section class="section"><h2>算子类型汇总</h2><div class="table-wrap"><table><thead><tr><th>类型</th><th>次数</th><th>总耗时(us)</th><th>平均(us)</th><th>占比</th></tr></thead><tbody id="statBody"></tbody></table></div></section><section class="section"><h2>检测目标</h2><div class="table-wrap"><table><thead><tr><th>类别</th><th>置信度</th><th>坐标 xyxy</th></tr></thead><tbody id="detectionBody"></tbody></table></div></section></div><section class="section"><h2>逐算子明细</h2><input id="search" class="search" placeholder="搜索算子名称或类型"><div class="table-wrap"><table><thead><tr><th>Op Name</th><th>OP Type</th><th>Duration(us)</th><th>AI Core(us)</th><th>Wait(us)</th><th>Memory Bound</th><th>Cube(%)</th></tr></thead><tbody id="operatorBody"></tbody></table></div><div id="profilePath" class="path"></div></section></main><script>let latestOperators=[];const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const fmt=(v,n=2)=>Number(v||0).toFixed(n);async function loadData(){try{const response=await fetch('/api/data',{cache:'no-store'});const data=await response.json();const m=data.metrics||{},s=data.service||{};document.getElementById('inference').textContent=fmt(m.inference_time_ms)+' ms';document.getElementById('fps').textContent=fmt(m.fps);document.getElementById('detections').textContent=m.detections_count??0;document.getElementById('modelExecute').textContent=fmt(data.model_execute_ms)+' ms';document.getElementById('operatorTotal').textContent=fmt(data.operator_total_ms)+' ms';document.getElementById('device').textContent='Device '+(m.device??0);document.getElementById('updatedAt').textContent='页面刷新：'+data.updated_at;const dot=document.getElementById('statusDot'),status=document.getElementById('statusText'),error=document.getElementById('errorBox');dot.className='dot'+(s.running?' running':s.last_error?' error':'');status.textContent=s.running?(s.task==='profile'?'性能分析运行中':'推理运行中'):s.last_error?'任务失败':'服务在线';error.style.display=s.last_error?'block':'none';error.textContent=s.last_error||'';document.getElementById('inferButton').disabled=!!s.running;document.getElementById('profileButton').disabled=!!s.running;const stamp=Date.now();document.getElementById('inputImage').src='/api/image/input?t='+stamp;document.getElementById('outputImage').src='/api/image/output?t='+stamp;renderBars(data.top_operators||[]);renderStats(data.statistics||[]);renderDetections(m.detections||[]);latestOperators=data.operators||[];renderOperators();document.getElementById('profilePath').textContent=data.profile_dir?'最新 Profiling：'+data.profile_dir:'尚无 Profiling 数据'}catch(error){document.getElementById('statusDot').className='dot error';document.getElementById('statusText').textContent='连接失败'}}async function runTask(task){await fetch('/api/'+task,{method:'POST'});setTimeout(loadData,300)}function renderBars(rows){const max=Math.max(...rows.map(r=>r.duration_us),1);document.getElementById('operatorBars').innerHTML=rows.length?rows.map(r=>`<div class="bar"><div class="bar-name" title="${esc(r.name)}">${esc(r.name)}<small>${esc(r.type)}</small></div><div class="bar-track"><div class="bar-fill" style="width:${Math.max(r.duration_us/max*100,.8)}%"></div></div><div class="bar-value">${fmt(r.duration_us,3)} us</div></div>`).join(''):'<div class="muted">尚无逐算子数据，请运行性能分析。</div>'}function renderStats(rows){document.getElementById('statBody').innerHTML=rows.map(r=>`<tr><td>${esc(r.type)}</td><td>${r.count}</td><td>${fmt(r.total_us,3)}</td><td>${fmt(r.avg_us,3)}</td><td>${fmt(r.ratio,2)}%</td></tr>`).join('')}function renderDetections(rows){document.getElementById('detectionBody').innerHTML=rows.map(r=>`<tr><td>${esc(r.class_name)}</td><td>${fmt(r.confidence,4)}</td><td>${(r.box_xyxy||[]).map(v=>fmt(v,1)).join(', ')}</td></tr>`).join('')}function renderOperators(){const q=document.getElementById('search').value.toLowerCase();document.getElementById('operatorBody').innerHTML=latestOperators.filter(r=>(r.name+' '+r.type).toLowerCase().includes(q)).map(r=>`<tr><td title="${esc(r.name)}">${esc(r.name)}</td><td>${esc(r.type)}</td><td>${fmt(r.duration_us,3)}</td><td>${fmt(r.aicore_us,3)}</td><td>${fmt(r.wait_us,3)}</td><td>${fmt(r.memory_bound,3)}</td><td>${fmt(r.cube_utilization,2)}</td></tr>`).join('')}document.getElementById('search').addEventListener('input',renderOperators);loadData();setInterval(loadData,2000);</script></body></html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "YOLOv8Dashboard/1.0"

    @property
    def state(self):
        return self.server.state

    def log_message(self, format, *args):
        return

    def send_bytes(self, content, content_type, status=HTTPStatus.OK):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, payload, status=HTTPStatus.OK):
        self.send_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self.send_bytes(DASHBOARD_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/data":
            self.send_json(self.state.data())
        elif path == "/api/health":
            self.send_json({"status": "ok", "service": self.state.service_status()})
        elif path in ("/api/image/input", "/api/image/output"):
            metrics = self.state.data().get("metrics", {})
            image_path = Path(metrics.get("image", self.state.image)) if path.endswith("input") else Path(metrics.get("output", self.state.output))
            if not image_path.exists():
                self.send_json({"error": "image not found"}, HTTPStatus.NOT_FOUND)
                return
            content_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
            self.send_bytes(image_path.read_bytes(), content_type)
        else:
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ("/api/infer", "/api/profile"):
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        task = "profile" if path.endswith("profile") else "infer"
        if not self.state.start(task):
            self.send_json({"accepted": False, "error": "another task is running"}, HTTPStatus.CONFLICT)
            return
        self.send_json({"accepted": True, "task": task}, HTTPStatus.ACCEPTED)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="models/weights/om/yolov8n_bs1.om")
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
    parser.add_argument("--interval", type=float, default=0, help="Automatically run normal inference every N seconds; 0 disables it")
    return parser.parse_args()


def automatic_inference(state, interval):
    while True:
        time.sleep(interval)
        state.start("infer")


def main():
    args = parse_args()
    state = DashboardState(args)
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    server.state = state
    if args.interval > 0:
        threading.Thread(target=automatic_inference, args=(state, args.interval), daemon=True).start()
    print(f"dashboard: http://{args.host}:{args.port}")
    print(f"model: {state.model}")
    print(f"image: {state.image}")
    server.serve_forever()


if __name__ == "__main__":
    main()
