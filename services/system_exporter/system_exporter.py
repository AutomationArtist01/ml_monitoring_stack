"""
System / hardware exporter (psutil → Prometheus).

Exposes what the machine running the stack is doing – CPU, memory, swap, disk, network, load –
plus the resource usage of the stack's own containers (via the Docker socket, if mounted).
This complements the *application* metrics (gateway/model/drift) with the *saturation* golden signal.

  GET /metrics
    sys_cpu_percent{mode="total"}            overall CPU utilisation (%)
    sys_cpu_percent_per_core{core="0".."n"}  per core (%)
    sys_load_average{period="1m|5m|15m"}
    sys_memory_bytes{kind="total|available|used"}   sys_memory_percent
    sys_swap_bytes{kind="total|used"}
    sys_disk_bytes{mount="/",kind="total|used|free"} sys_disk_percent{mount="/"}
    sys_net_bytes_total{direction="sent|recv"}      (counter)
    sys_boot_time_seconds, sys_process_count
    container_cpu_percent{name}, container_memory_bytes{name}, container_memory_limit_bytes{name}
      (only when /var/run/docker.sock is mounted)

Env: SCRAPE_INTERVAL (5) seconds between psutil samples · DISK_MOUNT (/) · DOCKER_STATS (1)
Note: inside Docker Desktop (macOS/Windows) "host" numbers describe the Linux VM that runs the containers.
"""
import os
import threading
import time

import psutil
from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest

INTERVAL = float(os.environ.get("SCRAPE_INTERVAL", "5"))
DISK_MOUNT = os.environ.get("DISK_MOUNT", "/")
DOCKER_STATS = os.environ.get("DOCKER_STATS", "1") == "1"

app = FastAPI(title="System exporter (psutil)", version="1.0.0")

CPU = Gauge("sys_cpu_percent", "CPU utilisation percent", ["mode"])
CPU_CORE = Gauge("sys_cpu_percent_per_core", "CPU utilisation percent per core", ["core"])
LOAD = Gauge("sys_load_average", "Load average", ["period"])
MEM = Gauge("sys_memory_bytes", "Memory bytes", ["kind"])
MEM_PCT = Gauge("sys_memory_percent", "Memory used percent")
SWAP = Gauge("sys_swap_bytes", "Swap bytes", ["kind"])
DISK = Gauge("sys_disk_bytes", "Disk bytes", ["mount", "kind"])
DISK_PCT = Gauge("sys_disk_percent", "Disk used percent", ["mount"])
NET = Counter("sys_net_bytes", "Network bytes", ["direction"])
BOOT = Gauge("sys_boot_time_seconds", "Boot time (unix)")
PROCS = Gauge("sys_process_count", "Number of processes")
CORES = Gauge("sys_cpu_cores", "Logical CPU cores")
C_CPU = Gauge("container_cpu_percent", "Container CPU percent (docker stats)", ["name"])
C_MEM = Gauge("container_memory_bytes", "Container memory usage bytes", ["name"])
C_MEM_LIMIT = Gauge("container_memory_limit_bytes", "Container memory limit bytes", ["name"])
LAST = Gauge("sys_exporter_last_sample_timestamp", "Unix time of last psutil sample")

_net_last = {"sent": 0, "recv": 0}


def sample_host():
    CPU.labels("total").set(psutil.cpu_percent(interval=None))
    for i, p in enumerate(psutil.cpu_percent(interval=None, percpu=True)):
        CPU_CORE.labels(str(i)).set(p)
    CORES.set(psutil.cpu_count() or 0)
    try:
        l1, l5, l15 = psutil.getloadavg()
        LOAD.labels("1m").set(l1); LOAD.labels("5m").set(l5); LOAD.labels("15m").set(l15)
    except (AttributeError, OSError):
        pass
    vm = psutil.virtual_memory()
    MEM.labels("total").set(vm.total); MEM.labels("available").set(vm.available); MEM.labels("used").set(vm.used)
    MEM_PCT.set(vm.percent)
    sw = psutil.swap_memory()
    SWAP.labels("total").set(sw.total); SWAP.labels("used").set(sw.used)
    du = psutil.disk_usage(DISK_MOUNT)
    DISK.labels(DISK_MOUNT, "total").set(du.total); DISK.labels(DISK_MOUNT, "used").set(du.used)
    DISK.labels(DISK_MOUNT, "free").set(du.free); DISK_PCT.labels(DISK_MOUNT).set(du.percent)
    n = psutil.net_io_counters()
    for k, v in (("sent", n.bytes_sent), ("recv", n.bytes_recv)):
        if v >= _net_last[k]:
            NET.labels(k).inc(v - _net_last[k])
        _net_last[k] = v
    BOOT.set(psutil.boot_time())
    PROCS.set(len(psutil.pids()))
    LAST.set(time.time())


def sample_containers():
    """Per-container CPU/memory via the Docker Engine API (one-shot stats)."""
    try:
        import docker
        client = docker.from_env()
        for c in client.containers.list():
            s = c.stats(stream=False)
            cpu_delta = s["cpu_stats"]["cpu_usage"]["total_usage"] - s["precpu_stats"]["cpu_usage"]["total_usage"]
            sys_delta = s["cpu_stats"].get("system_cpu_usage", 0) - s["precpu_stats"].get("system_cpu_usage", 0)
            ncpu = s["cpu_stats"].get("online_cpus") or len(s["cpu_stats"]["cpu_usage"].get("percpu_usage", [1]))
            pct = (cpu_delta / sys_delta) * ncpu * 100.0 if sys_delta > 0 else 0.0
            name = c.name.replace("ml-monitoring-", "").rsplit("-", 1)[0]
            C_CPU.labels(name).set(pct)
            C_MEM.labels(name).set(s["memory_stats"].get("usage", 0))
            C_MEM_LIMIT.labels(name).set(s["memory_stats"].get("limit", 0))
    except Exception as e:  # noqa: BLE001 – docker socket not mounted or not permitted
        print("[system-exporter] docker stats unavailable:", e.__class__.__name__)


def loop():
    psutil.cpu_percent(interval=None)  # prime
    while True:
        try:
            sample_host()
            if DOCKER_STATS and os.path.exists("/var/run/docker.sock"):
                sample_containers()
        except Exception as e:  # noqa: BLE001
            print("[system-exporter] sample error:", e)
        time.sleep(INTERVAL)


threading.Thread(target=loop, daemon=True).start()


@app.get("/health")
def health():
    return {"status": "ok", "cpu_percent": psutil.cpu_percent(interval=None), "memory_percent": psutil.virtual_memory().percent}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
