# catcher.py — fake Slack: stores the JSON Alertmanager sends and shows it at /
import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

app = FastAPI(title="Webhook catcher (fake Slack)")
received: list[dict] = []

@app.post("/slack")
@app.post("/{path:path}")
async def catch(request: Request, path: str = "slack"):
    try:
        body = await request.json()
    except Exception:
        body = {"raw": (await request.body()).decode(errors="replace")}
    received.insert(0, {"ts": time.strftime("%H:%M:%S"), "path": "/" + path, "body": body})
    del received[200:]
    print("[webhook]", json.dumps(body)[:500], flush=True)
    return {"ok": True}

@app.get("/", response_class=HTMLResponse)
def index():
    rows = "".join(
        f"<div style='border:1px solid #ccc;margin:8px;padding:8px;font-family:monospace'>"
        f"<b>{r['ts']}</b> {r['path']}<pre style='white-space:pre-wrap'>{json.dumps(r['body'], indent=1)}</pre></div>"
        for r in received)
    return f"<h2>Fake Slack – {len(received)} notifications</h2>{rows or '<i>nothing yet</i>'}"

@app.get("/api")
def api():
    return received
