#!/usr/bin/env python3
"""Print current alert state from Prometheus + Alertmanager and what the notification sinks received."""
import json
import urllib.request


def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.load(r)

print("== Prometheus ALERTS")
res = get("http://localhost:9090/api/v1/query?query=ALERTS")["data"]["result"]
for r in sorted(res, key=lambda r: (r["metric"]["alertstate"], r["metric"]["alertname"])):
    m = r["metric"]
    print(f"  {m['alertstate']:8s} {m['alertname']:26s} {m.get('feature', ''):18s} {m.get('severity', '')}")
if not res:
    print("  (none)")

print("== Alertmanager active alerts")
for a in get("http://localhost:9093/api/v2/alerts"):
    lb = a["labels"]
    print(f"  {a['status']['state']:10s} {lb['alertname']:26s} {lb.get('feature', ''):18s} receivers={[x['name'] for x in a['receivers']]}")

print("== Fake Slack (webhook catcher)")
d = get("http://localhost:8091/api")
print(f"  {len(d)} notifications")
for x in d[:8]:
    b = x["body"]
    att = (b.get("attachments") or [{}])[0]
    print(f"   {x['ts']} {b.get('channel', ''):12s} {att.get('title', '')[:90]}")

print("== Mailhog")
d = get("http://localhost:8025/api/v2/messages")
print(f"  {d['total']} emails")
for m in d["items"][:8]:
    print("   ", m["Content"]["Headers"]["Subject"][0])
