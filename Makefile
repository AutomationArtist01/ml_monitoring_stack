# Team 4 – Real-Time Model Monitoring & Alerting Stack
SHELL := /bin/bash
LG    := http://localhost:8090

.PHONY: up down restart logs ps build train dashboards reload-prom status \
        scenario-normal scenario-drift scenario-latency scenario-errors scenario-burst scenario-bad \
        predict smoke open

up:            ## build + start the whole stack
	docker compose up -d --build
	@echo "Grafana http://localhost:3000 (admin/admin) · Prometheus :9090 · Alertmanager :9093 · MLflow :5001"
down:          ## stop and remove containers (keeps volumes)
	docker compose down
nuke:          ## stop and DELETE volumes (Prometheus/Grafana/MLflow data)
	docker compose down -v
restart:
	docker compose restart
build:
	docker compose build
logs:
	docker compose logs -f --tail=100
ps:
	docker compose ps
train:         ## ZenML pipeline: validate → Optuna (25 trials) → evaluate → register → drift reference
	docker compose run --rm trainer
train-quick:   ## faster: 8 Optuna trials
	docker compose run --rm trainer python run_pipeline.py --trials 8
dashboards:    ## regenerate Grafana dashboards from monitoring/grafana/build_dashboards.py
	python3 monitoring/grafana/build_dashboards.py
reload-prom:   ## hot-reload Prometheus config/rules after editing
	curl -s -X POST http://localhost:9090/-/reload && echo reloaded
status:        ## quick health summary
	@curl -s localhost:8000/health; echo
	@curl -s localhost:8080/health; echo
	@curl -s localhost:9105/health; echo
	@curl -s localhost:8090/status; echo
	@curl -s 'localhost:9090/api/v1/query?query=up' | python3 -c "import sys,json;[print(r['metric']['job'].ljust(16),'up='+r['value'][1]) for r in json.load(sys.stdin)['data']['result']]" 
# ---- demo scenarios ------------------------------------------------------
scenario-normal:
	curl -s -X POST $(LG)/scenario/normal; echo
scenario-drift:
	curl -s -X POST $(LG)/scenario/drift; echo
scenario-latency:
	curl -s -X POST '$(LG)/scenario/latency?ms=800'; echo
scenario-errors:
	curl -s -X POST '$(LG)/scenario/errors?rate=0.3'; echo
scenario-burst:
	curl -s -X POST '$(LG)/scenario/burst?rps=40'; echo
scenario-bad:
	curl -s -X POST $(LG)/scenario/bad_payload; echo
predict:       ## one prediction through the gateway
	curl -s -X POST localhost:8080/predict -H 'content-type: application/json' -d @tests/sample_request.json; echo
smoke:         ## end-to-end smoke test
	python3 tests/smoke_test.py
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
