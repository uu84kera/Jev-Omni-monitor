# Jev-Omni Monitor MVP

A small, provider-neutral implementation of a hierarchical camera anomaly pipeline:

```text
camera or image stream
  -> sample 3 ordered frames
  -> cheap motion gate
  -> Jev-Omni safe/suspicious classification
  -> suspicious-only strong VLM analysis
  -> typed anomaly + severity
  -> policy decision + JSONL event log
```

## Why this shape

Jev-Omni is a 12B self-hosted classifier with substantial CUDA memory requirements. The MVP therefore keeps model calls behind interfaces and starts with deterministic mock providers. This validates sampling, thresholds, escalation, structured output, and logging on a laptop before GPU or paid API work begins.

Three frames are rendered into an ordered contact sheet for Jev-Omni's image input. The stronger VLM receives the original frames through a separate adapter. A low-motion window is normally skipped, but every Nth window is still classified so stationary anomalies are not permanently invisible.

## Project layout

```text
src/jev_monitor/
  api.py          HTTP service: health, analyze one 3-frame window, recent events
  cli.py          runnable demo and folder-stream commands
  config.py       environment configuration
  domain.py       typed pipeline inputs and outputs
  ports.py        provider and logging interfaces
  pipeline.py     orchestration and escalation policy
  adapters.py     motion gate, contact sheet, mock/HTTP models, JSONL log
tests/
  test_pipeline.py
```

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env

jev-monitor demo
pytest
uvicorn jev_monitor.api:app --reload --port 8000
```

Analyze three frames over HTTP:

```bash
curl -X POST http://localhost:8000/v1/analyze-window \
  -F frame1=@frame-001.jpg \
  -F frame2=@frame-002.jpg \
  -F frame3=@frame-003.jpg
```

Or process an ordered image directory as a stream:

```bash
jev-monitor run-folder ./sample-frames
```

## Provider contracts

Set `JEV_MONITOR_JEV_PROVIDER=http` to call a Jev inference service. The MVP contract is:

```text
POST /v1/classify (multipart/form-data)
  state: string
  question: string
  options_json: '["safe", "suspicious"]'
  media: contact-sheet.jpg

200 OK
  {"probabilities": {"safe": 0.12, "suspicious": 0.88}}
```

The service wrapper can host `load_jev_omni()` on a CUDA machine or translate to another hosted inference provider.

Set `JEV_MONITOR_VLM_PROVIDER=http` for the strong model. The generic contract is:

```text
POST /v1/analyze
{
  "frames": [{"mime_type": "image/jpeg", "data_base64": "..."}],
  "context": {"camera_id": "front-door", "captured_at": "..."},
  "response_schema": {"...": "JSON Schema supplied by this app"}
}

200 OK
{
  "anomaly_type": "fall",
  "severity": "high",
  "summary": "A person appears to fall between frames 1 and 3.",
  "confidence": 0.91,
  "evidence": ["standing in frame 1", "on floor in frame 3"],
  "recommended_action": "notify"
}
```

## API surface

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Process health and active provider names |
| `POST /v1/analyze-window` | Analyze exactly three ordered images |
| `GET /v1/events?limit=50` | Read recent structured events |

## MVP policy

- Motion below the threshold: log `ignored`, except periodic forced classification.
- Jev suspicious probability below 0.70: log `safe`.
- Jev suspicious probability at or above 0.70: call the strong VLM.
- Strong VLM `critical`: `alert_immediately`.
- Strong VLM `high`: `notify`.
- Strong VLM `medium`: `queue_review`.
- Strong VLM `low` or `none`: `log_only`.

Before production, collect labeled windows and tune both thresholds for recall, calibration, camera placement, lighting, and the actual anomaly definitions.

