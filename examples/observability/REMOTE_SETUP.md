# Bringing this observability stack up on another server

Everything needed lives in **this directory**. Copy it to the target machine and run
one command; the two ingestion paths below decide what you get.

```
[LMCache MP / naru exporters] --OTLP gRPC--> collector :4320->4317 --:8889--> Prometheus :9091 --> Grafana :3000
[vLLM /metrics  lmcache:*   ] <---------------- scraped directly by Prometheus ---┘        Tempo :3200 (traces)
```

## Quick start

```bash
scp -r examples/observability <server>:~/obs
ssh <server> 'cd ~/obs && docker compose up -d'
# Grafana  http://<server>:3000   (anonymous Admin, dashboards auto-provisioned)
# Prometheus http://<server>:9091 ; check targets:
curl -s localhost:9091/api/v1/targets | grep -o '"health":"[a-z]*"'
```

## The two ingestion paths

### (1) PULL — native `lmcache:*`, zero extra code (recommended baseline)

Prometheus scrapes each vLLM `/metrics` directly (`prometheus.yml` job
`vllm-lmcache-native`). This yields the metrics documented in the LMCache metrics
reference — `lmcache:retrieve_hit_rate`, `time_to_retrieve`, `retrieve_speed`,
`local_cache_usage`, `num_stored_tokens`, `num_p2p_*`, … — in **both MP and
in-process mode**, with no exporters to run.

Edit the targets for your layout:

```yaml
  - job_name: vllm-lmcache-native
    metrics_path: /metrics
    static_configs:
      - targets: ["host.docker.internal:9010"]   # same host as this stack
      # - targets: ["10.0.0.7:8000", "10.0.0.8:8000"]   # vLLM on other machines
```

then `docker compose restart prometheus`.

### (2) PUSH — the `lmcache_mp_*` family the demo dashboards use

* **MP mode**: LMCache's `mp_observability` subscribers push `lmcache_mp_*` natively.
  Point the server at this collector and the provisioned dashboards
  (`lmcache_demo_mp.json`, `lmcache_mp_hardware.json`, `lmcache_causal_demo.json`)
  light up with no extra process.
* **in-process / p2p mode**: there is **no MP server**, so nothing emits `lmcache_mp_*`
  natively. The naru side supplies it (`naru/tools/*_exporter.py` +
  `python -m naru.ttft_otel`), bridging `lmcache:*`/NVML/`/sys`/Maru into that schema.
  See `naru` branch `26-fms-demo`, `docs/fms-demo/README.md`.

OTLP endpoint for producers is `http://<obs-host>:4320` (gRPC, host port 4320 maps to
the collector's 4317).

## Ports / persistence

| service | host port | note |
|---|---|---|
| OTel collector | **4320** -> 4317 (OTLP gRPC), 8889 (scrape) | `extra_hosts: host.docker.internal` |
| Prometheus | **9091** -> 9090 | retention 7d, named volume `prometheus_data` |
| Grafana | **3000** | anonymous Admin, no login form |
| Tempo | 3200 | traces |

Volumes survive `docker compose down` (use `down -v` only to wipe history).

## Pitfalls that actually bit us — do not repeat

1. **Do not put `service.instance.id` in the OTLP *resource*.** The collector runs with
   `resource_to_telemetry_conversion: enabled`, so resource attributes become labels;
   a resource-level `service.instance.id` collided across exporters and the series was
   **dropped**. Keep it as a datapoint attribute.
2. **Never leave duplicate/backup dashboard JSON in `grafana/provisioning/dashboards/`.**
   Duplicate `uid`/title makes the provider refuse to write ("no database write
   permissions because of duplicates") and dashboards silently stop updating —
   a Grafana restart was needed to clear it. Keep backups outside this tree.
3. **One metric = one emitter.** Two exporter copies (e.g. a stale one left running)
   tagged the same GPUs with *different* `service_instance_id` values, so two
   experiment arms appeared to run concurrently. Add singleton guards, and write
   panel queries as `max by(service_instance_id)(<metric>{...})` so a stray series
   can never draw a duplicate line.
4. **Verify what a metric means before trusting a panel.** Examples from this repo:
   `lmcache_mp_host_dram_used_bytes` is whole-NUMA occupancy (page cache + model
   weights), *not* KV; `lmcache:num_stored_tokens` is a cumulative store-op counter
   (~100x the live KV); the Maru resource manager's pool `total-free` is the
   **pre-reserved** pool, while `MaruServer.get_stats()["kv_manager"]["total_size"]`
   is the KV actually stored.
5. **Port mapping is not 1:1** — Prometheus is `9091` on the host (9090 inside), the
   collector is `4320` on the host (4317 inside). Scrape configs use the *in-network*
   names/ports (`otel-collector:8889`, `prometheus:9090`).
6. **Remote scraping needs the ports open** on the vLLM machines (default 9010/9011
   here) and 4320 on the observability host for pushes.
