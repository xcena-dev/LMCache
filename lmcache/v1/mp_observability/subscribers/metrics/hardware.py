# SPDX-License-Identifier: Apache-2.0

"""Hardware observability gauges — host DRAM, GPU (NVML), and CXL device.

These observable gauges expose host- and device-level telemetry alongside the
LMCache software metrics, so a single dashboard can correlate the KV-cache
workload with the hardware it runs on (memory occupancy, GPU utilisation, PCIe
DMA, power draw).

They are registered through :func:`register_gauge`, so they flow the same
OTel MeterProvider -> OTLP/Prometheus pipeline as the L1/L2 counters and
inherit the MP server's ``service.instance.id`` resource attribute (which
distinguishes backends, e.g. ``mp-dram`` vs ``mp-cxl``).

All reads are best-effort. Missing tools (``pynvml``, ``xcena_cli``),
unavailable devices, or permission errors degrade to *no datapoints* for that
gauge rather than raising — a monitoring gauge must never crash the server.

Attribution caveats (important for interpretation):
    - ``host_dram_used_bytes`` is whole-NUMA-node occupancy from ``/sys`` —
      it includes page cache and every tenant, so it is NOT KV-attributed.
    - ``gpu_pcie_*_bytes_per_second`` is the GPU's total PCIe traffic; it
      cannot distinguish a DRAM-sourced from a CXL-sourced DMA (both cross the
      same link). It confirms "same PCIe lane", not "which memory fed it".
      Source attribution requires host iMC / CXL.mem uncore counters.
    - ``cxl_capacity_bytes`` is device capacity, not live usage.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
import glob
import re
import subprocess

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.otel_init import register_gauge

logger = init_logger(__name__)

# A gauge callback that reports one datapoint per attribute set.
GaugeCallback = Callable[[], list[tuple[float, dict[str, object]]]]


def _host_dram_used_bytes() -> list[tuple[float, dict[str, object]]]:
    """Report used DRAM per NUMA node from ``/sys`` (bytes).

    Returns:
        One ``(used_bytes, {"node": <id>})`` tuple per NUMA node. Empty when
        ``/sys`` node meminfo is unavailable.
    """
    out: list[tuple[float, dict[str, object]]] = []
    for path in sorted(glob.glob("/sys/devices/system/node/node*/meminfo")):
        match = re.search(r"node(\d+)", path)
        if match is None:
            continue
        total_kb = free_kb = 0
        try:
            with open(path) as handle:
                for line in handle:
                    if "MemTotal:" in line:
                        total_kb = int(line.split()[3])
                    elif "MemFree:" in line:
                        free_kb = int(line.split()[3])
        except (OSError, ValueError, IndexError):
            continue
        out.append((float((total_kb - free_kb) * 1024), {"node": match.group(1)}))
    return out


class _NvmlReader:
    """Lazy NVML handle holder for GPU gauges.

    NVML is initialised once on construction. When ``pynvml`` is absent or
    initialisation fails, ``handles`` is empty and every read returns no
    datapoints.
    """

    def __init__(self) -> None:
        self._nvml = None
        self._handles: list[object] = []
        try:
            # Third Party
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i)
                for i in range(pynvml.nvmlDeviceGetCount())
            ]
            logger.info("Hardware gauges: NVML initialised, %d GPU(s)", len(self._handles))
        except Exception as exc:  # noqa: BLE001 — NVML absence must not raise
            logger.info("Hardware gauges: NVML unavailable (%s); GPU gauges disabled", exc)

    def read(self, metric: str) -> list[tuple[float, dict[str, object]]]:
        """Read one GPU metric across all devices.

        Args:
            metric: One of ``util``, ``mem``, ``power``, ``pcie_rx``,
                ``pcie_tx``.

        Returns:
            One ``(value, {"gpu": <index>})`` tuple per GPU; empty when NVML
            is unavailable or the metric read fails on every device.
        """
        if self._nvml is None:
            return []
        nvml = self._nvml
        out: list[tuple[float, dict[str, object]]] = []
        for index, handle in enumerate(self._handles):
            attrs: dict[str, object] = {"gpu": str(index)}
            try:
                if metric == "util":
                    out.append((nvml.nvmlDeviceGetUtilizationRates(handle).gpu / 100.0, attrs))
                elif metric == "mem":
                    out.append((float(nvml.nvmlDeviceGetMemoryInfo(handle).used), attrs))
                elif metric == "power":
                    out.append((nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0, attrs))
                elif metric == "pcie_rx":
                    kb_s = nvml.nvmlDeviceGetPcieThroughput(handle, nvml.NVML_PCIE_UTIL_RX_BYTES)
                    out.append((float(kb_s) * 1024.0, attrs))
                elif metric == "pcie_tx":
                    kb_s = nvml.nvmlDeviceGetPcieThroughput(handle, nvml.NVML_PCIE_UTIL_TX_BYTES)
                    out.append((float(kb_s) * 1024.0, attrs))
            except Exception:  # noqa: BLE001 — skip a device that fails to read
                continue
        return out


def _cxl_capacity_bytes() -> list[tuple[float, dict[str, object]]]:
    """Report XCENA CXL device capacity via ``xcena_cli`` (bytes).

    Returns:
        One ``(capacity_bytes, {"device": <id>, "target": <name>})`` tuple per
        device. Empty when ``xcena_cli`` is unavailable. Capacity is a static
        value; live usage is not exposed by a non-root interface.
    """
    out: list[tuple[float, dict[str, object]]] = []
    try:
        listing = subprocess.run(
            ["xcena_cli", "num-device"], capture_output=True, text=True, timeout=8
        ).stdout
        count_match = re.search(r"Number of devices\s*:\s*(\d+)", listing)
        if count_match is None:
            return out
        count = int(count_match.group(1))
    except (OSError, subprocess.SubprocessError, ValueError):
        return out
    for device in range(count):
        try:
            info = subprocess.run(
                ["xcena_cli", "device-info", str(device)],
                capture_output=True,
                text=True,
                timeout=8,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        cap_match = re.search(r"CXL Memory\s*:\s*(\d+)\s*GB", info)
        target_match = re.search(r"Target\s*:\s*(\S+)", info)
        if cap_match is None:
            continue
        out.append(
            (
                float(int(cap_match.group(1)) * (1024**3)),
                {"device": str(device), "target": target_match.group(1) if target_match else "?"},
            )
        )
    return out


def register_hardware_gauges() -> None:
    """Register host DRAM, GPU (NVML), and CXL observable gauges.

    Safe to call once at MP server startup after the OTel MeterProvider is
    set. Each gauge is best-effort: an unavailable source simply reports no
    datapoints. Never raises.
    """
    try:
        register_gauge(
            "lmcache.host",
            "lmcache_mp.host_dram_used_bytes",
            "Used host DRAM per NUMA node (whole-node, not KV-attributed)",
            _host_dram_used_bytes,
        )

        nvml = _NvmlReader()
        gpu_gauges: list[tuple[str, str]] = [
            ("lmcache_mp.gpu_utilization_ratio", "util"),
            ("lmcache_mp.gpu_memory_used_bytes", "mem"),
            ("lmcache_mp.gpu_power_watts", "power"),
            ("lmcache_mp.gpu_pcie_rx_bytes_per_second", "pcie_rx"),
            ("lmcache_mp.gpu_pcie_tx_bytes_per_second", "pcie_tx"),
        ]
        descriptions = {
            "util": "GPU utilization 0..1 (NVML)",
            "mem": "GPU memory used in bytes (NVML)",
            "power": "GPU power draw in watts (NVML)",
            "pcie_rx": "GPU PCIe inbound (H2D DMA) bytes/s (NVML)",
            "pcie_tx": "GPU PCIe outbound (D2H DMA) bytes/s (NVML)",
        }
        for gauge_name, metric in gpu_gauges:
            register_gauge(
                "lmcache.gpu",
                gauge_name,
                descriptions[metric],
                # Bind the current metric via default arg to avoid late binding.
                (lambda m=metric: nvml.read(m)),
            )

        register_gauge(
            "lmcache.cxl",
            "lmcache_mp.cxl_capacity_bytes",
            "CXL device capacity in bytes (xcena_cli; capacity, not live usage)",
            _cxl_capacity_bytes,
        )
        logger.info("Hardware observability gauges registered")
    except Exception as exc:  # noqa: BLE001 — registration must not break startup
        logger.warning("Hardware gauge registration failed: %s", exc)
