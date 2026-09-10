# SPDX-License-Identifier: Apache-2.0
"""Measurement instrumentation: thread/context-local prefetch id.

The prefetch controller sets the in-flight prefetch request id around
``submit_load_task`` so the transfer channel can tag its per-read QoS line
(``P2P-READ-QOS ... pf=<id>``) without changing any interface. The prefetch
id is joined to the engine request id via the ``Prefetch request submitted``
/ ``completed`` lines, which carry both.
"""
import contextvars

current_prefetch_id: contextvars.ContextVar[int] = contextvars.ContextVar(
    "lmcache_current_prefetch_id", default=-1
)
