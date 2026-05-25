# vLLM-Omni Layered Architecture

This document maps out the layers a request passes through inside vLLM-Omni, from the user's Python call down to a GPU kernel. It is meant as a navigation aid: if you know what concern you're working on, you should be able to jump straight to the right file.

## Overview

vLLM-Omni is organized as **six concentric layers**. Each layer owns one concern and hides it from the layer above.

```
┌─────────────────────────────────────────────────────────────┐
│ 1. ENTRYPOINTS         Omni / AsyncOmni / OpenAI server     │  ← user-facing
│   (what the user calls)                                      │
├─────────────────────────────────────────────────────────────┤
│ 2. ASYNCOMNIENGINE     proxy: config, spawn, IPC queues     │  ← lifecycle owner
│   (one per process)                                          │
├─────────────────────────────────────────────────────────────┤
│ 3. ORCHESTRATOR        request routing across stages         │  ← coordinator
│   (one asyncio loop, one thread)                             │
├─────────────────────────────────────────────────────────────┤
│ 4. STAGEPOOL           multi-replica fan-out for one stage   │  ← load balancer
│   (one per logical stage)                                    │
├─────────────────────────────────────────────────────────────┤
│ 5. STAGECLIENT         IPC handle to one worker process      │  ← transport
│   (one per replica)                                          │
├─────────────────────────────────────────────────────────────┤
│ 6. WORKER PROCESS      pipeline.forward() on the GPU         │  ← compute
│   (DiffusionModelRunner / EngineCore, in its own process)    │
└─────────────────────────────────────────────────────────────┘
         ▼ direction of a request
         ▲ direction of an output
```

The key invariant: **each layer only talks to the one directly below it.** The user never sees a `StagePool`. The orchestrator never sees a worker process. This is what makes the codebase navigable once you internalize it.

---

## Layer 1 — Entrypoints

What the user actually imports and calls.

| Class | File | Role |
|---|---|---|
| `Omni` | [vllm_omni/entrypoints/omni.py](../../../vllm_omni/entrypoints/omni.py) | Sync offline (`omni.generate(...)`) |
| `AsyncOmni` | [vllm_omni/entrypoints/async_omni.py](../../../vllm_omni/entrypoints/async_omni.py) | Async (`async for x in omni.generate(...)`) |
| `OmniServingChat` | [vllm_omni/entrypoints/openai/serving_chat.py](../../../vllm_omni/entrypoints/openai/serving_chat.py) | OpenAI-compatible HTTP endpoint |

**What they own:** input-shape conversion (user dict / OpenAI JSON → `OmniDiffusionRequest`).

**Where they delegate:** each holds one `AsyncOmniEngine` and forwards to its `add_request` / `try_get_output`.

`Omni` is just `AsyncOmni` with the async hidden by blocking polls.

---

## Layer 2 — AsyncOmniEngine

`class AsyncOmniEngine` at [vllm_omni/engine/async_omni_engine.py](../../../vllm_omni/engine/async_omni_engine.py).

**What it owns:**
- Stage config resolution (YAML or synthesized default for a single diffusion stage)
- Spawning the orchestrator thread + worker processes
- Three `janus.Queue`s: `request_queue`, `output_queue`, `rpc_output_queue`
- Caller-thread input preprocessing (tokenization for LLM stages, passthrough for diffusion)

**Why it exists:** to separate "things that need the user's Python interpreter" (input processing, lifecycle) from "things that need an asyncio loop" (routing, polling sockets). It is the **proxy** that sits between the user's thread and the orchestrator's thread.

**What it does *not* do:** per-request routing, scheduling, or model execution. Everything per-request is handed off through the IPC queues.

---

## Layer 3 — Orchestrator

`class Orchestrator` at [vllm_omni/engine/orchestrator.py](../../../vllm_omni/engine/orchestrator.py). Runs in a daemon thread with its own asyncio loop.

**What it owns:**
- `self.request_states: dict[str, OrchestratorRequestState]` — the single source of truth for in-flight requests
- A list of `StagePool`s (one per logical stage)
- The PD-disaggregation and CFG-companion state machines

**What it does:**
- **Inbound** (`_request_handler`): dispatches `request_queue` messages by type (`add_request`, `abort`, `streaming_update`, `collective_rpc`, …)
- **Outbound** (`_orchestration_loop`): polls every `StagePool` in round-robin and routes each output — either to the next stage in the pipeline, or back to the user via `output_queue`

**Mental model:** the orchestrator is a **router**. It moves requests through stages. It has no concept of "GPU" or "worker process" — those are abstracted away by `StagePool`.

---

## Layer 4 — StagePool

`class StagePool` at [vllm_omni/engine/stage_pool.py](../../../vllm_omni/engine/stage_pool.py). **One per logical stage.**

**Why it exists:** a "stage" is a logical thing (e.g. "the diffusion stage"). A stage can have N data-parallel replicas. The pool hides that fan-out from the orchestrator.

**What it owns:**
- `self.clients: list[StageClient]` — one client per replica
- A `LoadBalancer` (random, round-robin, etc.)
- An optional distributed hub for cross-process replicas
- An `OutputProcessor` (for LLM stages — detokenizes, builds `RequestOutput`s)

**Key methods called by the orchestrator:**
- `submit_initial(...)` — admit a new request: pick a replica → call that replica's `add_request_async`
- `poll_diffusion_output(replica_id)` — non-blocking recv from one diffusion replica's socket
- `select_replica_id(request_id)` — deterministic affinity (for cache reuse / companion routing)

**Mental model:** if `Orchestrator` is "what request goes where," `StagePool` is "which replica of that stage handles it."

---

## Layer 5 — StageClient

The interface: `class StageClient(Protocol)` at [vllm_omni/engine/stage_client.py](../../../vllm_omni/engine/stage_client.py). Concrete implementations:

- `StageDiffusionClient` at [vllm_omni/diffusion/stage_diffusion_client.py](../../../vllm_omni/diffusion/stage_diffusion_client.py) — ZMQ to a diffusion worker
- `StageEngineCoreClient` — vLLM's engine-core async MP client to an LLM worker

**What it owns:** the ZMQ socket pair (request socket out, response socket in) for **one worker process**, plus the metadata about what that worker does (stage type, supported sampling params).

**Key methods:**
- `add_request_async(request_id, prompt, params)` — serialize + send via ZMQ
- `get_output()` / receiver coroutine — pull serialized response off the socket

**Mental model:** a `StageClient` is what the worker process looks like from inside the orchestrator's process. It is a remote-procedure-call stub. No model code, no GPU work — just serialization and socket I/O.

---

## Layer 6 — Worker process

Spawned by `mp.Process` from `spawn_diffusion_proc(...)` (called inside `_initialize_diffusion_replica`).

Inside a diffusion worker, the actors are:
- `DiffusionWorker` at [vllm_omni/diffusion/worker/diffusion_worker.py](../../../vllm_omni/diffusion/worker/diffusion_worker.py) — the process's main loop: recv from socket → invoke model runner → send result
- `DiffusionModelRunner` at [vllm_omni/diffusion/worker/diffusion_model_runner.py](../../../vllm_omni/diffusion/worker/diffusion_model_runner.py) — owns `self.pipeline`; `execute_model(req)` calls `pipeline.forward(req)`
- The actual model — `QwenImagePipeline`, `WanPipeline`, etc.

For LLM workers it is vLLM's `EngineCore` running in a subprocess — same idea, different model code.

**This is the only layer that touches CUDA.**

---

## End-to-end: one diffusion request through all six layers

```
USER (Layer 1)
  omni.generate([{"prompt": "..."}], OmniDiffusionSamplingParams(...))
   │
   ▼ method call
ASYNCOMNIENGINE (Layer 2)
  _build_add_request_message()  ← packs into StageSubmissionMessage
  request_queue.sync_q.put_nowait(msg)
   │
   ▼ janus queue (thread → thread)
ORCHESTRATOR (Layer 3)
  _request_handler awaits msg
  _handle_add_request(msg)
    → creates OrchestratorRequestState[req_id]
    → await stage_pools[0].submit_initial(...)
   │
   ▼ method call (same loop)
STAGEPOOL (Layer 4)
  submit_initial(...)
    → load_balancer.pick() → replica_id
    → await clients[replica_id].add_request_async(...)
   │
   ▼ method call
STAGECLIENT (Layer 5)
  StageDiffusionClient.add_request_async(...)
    → serialize the request
    → ZMQ socket.send(serialized)
   │
   ▼ ZMQ socket (process → process)
WORKER (Layer 6)
  DiffusionWorker main loop: recv()
    → DiffusionModelRunner.execute_model(req)
      → pipeline.forward(req)   ← actual GPU work
    → DiffusionOutput(images)
   ▲
   │ ZMQ socket.send(serialized result)
STAGECLIENT (Layer 5)
   ▲ socket recv in background; deserialized output buffered
   │
   ▲
STAGEPOOL (Layer 4)
  Orchestrator polls: poll_diffusion_output(replica_id) → output
   │
   ▲
ORCHESTRATOR (Layer 3)
  _orchestration_loop catches it
  _route_output: pool.final_output==True
    → output_async_queue.put(OutputMessage(...))
   │
   ▲ janus queue (thread → thread)
ASYNCOMNIENGINE (Layer 2)
  try_get_output() returns the OutputMessage
   │
   ▲
USER (Layer 1)
  omni.generate() returns list[OmniRequestOutput]
  user does output.images[0].save("out.png")
```

---

## Communication mechanisms

Different layer boundaries use different transports, depending on what they cross.

| Boundary | Transport | Why |
|---|---|---|
| Layer 1 → 2 (entrypoint → engine) | Plain method call | Same object, same thread |
| Layer 2 → 3 (engine → orchestrator) | `janus.Queue` | Crosses **thread** boundary (caller thread → orchestrator thread). Janus has a sync side and an async side that talk to each other. |
| Layer 3 → 4 (orchestrator → pool) | Plain `await` method call | Same asyncio loop |
| Layer 4 → 5 (pool → client) | Plain `await` method call | Same loop |
| Layer 5 → 6 (client → worker) | ZMQ sockets | Crosses **process** boundary. Workers are `mp.Process` children — they need a real wire protocol. |
| In the worker, model runner → GPU | CUDA / PyTorch | Standard |

Three different transports, picked because of three different boundary types: same-thread, cross-thread, cross-process. Each one is plumbing — no per-request logic lives in the transport itself.

---

## Why these particular layers?

Each layer earns its existence by hiding a single concern from the one above:

| Layer | Hides from above |
|---|---|
| Entrypoints | "is the engine sync or async?" |
| AsyncOmniEngine | "where do the worker processes come from?" |
| Orchestrator | "are there multiple stages in a pipeline?" |
| StagePool | "are there multiple replicas of this stage?" |
| StageClient | "what is the wire protocol to talk to a worker?" |
| Worker | "how does the model actually run on the GPU?" |

Delete any layer and the one above it would have to absorb that concern: the entrypoints would need to know about multiprocess spawning; the orchestrator would need to know about ZMQ; and so on. Each layer is justified by the abstraction it provides.

---

## Cheat sheet for navigating the code

When you are reading code and see something happening, ask **"which layer is this?"** to know where to look next.

| Question | Layer | Where to look |
|---|---|---|
| How is `omni.generate` argument shape validated? | 1 | [entrypoints/omni.py](../../../vllm_omni/entrypoints/omni.py) |
| When is tokenization applied? | 2 | `_build_add_request_message` in [async_omni_engine.py](../../../vllm_omni/engine/async_omni_engine.py) |
| How does an AR-stage output get to the DiT stage? | 3 | `_forward_to_next_stage` in [orchestrator.py](../../../vllm_omni/engine/orchestrator.py) |
| How is a request load-balanced across N replicas? | 4 | `StagePool.submit_initial` in [stage_pool.py](../../../vllm_omni/engine/stage_pool.py) → load balancer |
| What is the wire format on the ZMQ socket? | 5 | [stage_diffusion_client.py](../../../vllm_omni/diffusion/stage_diffusion_client.py) |
| How does the pipeline actually run? | 6 | `DiffusionModelRunner.execute_model` in [diffusion_model_runner.py](../../../vllm_omni/diffusion/worker/diffusion_model_runner.py) |

After a while, you stop thinking about layers — you navigate directly to the right file based on what concern the question lives at.
