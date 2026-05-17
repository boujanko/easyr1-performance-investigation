# Protocol Changes Performance Investigation

This branch is dedicated to **protocol-related performance regression investigation**, suspected after commit `098931530606d22f867fd121b1dcb3225a43661f`.

## Suspect commit
- SHA: `098931530606d22f867fd121b1dcb3225a43661f`
- Message: `fix: data proto`

## What changed (performance-relevant)
### `verl/protocol.py`
- Added **custom pickle serialization** for `DataProto` via `__getstate__` / `__setstate__`.
- On serialization:
  ```python
  batch_to_save = self.batch.contiguous()
  batch_to_save = batch_to_save.consolidate()
  ```
  This may cause extra copies when passing `DataProto` through Ray.
- On deserialization:
  ```python
  batch = torch.load(..., map_location="cpu")
  ```
  forcing tensors onto CPU before any transfer.

### `examples/config.yaml`
Introduced new data/training options that may change performance characteristics:
- `mini_rollout_batch_size`
- `val_batch_size`
- `shuffle`
- `override_chat_template`
- `filter_overlong_prompts`

These can affect:
- Number of rollout iterations
- Padding length
- CPU/GPU load during data prep
- Load imbalance

## Hypotheses
1. **Extra copies / CPU transfers in Ray object store**
   - `contiguous()` + `consolidate()` on every serialization adds latency.
   - `map_location="cpu"` on deserialization forces CPU staging before GPU use.
   - Manifested as increased time in:
     - `ray.put`
     - `ray.get`
     - `DataProtoFuture.get`

2. **More frequent rollout iterations**
   - When `mini_rollout_batch_size &lt; rollout_batch_size`, more rollout workers are launched.
   - May increase overhead of Ray task submission and inter-worker communication.

3. **Longer prompts / more padding**
   - Changed prompt handling (chat templates, filtering) can increase token count and padding.
   - Leads to higher:
     - KV-cache usage
     - Compute time per generation step
     - Memory pressure

## Experiment plan
Each experiment should be run **twice**:
- on the **baseline commit immediately before `0989315`** (e.g. `c5890a7`), and
- on `0989315` or later

Use **the same hardware, software versions, and config** for both runs.

### Experiment 1: Ray object store micro-benchmark
**Objective:** Measure impact of `DataProto` serialization changes.

**Method:**
```python
import timeit
from verl.protocol import DataProto
import torch

dp = DataProto.from_dict(
    tensors={"ids": torch.randint(0, 1000, (8192, 64))},
    non_tensors={"a": [1,2,3]},
    meta_info={"b": 42},
)

# Test put latency/throughput
timeit.timeit(lambda: ray.put(dp), number=200)
# Test get latency/throughput
ref = ray.put(dp)
timeit.timeit(lambda: ray.get(ref), number=200)
```

**Metrics to capture:**
- `ray.put` latency (p50/p95)
- `ray.get` latency (p50/p95)
- Throughput (items/sec)
- CPU utilization during benchmark

### Experiment 2: Rollout throughput comparison
**Objective:** See if rollout performance degraded.

**Method:**
- Run the same workload on both commits, **without** `mini_rollout_batch_size` (or set equal to `rollout_batch_size`).
- Measure:
  - Time spent in rollout phase
  - Tokens generated / sec
  - Time per generation step
  - Number of Ray tasks and total task time

**Success criterion for regression:**
- ≥ **5-10%** increase in rollout time OR drop in tokens/sec, with similar output quality.

### Experiment 3: Effect of `mini_rollout_batch_size`
**Objective:** Understand whether smaller rollout batches introduced overhead.

**Method:**
- On the suspect commit, run with:
  - `mini_rollout_batch_size = rollout_batch_size`
  - `mini_rollout_batch_size &lt; rollout_batch_size` (as in reported regressions)
- Compare:
  - Wall-clock time
  - Rollout time
  - Number of Ray tasks
  - CPU time in driver / workers

### Experiment 4: Data / prompt processing comparison
**Objective:** Detect changes in prompt length, padding, or data transfer.

**Method:**
- Instrument the code to log per-batch:
  - Prompt token count (before/after template)
  - Response token count
  - Padding tokens
  - DataProto size in bytes
- Compare distributions between baseline and suspect commit.

**Success criterion for regression:**
- Statistically significant increase in:
  - Prompt length
  - Padding fraction
  - DataProto byte size

## Reporting format
Please post results back to the main issue with:
- Commit SHA tested
- Full config (YAML + CLI overrides)
- Hardware + software versions
- Metrics (with min/mean/median/p50/p95/max) and number of samples

## Next steps
After experiments:
1. Identify the root cause (protocol serialization vs. rollout orchestration vs. data).
2. Propose a fix (e.g., avoid `map_location="cpu"`, cache contiguous buffers, tune batch sizes).
3. Validate the fix with the same experiments.
