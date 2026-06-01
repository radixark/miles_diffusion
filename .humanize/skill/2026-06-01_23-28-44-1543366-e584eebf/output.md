**Findings**
[P1] Ring backward path will crash in this environment.  
[usp.py](</workspace/809a2940-8360-4812-81c2-c7383f3f43e7/sglang/python/sglang/multimodal_gen/runtime/layers/usp.py:219>) imports `_templated_ring_attention_backward` from `torch.distributed.tensor.experimental._attention`, but local `torch==2.11.0+cu130` does not export it there. The symbol exists under `torch.distributed.tensor.experimental._context_parallel._attention`. Any training run with `ring_degree > 1` that reaches backward will fail before validating the math. The inference path is still unchanged for `no_grad`, but the training Ring fix is not currently runnable.

[P1] Stage 4 weight-sync grouping is only correct under a stricter rollout topology than the code enforces.  
[diffusion_update_weight_utils.py](</workspace/809a2940-8360-4812-81c2-c7383f3f43e7/miles_diffusion/miles/backends/fsdp_utils/diffusion_update_weight_utils.py:116>) groups train ranks as `i * rollout_num_gpus_per_engine : (i+1)*...`, while rollout engine creation uses `min(rollout_num_gpus_per_engine, num_gpus_per_node)` for engine count in [rollout.py](</workspace/809a2940-8360-4812-81c2-c7383f3f43e7/miles_diffusion/miles/ray/rollout.py:467>). Counterexamples:
- `world=10`, `rollout_num_gpus_per_engine=4`: only groups `0..3` and `4..7`; ranks `8,9` never get `_ipc_gather_src`, then `update_bucket_weights` dereferences missing attrs.
- `rollout_num_gpus_per_engine > num_gpus_per_node`: rollout creates per-node engines, but updater groups by full per-engine TP size, so later groups can point outside train world or to the wrong engine.
For valid colocate configs where `world == rollout_num_gpus`, `world % rollout_num_gpus_per_engine == 0`, and engine creation uses the same TP size, `gather_object` order and `_select_rank_scoped_payload[tp_rank]` do align.

[P2] The checksum algorithm is symmetric, but `/get_weights_checksum` is not symmetric for TP-sharded rollout.  
Training hashes full reconstructed tensors in [diffusion_update_weight_utils.py](</workspace/809a2940-8360-4812-81c2-c7383f3f43e7/miles_diffusion/miles/backends/fsdp_utils/diffusion_update_weight_utils.py:253>). SGLang hashes `iter_materialized_weights(module)` in [gpu_worker_post_training_mixin.py](</workspace/809a2940-8360-4812-81c2-c7383f3f43e7/sglang/python/sglang/multimodal_gen/runtime/post_training/gpu_worker_post_training_mixin.py:113>), and Wan uses TP-sharded `ColumnParallelLinear` / `RowParallelLinear` from [wanvideo.py](</workspace/809a2940-8360-4812-81c2-c7383f43e7/sglang/python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py:156>). With `rollout_num_gpus_per_engine > 1`, the live checksum is for a local TP shard, while the expected checksum is full weight, so `MILES_VERIFY_WEIGHT_SYNC=1` can report a stable mismatch even when loading is correct. LoRA name remapping itself remains correct because it hashes the remapped names.

**Validated**
Stage 3 SUM, not mean, is mathematically correct with the current gather point. `proj_out` runs on local sequence, gather happens after it, and `_GatherSequence.backward` returns only the local token slice. Each SP rank has the full-loss local token contribution; SP all-reduce must be SUM.

Stage 4’s core Option B premise is also correct for valid topology: FSDP is applied on `dp_mesh`, so SP ranks hold replicated params after `redistribute([Replicate()])`. The issue is topology contract enforcement/mismatch, not SP rank math.

**Tests**
I could not run GPU parity tests here. I tried `pytest tests/test_sp_mesh.py`; collection failed because importing `miles.backends.fsdp_utils.sp_mesh` executes package `__init__`, which imports `actor.py` and requires missing `ray`.
