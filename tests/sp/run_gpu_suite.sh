#!/usr/bin/env bash
# Full SP GPU test suite (needs 4 GPUs). CPU tests: pytest tests/sp/test_sp_mesh.py
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.

torchrun() { python -m torch.distributed.run "$@"; }

run() { echo; echo "===== $* ====="; "$@"; }

run python -m pytest -q tests/sp/sglang_usp_import_guard.py

run torchrun --standalone --nproc_per_node=4 tests/sp/sp_init_smoke.py --sp 2
run torchrun --standalone --nproc_per_node=4 tests/sp/sp_init_smoke.py --sp 2 --shard-mode dp
run torchrun --standalone --nproc_per_node=4 tests/sp/sp_init_smoke.py --sp 4 --ulysses 4
run torchrun --standalone --nproc_per_node=4 tests/sp/sp_init_smoke.py --sp 4 --ulysses 2 --ring 2

run torchrun --standalone --nproc_per_node=2 tests/sp/sp_attention_parity.py --sp 2
run torchrun --standalone --nproc_per_node=2 tests/sp/sp_attention_parity.py --sp 2 --ckpt
run torchrun --standalone --nproc_per_node=4 tests/sp/sp_attention_parity.py --sp 4 --ulysses 4
run torchrun --standalone --nproc_per_node=4 tests/sp/sp_attention_parity.py --sp 4 --ulysses 4 --ckpt
run torchrun --standalone --nproc_per_node=4 tests/sp/sp_attention_parity.py --sp 4 --ulysses 2 --ring 2
run torchrun --standalone --nproc_per_node=4 tests/sp/sp_attention_parity.py --sp 4 --ulysses 2 --ring 2 --ckpt

run torchrun --standalone --nproc_per_node=4 tests/sp/sp_grad_sync_parity.py --shard-mode dp
run torchrun --standalone --nproc_per_node=4 tests/sp/sp_grad_sync_parity.py --shard-mode dp_sp
run torchrun --standalone --nproc_per_node=4 tests/sp/sp_grad_sync_parity.py --shard-mode dp_sp --sp 4 --ulysses 2 --ring 2
run torchrun --standalone --nproc_per_node=4 tests/sp/sp_grad_sync_parity.py --shard-mode dp_sp --sp 2 --ulysses 2

run torchrun --standalone --nproc_per_node=4 tests/sp/sp_weight_sync_parity.py --sp 2 --ulysses 2 --shard-mode dp
run torchrun --standalone --nproc_per_node=4 tests/sp/sp_weight_sync_parity.py --sp 2 --ulysses 2
run torchrun --standalone --nproc_per_node=4 tests/sp/sp_weight_sync_parity.py --sp 4 --ulysses 4 --shard-mode dp
run torchrun --standalone --nproc_per_node=4 tests/sp/sp_weight_sync_parity.py --sp 4 --ulysses 4
run torchrun --standalone --nproc_per_node=4 tests/sp/sp_weight_sync_parity.py --sp 4 --ulysses 2 --ring 2 --shard-mode dp
run torchrun --standalone --nproc_per_node=4 tests/sp/sp_weight_sync_parity.py --sp 4 --ulysses 2 --ring 2

echo; echo "ALL SP GPU TESTS PASSED"
