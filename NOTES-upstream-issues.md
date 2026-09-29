# TensorFold — upstream issues triage (parked 2026-09-29)

Branch: `docs/issue-97-simd-qmm-prefill` (commit `310227f`, pushed).
Fork = upstream at 0.5.0 (`9cd52ab`). Do NOT run MLX tests on this Linux box —
`mlx` needs Apple Metal; all kernel tests skip/fail on import. Baseline of
mlx-less failures was confirmed identical on the unmodified tree.

## Already resolved upstream — candidates to close

### #87 — object-typed tool params returned as JSON strings
Fixed in 0.5.0 by `closed_json()` + typed `decode_parameter()` in
`src/tensorfold/tool_parameters.py`; regression test
`tests/test_tool_parameter_types.py` is tagged `#87`. Verified: the exact repro
(one-closer-short `job` object from the issue) returns a `dict` through all
three parse paths (HTTP `parse_tool_calls_from_content`, CUDA
`cuda/reply_text.parse_tool_calls`, streaming `ToolCallStreamer`). Repro
script: `/tmp/repro87.py` (not committed). Nothing to implement.

## Committed

### #97 — `simd_qmm.install()` has no callers; prefill runs stock `mx.quantized_matmul`
The reporter's own follow-up settles it: simd_qmm LOSES to stock MLX at prefill
row counts (0.79x @1K rows, 0.57x @4K, M2 Max), so wiring it in would regress
prefill ~20%. The split is intentional — decode goes through `row_matmul`
(simd_qmm), prefill keeps `mx.quantized_matmul` for bit-identical chunks.
Docs-only commit on the branch:
- `simd_qmm.py` `install()` docstring: states serving never calls it and why
- `docs/quantization.md`: M1-M4 decode-vs-prefill note
- `CHANGELOG.md`: `## Unreleased` entry
Optional follow-up: post a short upstream comment on #97 pointing at the
docstring + docs note; nothing else needed.

## Parked — real work, needs hardware or is feature-sized

### #98 — CUDA prefix retention never engages (DGX Spark)
NOT missing machinery: `FlashNextEngine`
(`src/tensorfold/families/qwen4_exp/cuda/engine.py`) has `_resume`, `_decode`,
`_remember`, kept-state cache; multi-stream path has `_slot_for`/`kept` in
`cuda/multi.py`. Reporter ran `--parallel 4` (multi path) and saw zero reuse
and zero snapshot/prefix log lines. Suspects to check first:
1. multi path (`--parallel`): does `Scheduler.submit` → `MultiDecoder.admit`
   → `_slot_for` actually get kept states across requests, or does
   `finish()` free the slot before the next turn arrives?
2. `--prompt-cache-gib` / `--checkpoint-slots` / `--snapshot-dir` flags: do
   they reach the CUDA engine at all (they're parsed in `server/app.py`; CUDA
   engine builds its own `cache` list — maybe the app-level CheckpointStore is
   MLX-only wiring)?
3. Report is 0.3.6.3; re-test on 0.5.0 before anything.
Needs a Spark (or any CUDA GPU ≥ 48 GB) to reproduce. Cheap control test in
the issue: resend byte-identical context, compare prefill wall time.

### #85 — mixed 4/8-bit MLX checkpoints for Flash Next (qwen4_exp)
Feature-sized. Already exists: mixed-width Metal expert kernels
(`flash_next/v1/experts.py`: `_AFFINE_EXPERTS`, `_format`, per-width WB/WG
template params), per-module spec parsing (`quantization.resolve_affine`).
Missing pieces:
- `families/qwen4_exp` loader still assumes uniform format
  (`check_quantization` for CUDA refuses anything but (4, 32); MLX side reads
  per-module specs via `quant_params` — partially there)
- fused-stack rule when experts disagree in bits/group (cf. dense
  `row_matmul.Stack` requires one format; `decode._stacked` splits by bits)
- expert reader/writer path in `cuda/weights.py` packs group-32 4-bit only
Upstream reporter offers format details (ddalcu / MTPLX layouts). This is a
multi-day kernel + loader change, not a quick win.

## Misc
- `/tmp/repro87.py`, `/tmp/probe87.py` are throwaway; safe to delete.
- `.venv/` in the repo root is a local build with only `pytest` + non-MLX deps.
