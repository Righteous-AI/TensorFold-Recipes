# Recipe 1 — Qwen3.8-27B, EXL3 6.0 bpw, DFlash2, 262,144 window

One dense 27B on one RTX 5090, drafted, at the model's full context.

| | |
| --- | --- |
| Engine | TensorFold v1.2.4, commit `267c5af6`, native CUDA server |
| Build | Zig 0.17.0, `zig build -j10 -Dgpu=sm_120 -Dnvcc=/usr/local/cuda-13.5/bin/nvcc` |
| Binary | `zig-out/native/bin/tensorfold-native`, 50,235,784 bytes, sha256 `100ac2c1a0eb231d1ba1dbdac16613b6cf83892571eeed1ace1c3147c4e8b868` |
| Weights | Qwen3.8-27B EXL3 6.0 bpw, 6-bit head, `mul1` codebook, 22,624,940,803 bytes |
| Drafter | `z-lab/Qwen3.8-27B-DFlash2` @ `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`, BF16 on disk |
| Endpoint | `0.0.0.0:12350`, model `qwen3.8-27b-exl3-6bpw`, key from `TENSORFOLD_KEY` (placeholder `change-me`) |

## Flags

```
--backend cuda
--parallel 1
--context 262144
--prompt-cache-gib 0
--profile stable
--kv-cache fp8
--drafter /mnt/models/qwen/Qwen3.8-27B-DFlash2
```

**`--backend cuda`.** The native CUDA server. `auto` picks the same thing on Linux.

**`--parallel 1`.** One stream. At the 262,144 window a stream costs 9.09 GiB, and only one fits beside the 19.91 GiB of weights, so a higher count would be refused rather than merely slower.

**`--context 262144`.** The model's own `max_position_embeddings`. This is the setting that pushes the embedding rows off the card: at this window the loader has no room for 2.54 GiB of bf16 embedding rows beside the weights, one stream and the drafter, so it maps them into pinned host memory and logs `embedding rows read from host memory`. Lower the window and the rows come back to the device. It is a real trade: full context against a resident embedding table.

**`--kv-cache fp8`.** The only choice on this build, apart from `bf16`, is `fp8`, which stores e4m3 rows with a scale per row at about half the bf16 cache. At the full window the 16 full-attention layers hold roughly 4.3 GiB at fp8, against about 8.6 GiB at bf16, and bf16 does not leave room for the stream at all. NVFP4 is a weight format in this engine and is refused for `--kv-cache`.

**`--prompt-cache-gib 0`.** No retained-prefix budget. Every prompt pays its own prefill, which is what a cold bench measures and what a single-agent workload with long unique prompts gets anyway.

**`--profile stable`.** The only profile this build ships.

**`--drafter`.** DFlash2 is the drafter for a dense target. The CUDA gate in `cuda_drafter_open.zig` opens the MTP path only for a MoE checkpoint that is not itself a DFlash configuration; a dense 27B always takes the DFlash branch, and there is no flag that routes it to MTP instead. The directory must hold the **BF16** DFlash2 model: the loader expects bf16 tensors and quantizes the projections itself to affine-4 with a group of 64. A pre-quantized drafter on disk is the wrong input, and the engine refuses it. `--drafter-bits` accepts only 4, which is the default, so leave it out.

## What is refused, and stays off

`--kv-cache nvfp4` and `--kv-cache fp4` exit with `invalid choice`. `--prefill-fp8` is not parsed by the native engine, and upstream states that EXL3 prompts never took FP8 activations, so there is no faster prompt lane for this pack. `--speed-up` is the two-link Mac path and is gated off for CUDA builds. `--slide` combined with fp8 KV hits `UnsupportedLearningFormat`. `--prompt-precision int8` is published against a different, MLX 4-bit weight, and the EXL3 prompt path is the pack's own GEMM, so it was left at `bf16`. `--ple-on-ssd`, `--ssd-experts` and `--tp` are not in this Linux build's help.

`serve --help` does not list `--kv-cache`, which is a gap in the usage string: the parser accepts it. Check a flag by running the server, not by reading `--help`.

## The cold bench, 2026-11-07 17:17 UTC (v1.2.4)

Five cold prompts, temperature 0, 256 tokens out, nothing reused, one pass on an idle card. `cached_tokens` 0 on all five, `finish=length` on all five, no errors.

| Prompt tokens | TTFT | Prefill | Decode | Rounds | Draft accepted | Tokens a round |
| --- | --- | --- | --- | --- | --- | --- |
| 8,244 | 4.823 s | 1,709.2 tok/s | 177.4 tok/s | 63 | 195/900 (21.7%) | 4.06 |
| 32,820 | 20.151 s | 1,628.7 tok/s | 153.8 tok/s | 68 | 188/931 (20.2%) | 3.76 |
| 131,124 | 96.157 s | 1,363.6 tok/s | 128.2 tok/s | 68 | 187/797 (23.5%) | 3.76 |
| 196,660 | 159.934 s | 1,229.6 tok/s | 119.9 tok/s | 66 | 190/699 (27.2%) | 3.88 |
| 249,908 | 219.516 s | 1,138.5 tok/s | 119.0 tok/s | 62 | 194/638 (30.4%) | 4.13 |

### The same pass on v1.2.3, 2026-11-07 12:10 UTC

Kept for the version comparison. Same card, same pack, same flags, same prompt files.

| Prompt tokens | TTFT | Prefill | Decode | Rounds | Draft accepted | Tokens a round |
| --- | --- | --- | --- | --- | --- | --- |
| 8,244 | 4.962 s | 1,661.4 tok/s | 177.4 tok/s | 63 | 195/900 (21.7%) | 4.06 |
| 32,820 | 20.721 s | 1,583.7 tok/s | 153.8 tok/s | 68 | 188/934 (20.1%) | 3.76 |
| 131,124 | 99.193 s | 1,321.9 tok/s | 128.2 tok/s | 68 | 187/798 (23.4%) | 3.76 |
| 196,660 | 165.363 s | 1,189.2 tok/s | 119.9 tok/s | 66 | 190/699 (27.2%) | 3.88 |
| 249,908 | 227.189 s | 1,100.0 tok/s | 122.7 tok/s | 60 | 196/622 (31.5%) | 4.27 |

v1.2.4 is 2.8 to 3.5 percent faster on prefill and time to first token at every length, which is the 1.2.4 last-layer tile change, and it returns byte-identical replies: the five finished lines carry the same reply hashes on both passes. Decode and draft acceptance match on the first four rows, same rounds and same accepted counts. [comparisons.md](../comparisons.md) has the row-by-row delta.

Reading the acceptance column: DFlash2 proposes about 14 tokens a round and keeps about 3, so the ratio looks low while the throughput does not. Tokens a round, 3.76 to 4.13, is the figure to compare, and it sits inside the range the engine publishes for this drafter on this model. Acceptance rises with context on this pass, 20.2% to 30.4%, because the fox-prompt text is more predictable once a long prefix is in the window.

Decode holds 119.0 to 177.4 tok/s across a 30-fold change in prompt length. Prefill falls from 1,709 to 1,138 tok/s over the same span, which is the attention term, and it is the EXL3 prompt GEMM rather than a misconfiguration: upstream measures the 3.00 bpw branch at 880–970 tok/s from 2k to 16k and 720–890 at 32k to 64k on a DGX Spark, and this 6.0 bpw pack on the 5090 is 1.3 to 1.9 times that.

Every reply was still reasoning when it reached `max_tokens`, so all 256 output tokens arrived as `reasoning_content`. Raise `max_tokens`, or send `chat_template_kwargs {"enable_thinking": false}`, when you want an answer rather than a chain of thought.

## Reproducing it

The client harness is a small Python script on the host (`bench.py` — it lives beside the bench logs, is not shipped in this repo, and any streaming client that sends the same five prompts reproduces the table): it streams `/v1/chat/completions`, sends five saved prompts of 8,192 to 249,856 user tokens built from repeated `" fox"` with a checkpoint tokenizer, and records the first-token time and the last-token time from the client clock. The prompts are deterministic, so two engines see byte-identical input.

```
.venv/bin/python -u bench.py 2>&1 | tee bench-$(date +%Y%m%d_%H%M).txt
```

Confirm the card is idle before a pass: a second engine on the GPU changes every number.

## Boot line

```
CUDA sm_120 device 0 (NVIDIA GeForce RTX 5090): model 19.91 GiB; 1 stream at once,
9.09 GiB each at a 262144-token window (fp8 caches), of 10.0 GiB left after a
1.0 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels
profile: stable; registered overrides: none
serving qwen3.8-27b-exl3-6bpw at http://0.0.0.0:12350/v1 (sampling: temperature 1.0,
top_k 20, top_p 0.95; drafts: DFlash2 (/mnt/models/qwen/Qwen3.8-27B-DFlash2) + copies;
context: 262144; loaded in 39.0s)
```

Startup phases on v1.2.4: `load=19.599s reserve=0.007s warmups=0.516s calibration=18.503s`, host memory peak 23.1 GB. The v1.2.3 boot read `loaded in 39.3s` with `load=19.788s reserve=0.018s warmups=0.528s calibration=18.528s`, and its memory line was character-identical to the one above.

`libtfgrammar.so` is absent on this build, so structured output and required tool calls return HTTP 400. Build it with `zig build grammar -Dxgrammar=DIR` against xgrammar v0.2.8 if you need those. Plain chat and drafted decoding do not need it.

## Keeping the build current

The engine moves quickly and the flags above have held across tags, so an update is a rebuild plus a bench. On the host that runs this recipe:

1. Confirm the card is idle and no other engine unit is holding it.
2. `git fetch origin tag vX.Y.Z` then `git checkout --detach vX.Y.Z`. Revert any local edit to `zig/build/product.zig` first: git carries a dirty file across a detach when the blob matches upstream, and from v1.2.4 the tree needs no local wire at all.
3. Copy the current binary aside, `zig-out/native/bin/.cuda13.5-vXYZ-backup/`, before you overwrite it.
4. Build with the command in the table above. `--version` on the new binary is the check that needs no model and no GPU: `CUDA_VISIBLE_DEVICES="" ./tensorfold-native --version`.
5. Re-check the flags by parsing them without a model, `CUDA_VISIBLE_DEVICES="" ./tensorfold-native serve /tmp/no-such-model --kv-cache fp8 --backend cuda`, which exits 1 on the missing path. A bad choice exits 2 with `invalid choice`. Read `serve --help` for shape only: it short-circuits before validating, and it omits `--kv-cache`.
6. Restart the server and run the bench on an idle card, then compare rounds, accepted and the reply hashes against the previous pass. Matching reply hashes are the cheapest proof that a release changed speed and not behaviour.

The serve script is the unit's `ExecStart`, so a flag edit needs no unit rewrite. A running process keeps the binary inode it opened, so a rebuild reaches the card only after the operator starts the unit again.
