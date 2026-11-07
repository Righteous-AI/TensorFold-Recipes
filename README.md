# TensorFold recipes — Qwen3.8-27B EXL3 on an RTX 5090

**Hardware:** ASUS ProArt Z890 · Intel Core Ultra 7 265K · 128 GB DDR5-5200 · RTX 5090 32 GB · Samsung 9100 PRO
**Engine:** [TensorFold](https://github.com/ashhart/TensorFold) v1.2.4, commit `267c5af6`, native CUDA server, built for sm_120 on CUDA 13.5
**What is running:** [recipe 1](recipes/01-exl3-6bpw-dflash2.md), Qwen3.8-27B in EXL3 6.0 bpw with the DFlash2 drafter, 262,144-token window
**Measured:** 2026-11-07, two passes on the same card: v1.2.3 at 12:10 UTC and v1.2.4 at 17:17 UTC

One server, one card, one recipe. A dense 27B quantized to 6.0 bits per weight, drafted with DFlash2, at the model's full 262,144-token window. This is a different machine from the [Strata-Recipes](https://github.com/Righteous-AI/Strata-Recipes) study, which serves a MoE FP4-Local pack with a CPU expert cache: the same prompts decode at a comparable rate and prefill several times slower here, and both numbers are honest for their own stack.

The cold numbers below are the v1.2.4 pass: five greedy prompts on an idle card, one pass, from the same client harness the Strata benches used. They are not llama-benchy runs. Where the engine publishes its own rate for the same request, both are given. The earlier v1.2.3 pass and the version-to-version delta are in [comparisons.md](comparisons.md).

---

## What to run

TensorFold v1.2.4, commit `267c5af6`, native CUDA binary built with Zig 0.17.0 for `sm_120`. Weights are the EXL3 6.0 bpw pack of Qwen3.8-27B. The drafter is the BF16 `z-lab/Qwen3.8-27B-DFlash2` at revision `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`; the loader quantizes its projections to 4-bit as it reads them, so no 4-bit drafter download exists or is wanted.

```
--backend cuda
--parallel 1
--context 262144
--prompt-cache-gib 0
--profile stable
--kv-cache fp8
--drafter /mnt/models/qwen/Qwen3.8-27B-DFlash2
```

`--kv-cache` takes `bf16` or `fp8` on this build. NVFP4 is a weight format there, never a KV format. `--drafter-bits` has one accepted value, 4, which is the default, so it stays unset. `--prefill-fp8` is not accepted by the native engine, and the EXL3 prompt path never took FP8 activations, so there is no faster prompt lane for this pack.

Serve script: [`configs/serve-32g.sh`](configs/serve-32g.sh). Unit: [`services/tensorfold-engine.service`](services/tensorfold-engine.service). Model name: `qwen3.8-27b-exl3-6bpw`, port 12350.

---

## The cold bench, 2026-11-07 17:17 UTC (v1.2.4)

Five cold prompts, temperature 0, 256 tokens out, nothing reused, idle card. `cached_tokens` was 0 on every row and every row ended at `finish=length`. TTFT, prefill, rounds and acceptance are the server's own finished line; decode is the client clock.

| Prompt tokens | TTFT | Prefill | Decode | Rounds | Draft accepted | Tokens a round |
| --- | --- | --- | --- | --- | --- | --- |
| 8,244 | 4.823 s | 1,709.2 tok/s | 177.4 tok/s | 63 | 195/900 (21.7%) | 4.06 |
| 32,820 | 20.151 s | 1,628.7 tok/s | 153.8 tok/s | 68 | 188/931 (20.2%) | 3.76 |
| 131,124 | 96.157 s | 1,363.6 tok/s | 128.2 tok/s | 68 | 187/797 (23.5%) | 3.76 |
| 196,660 | 159.934 s | 1,229.6 tok/s | 119.9 tok/s | 66 | 190/699 (27.2%) | 3.88 |
| 249,908 | 219.516 s | 1,138.5 tok/s | 119.0 tok/s | 62 | 194/638 (30.4%) | 4.13 |

Against the v1.2.3 pass on the same card and the same prompts, prefill and time to first token improved 2.8 to 3.5 percent at every length, and total request time by the same amount: 7.7 s off the 249,908-token first token. The engine's 1.2.4 changelog describes the cause, a Qwen 3.5 and 3.6 CUDA prompt chunk that computes q, attention, o and the MLP of the kept row over its own 128-row tile while k and v still cover every row. The five finished lines carry the same reply hashes as the v1.2.3 pass, so the faster prompt path produced byte-identical output.

Decode and acceptance did not move: 177.4, 153.8 and 128.2 tok/s match to the decimal, and the first four rows have identical rounds and accepted counts. The 249,908 row took 62 rounds where v1.2.3 took 60, which reads as 4.13 tokens a round there. One drafter coin flip on one row, with one sample per length on both passes.

The 249,908-token row is 95% of the window and it ran without an allocation failure. Accepted-over-drafted looks low because the drafter proposes about 14 tokens a round and keeps about 3; tokens a round, 3.76 to 4.13, is the figure that lines up with the engine's own published range for this drafter.

Two properties of this pack are worth stating before anyone tunes it. Prefill on the EXL3 path is far from the MoE numbers the same card gives elsewhere on this host, and that is the pack's GEMM rather than a flag: upstream measures 880–970 tok/s from 2k to 16k for the 3.00 bpw branch on a DGX Spark, and this 6.0 bpw pack on the 5090 is 1.3 to 1.9 times that. And every reply in this pass was still reasoning when it hit `max_tokens`, so all 256 output tokens were `reasoning_content`. The comparison tables stay fair because the other engines' rows used the same prompts and the same 256-token cap.

---

## What sits on the card

From the server's own startup line:

```
CUDA sm_120 device 0 (NVIDIA GeForce RTX 5090): model 19.91 GiB; 1 stream at once,
9.09 GiB each at a 262144-token window (fp8 caches), of 10.0 GiB left after a
1.0 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache
```

```
RTX 5090  32,607 MiB
├── 19.91 GiB  EXL3 weights (6.0 bpw, 6-bit head, mul1 codebook)
├──  9.09 GiB  one stream at the 262,144-token window, fp8 caches
│   ├── about 4.3 GiB  the 16 full-attention layers' K/V at fp8 (the other 48 are linear attention)
│   └── the drafter's ring, taps and frame
├── about 0.94 GiB  drafter weights, affine-4 group 64, quantized at load
├── about 1.0 GiB   engine reserve
└── live gauge during the run: MEM 29.292Gi/31.843Gi (v1.2.3 pass)
```

Host memory peak for the boot was 23.1 GB on both passes. Two things live on the host rather than the card: the drafter's selector codebooks, 242.5 MiB, and, because the card had no room left, the model's bf16 embedding rows (248,320 × 5,120, so 2.54 GiB), which the engine maps into pinned host memory and reads in place. That is adaptive behaviour, logged as `embedding rows read from host memory: no room beside a 262144-token stream`, and it is what a full window costs on a 32 GB card.

Load took 39.0 s on v1.2.4 and 39.3 s on v1.2.3: about 19.6 to 19.8 s to read the pack and 18.5 s of kernel calibration. v1.2.4 also reads with direct reads by default, a change worth 28 to 31 percent on a GB10 for the models it names; on this EXL3 pack the load time did not move.

---

## Install

The paths assume an install root of `/opt/tensorfold` and a checkout at `/opt/tensorfold/tensorfold`. Change `User`, `Group`, `WorkingDirectory` and the model paths to your layout. The API key comes from `TENSORFOLD_KEY`; the script falls back to the placeholder `change-me`, so set your own LAN token before starting. The bind is `0.0.0.0:12350`, so keep it off the public internet.

```
sudo cp services/tensorfold-engine.service /etc/systemd/system/tensorfold-engine.service
sudo systemctl daemon-reload
sudo systemctl start tensorfold-engine.service
curl -sS http://127.0.0.1:12350/v1/models -H "Authorization: Bearer $TENSORFOLD_KEY"
```

Build, from a checkout, with Zig 0.17.0 and CUDA 13.5:

```
export CUDA_HOME=/usr/local/cuda-13.5
zig build -j10 -Dgpu=sm_120 -Dnvcc=/usr/local/cuda-13.5/bin/nvcc
```

The binary is `zig-out/native/bin/tensorfold-native`. `--version` prints `tensorfold-native 1.2.4` without loading a model. `-j10` has no space; a spaced `-j 10` makes Zig read the job count as empty. `sm_120` is the only architecture flag, and optimize `fast` with `strip` true are the defaults on that path.

Up to v1.2.3 this build needed a two-line local addition to `zig/build/product.zig`, because `build.zig` returned from the `-Dgpu` path before attaching the `profile` module that `engine_api` imports. v1.2.4 fixed it upstream: that path now calls `profile.walk`, so the checkout builds clean and a stale local edit to `product.zig` should be reverted before detaching onto a new tag.

Weights and drafter:

```
hf download turboderp/Qwen3.8-27B-exl3 --revision 6.00bpw --local-dir /mnt/models/qwen/Qwen3.8-27B-exl3
hf download z-lab/Qwen3.8-27B-DFlash2 --revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 \
  --local-dir /mnt/models/qwen/Qwen3.8-27B-DFlash2
```

Point the serve script's `MODEL` at whichever pack you downloaded. The EXL3 pack's `tokenizer.json` is byte-identical to the Flash Next FP4 pack's used by the Strata study, both 12,809,320 bytes, so a prompt file built for one encodes the same for the other and the two studies share their bench prompts.

The server is one process per card. Stop whatever else holds the GPU first; the unit lists the other engines in `Conflicts=`.

---

## Files

```
README.md                              this page
hardware.md                            the machine
comparisons.md                         the measured tables, and the same-card comparison
recipes/01-exl3-6bpw-dflash2.md        the recipe: flags, drafter, bench, what each flag is for
configs/serve-32g.sh                   the serve script the unit runs
services/tensorfold-engine.service     the unit
```

Weights stay on the machine that downloaded them. This repo does not ship a model.

---

## License

Measurements and write-up: [CC BY 4.0](LICENSE). Engine: TensorFold, as published by its author. Model weights: the Qwen license and the pack's own terms, as published with each checkpoint.
