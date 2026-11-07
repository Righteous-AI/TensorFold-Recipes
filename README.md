# TensorFold recipes — Qwen3.8-27B EXL3 on an RTX 5090

**Hardware:** ASUS ProArt Z890 · Intel Core Ultra 7 265K · 128 GB DDR5-5200 · RTX 5090 32 GB · Samsung 9100 PRO
**Engine:** [TensorFold](https://github.com/ashhart/TensorFold) v1.2.3, native CUDA server, built for sm_120 on CUDA 13.5
**What is running:** [recipe 1](recipes/01-exl3-6bpw-dflash2.md), Qwen3.8-27B in EXL3 6.0 bpw with the DFlash2 drafter, 262,144-token window
**Measured:** 2026-11-07

One server, one card, one recipe. A dense 27B quantized to 6.0 bits per weight, drafted with DFlash2, at the model's full 262,144-token window. This is a different machine from the [Strata-Recipes](https://github.com/Righteous-AI/Strata-Recipes) study, which serves a MoE FP4-Local pack with a CPU expert cache: the same prompts decode at a comparable rate and prefill several times slower here, and both numbers are honest for their own stack.

The cold numbers below are five greedy prompts on an idle card, one pass each, from the same client harness the Strata benches used. They are not llama-benchy runs. Where the engine publishes its own rate for the same request, both are given.

---

## What to run

TensorFold v1.2.3, commit `6ec5fa4c`, native CUDA binary built with Zig 0.17.0 for `sm_120`. Weights are the EXL3 6.0 bpw pack of Qwen3.8-27B. The drafter is the BF16 `z-lab/Qwen3.8-27B-DFlash2` at revision `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`; the loader quantizes its projections to 4-bit as it reads them, so no 4-bit drafter download exists or is wanted.

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

## The cold bench, 2026-11-07 12:10 UTC

Five cold prompts, temperature 0, 256 tokens out, nothing reused, idle card. `cached_tokens` was 0 on every row and every row ended at `finish=length`. TTFT, prefill, rounds and acceptance are the server's own finished line; decode is the client clock.

| Prompt tokens | TTFT | Prefill | Decode | Rounds | Draft accepted | Tokens a round |
| --- | --- | --- | --- | --- | --- | --- |
| 8,244 | 4.962 s | 1,661.4 tok/s | 177.4 tok/s | 63 | 195/900 (21.7%) | 4.06 |
| 32,820 | 20.721 s | 1,583.7 tok/s | 153.8 tok/s | 68 | 188/934 (20.1%) | 3.76 |
| 131,124 | 99.193 s | 1,321.9 tok/s | 128.2 tok/s | 68 | 187/798 (23.4%) | 3.76 |
| 196,660 | 165.363 s | 1,189.2 tok/s | 119.9 tok/s | 66 | 190/699 (27.2%) | 3.88 |
| 249,908 | 227.189 s | 1,100.0 tok/s | 122.7 tok/s | 60 | 196/622 (31.5%) | 4.27 |

The 249,908-token row is 95% of the window and it ran without an allocation failure. Accepted-over-drafted looks low because the drafter proposes about 14 tokens a round and keeps about 3; tokens a round, 3.76 to 4.27, is the figure that lines up with the engine's own published range for this drafter.

Two properties of this pack are worth stating before anyone tunes it. Prefill on the EXL3 path is far from the MoE numbers the same card gives elsewhere on this host, and that is the pack's GEMM rather than a flag: upstream measures 880–970 tok/s from 2k to 16k for the 3.00 bpw branch on a DGX Spark, and this 6.0 bpw pack on the 5090 is 1.2 to 1.9 times that. And every reply in this pass was still reasoning when it hit `max_tokens`, so all 256 output tokens were `reasoning_content`. The comparison tables stay fair because the other engines' rows used the same prompts and the same 256-token cap.

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
└── live gauge during the run: MEM 29.292Gi/31.843Gi
```

Host memory peak for the boot was 23.1 GB. Two things live on the host rather than the card: the drafter's selector codebooks, 242.5 MiB, and, because the card had no room left, the model's bf16 embedding rows (248,320 × 5,120, so 2.54 GiB), which the engine maps into pinned host memory and reads in place. That is adaptive behaviour, logged as `embedding rows read from host memory: no room beside a 262144-token stream`, and it is what a full window costs on a 32 GB card.

Load took 39.3 s: 19.8 s to read the pack, 18.5 s of kernel calibration.

---

## Install

The paths assume an install root of `/opt/tensorfold` and a checkout at `/opt/tensorfold/tensorfold`. Change `User`, `Group`, `WorkingDirectory` and the model paths to your layout. The API key is the LAN token `local`, the same one the other servers on this host use; the bind is `0.0.0.0:12350`, so keep it off the public internet.

```
sudo cp services/tensorfold-engine.service /etc/systemd/system/tensorfold-engine.service
sudo systemctl daemon-reload
sudo systemctl start tensorfold-engine.service
curl -sS http://127.0.0.1:12350/v1/models -H 'Authorization: Bearer local'
```

Build, from a checkout, with Zig 0.17.0 and CUDA 13.5:

```
export CUDA_HOME=/usr/local/cuda-13.5
zig build -j10 -Dgpu=sm_120 -Dnvcc=/usr/local/cuda-13.5/bin/nvcc
```

The binary is `zig-out/native/bin/tensorfold-native`. `--version` prints `tensorfold-native 1.2.3` without loading a model.

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
