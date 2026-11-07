# Comparisons

Every number here is a cold prompt: temperature 0, 256 tokens out, `cached_tokens` 0, one pass per length, one process on the card, client clock. The five prompts are the same files across engines, so the input is byte-identical.

## The same card, two engines — 2026-11-07

| Prompt tokens | Strata prefill | TensorFold prefill | Strata decode | TensorFold decode | Strata TTFT | TensorFold TTFT |
| --- | --- | --- | --- | --- | --- | --- |
| 8,244 | 5,712.7 | 1,661.4 | 156.6 | **177.4** | 1.443 s | 4.962 s |
| 32,820 | 7,244.0 | 1,583.7 | **174.1** | 153.8 | 4.531 s | 20.721 s |
| 131,124 | 7,724.6 | 1,321.9 | **174.1** | 128.2 | 16.975 s | 99.193 s |
| 196,660 | 7,591.6 | 1,189.2 | **169.8** | 119.9 | 25.905 s | 165.363 s |
| 249,908 | 7,221.3 | 1,100.0 | **178.6** | 122.7 | 34.607 s | 227.189 s |

Rates are tok/s. Strata is [recipe 6 of Strata-Recipes](https://github.com/Righteous-AI/Strata-Recipes/blob/main/recipes/06-cuda135-v0212.md): Strata v0.2.12, engine `0.1.58`, Qwen3.8-Flash-Next FP4-Local, an MoE with a pinned CPU expert cache, DFlash2 at spec 6. TensorFold is [recipe 1](recipes/01-exl3-6bpw-dflash2.md) at v1.2.3, the dense Qwen3.8-27B in EXL3 6.0 bpw, DFlash2, benched at 12:10 UTC. The v1.2.4 pass at 17:17 UTC is 2.8 to 3.5 percent faster on prefill and time to first token, so the current build closes that prefill gap by a few percent and nothing else; the table keeps the pair of passes that were run closest together.

These are different models and different weights, so the table is a description of two machines rather than a verdict on two engines. What the gap actually reflects:

- **Prefill, 3.4 to 6.6 times faster on Strata.** A MoE activates a few billion parameters a token and keeps its experts pinned in host RAM; a dense 27B reads and multiplies the whole model at every token. TensorFold's EXL3 prompt GEMM then runs dequantized weights, and that pack has no FP8 prompt lane. Against its own engine, this pack is normal: upstream measures the 3.00 bpw branch at 880–970 tok/s from 2k to 16k on a DGX Spark, and 1,100–1,661 tok/s on the 5090 is 1.2 to 1.9 times that.
- **Decode, within about 15% of each other, and ahead at the short prompt.** 177.4 against 156.6 tok/s at 8,244 tokens is a dense 6-bit model keeping up with a quantized MoE, which is what speculative decoding buys here: 3.76 to 4.27 accepted tokens a round. From 131,124 tokens up Strata pulls ahead by 36 to 46%, where the dense model's attention and full-width GEMMs cost more per token.
- **Time to first token follows prefill.** A 249,908-token prompt starts answering after 34.6 s on Strata and 227.2 s here. For long-document workloads that single number decides which server you use, whatever the decode rate says.

## The same recipe, two builds — 2026-11-07

Same card, same pack, same flags, same five saved prompts, one pass each. v1.2.3 ran at 12:10 UTC and v1.2.4 at 17:17 UTC. Left value is v1.2.3, right value is v1.2.4.

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | Rounds | Draft accepted | Tokens a round | Total s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8,244 | 4.962 → **4.823** (−2.8%) | 1,661.4 → **1,709.2** (+2.9%) | 177.4 → 177.4 | 63 → 63 | 195/900 → 195/900 | 4.06 → 4.06 | 6.401 → **6.261** (−2.2%) |
| 32,820 | 20.721 → **20.151** (−2.8%) | 1,583.7 → **1,628.7** (+2.8%) | 153.8 → 153.8 | 68 → 68 | 188/934 → 188/931 | 3.76 → 3.76 | 22.383 → **21.810** (−2.6%) |
| 131,124 | 99.193 → **96.157** (−3.1%) | 1,321.9 → **1,363.6** (+3.2%) | 128.2 → 128.2 | 68 → 68 | 187/798 → 187/797 | 3.76 → 3.76 | 101.186 → **98.147** (−3.0%) |
| 196,660 | 165.363 → **159.934** (−3.3%) | 1,189.2 → **1,229.6** (+3.4%) | 119.9 → 119.9 | 66 → 66 | 190/699 → 190/699 | 3.88 → 3.88 | 167.496 → **162.061** (−3.2%) |
| 249,908 | 227.189 → **219.516** (−3.4%) | 1,100.0 → **1,138.5** (+3.5%) | 122.7 → 119.0 (−3.0%) | 60 → 62 | 196/622 → 194/638 | 4.27 → 4.13 | 229.272 → **221.659** (−3.3%) |

- **The prompt path is what changed.** Prefill gained 2.8 to 3.5 percent at every length and the gain grows slowly with prompt size, so the whole 249,908-token request finishes 7.6 s sooner. The 1.2.4 changelog names the mechanism: a Qwen 3.5 or 3.6 CUDA prompt chunk now computes q, attention, o and the MLP of the kept row over that row's 128-row tile, while k and v still run over every row, with the cache layout and the last row's logits left alone.
- **Output is identical.** The five finished lines report the same reply hashes in the same order on both passes: `5d803fcde338`, `9dd08c4c5637`, `87b4a5d0979f`, `5d836571bf52`, `f572a202ac3a`. Same prompts, same greedy tokens, same bytes back.
- **Decoding is where it stayed.** The first four rows have the same rounds and the same accepted counts, and decode reads the same to the decimal. The 249,908 row went from 60 rounds and 196/622 to 62 rounds and 194/638, one drafter coin flip, which is what moves its 4.27 to 4.13 and its client decode from 122.7 to 119.0. Accepted stays in the 20 to 31 percent band on both passes.
- **Loading is where it stayed.** 39.3 s to 39.0 s, with the read phase at 19.788 s and 19.599 s. v1.2.4 turned direct reads on by default, published at 28 to 31 percent faster cold loads on a GB10 for the models it names, and this EXL3 pack on this host did not move. `TF_LOAD_DIRECT=0` returns the old read-ahead if you want to A/B it.
- **Memory plan unchanged.** 19.91 GiB of weights, one 9.09 GiB fp8 stream at the full window, host peak 23.1 GB on both boots.

One pass each. Five consistent prefill gains at five different lengths is a real signal; the single decode sample at 249,908 is not a regression, and the drafter's own numbers did not change on the other four rows.

## The same engine, published numbers

Upstream TensorFold measures the 3.00 bpw branch of this pack on one DGX Spark, medians of 15 runs:

| Cell | EXL3 3.00bpw (GB10) | This recipe, 6.0bpw (5090, v1.2.4) |
| --- | --- | --- |
| Cold prefill 2k–16k | 880–970 tok/s | 1,364–1,709 tok/s |
| Cold prefill 32k–64k | 720–890 tok/s | 1,230–1,629 tok/s |
| Drafted tokens a round, code | 4.0–5.2 | 3.76–4.13 |
| Drafted tokens a round, chat | 3.0–3.3 | — |

The 5090 sits 1.3 to 1.9 times above the published Spark figures on prefill while carrying a heavier 6.0 bpw pack, and its drafted round length lands inside the published range.

## What is not in this repo

No agent traffic. The Strata study has 852 finished agentic requests behind its headline decode numbers; this TensorFold pass is five deliberate prompts and nothing else. A run through a real coding pipeline on this pack is the missing measurement, and it is the one that would make the decode column mean something for day-to-day use.
