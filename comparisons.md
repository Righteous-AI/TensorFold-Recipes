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

Rates are tok/s. Strata is [recipe 6 of Strata-Recipes](https://github.com/Righteous-AI/Strata-Recipes/blob/main/recipes/06-cuda135-v0212.md): Strata v0.2.12, engine `0.1.58`, Qwen3.8-Flash-Next FP4-Local, an MoE with a pinned CPU expert cache, DFlash2 at spec 6. TensorFold is [recipe 1](recipes/01-exl3-6bpw-dflash2.md): v1.2.3, the dense Qwen3.8-27B in EXL3 6.0 bpw, DFlash2.

These are different models and different weights, so the table is a description of two machines rather than a verdict on two engines. What the gap actually reflects:

- **Prefill, 3.4 to 6.6 times faster on Strata.** A MoE activates a few billion parameters a token and keeps its experts pinned in host RAM; a dense 27B reads and multiplies the whole model at every token. TensorFold's EXL3 prompt GEMM then runs dequantized weights, and that pack has no FP8 prompt lane. Against its own engine, this pack is normal: upstream measures the 3.00 bpw branch at 880–970 tok/s from 2k to 16k on a DGX Spark, and 1,100–1,661 tok/s on the 5090 is 1.2 to 1.9 times that.
- **Decode, within about 15% of each other, and ahead at the short prompt.** 177.4 against 156.6 tok/s at 8,244 tokens is a dense 6-bit model keeping up with a quantized MoE, which is what speculative decoding buys here: 3.76 to 4.27 accepted tokens a round. From 131,124 tokens up Strata pulls ahead by 36 to 46%, where the dense model's attention and full-width GEMMs cost more per token.
- **Time to first token follows prefill.** A 249,908-token prompt starts answering after 34.6 s on Strata and 227.2 s here. For long-document workloads that single number decides which server you use, whatever the decode rate says.

## The same engine, published numbers

Upstream TensorFold measures the 3.00 bpw branch of this pack on one DGX Spark, medians of 15 runs:

| Cell | EXL3 3.00bpw (GB10) | This recipe, 6.0bpw (5090) |
| --- | --- | --- |
| Cold prefill 2k–16k | 880–970 tok/s | 1,321–1,661 tok/s |
| Cold prefill 32k–64k | 720–890 tok/s | 1,189–1,584 tok/s |
| Drafted tokens a round, code | 4.0–5.2 | 3.76–4.27 |
| Drafted tokens a round, chat | 3.0–3.3 | — |

The 5090 sits 1.2 to 1.9 times above the published Spark figures on prefill while carrying a heavier 6.0 bpw pack, and its drafted round length lands inside the published range.

## What is not in this repo

No agent traffic. The Strata study has 852 finished agentic requests behind its headline decode numbers; this TensorFold pass is five deliberate prompts and nothing else. A run through a real coding pipeline on this pack is the missing measurement, and it is the one that would make the decode column mean something for day-to-day use.
