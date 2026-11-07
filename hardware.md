# The machine

```
HOST RAM  128 GB DDR5-5200 (4×32). Swap 32 GB.
├── 2.54 GiB   pinned bf16 embedding rows, mapped because the card had no room
├── 242.5 MiB  the drafter's two selector codebooks
├── page cache for the 22.6 GB pack while it loads
└── OS, the server. Peak for the boot: 23.1 GB

RTX 5090  32,607 MiB, compute capability 12.0, power limit 600 W
├── 19.91 GiB  EXL3 6.0 bpw weights with a 6-bit head and mul1 codebook
├──  9.09 GiB  one stream at the 262,144-token window, fp8 caches
├── about 0.94 GiB  drafter projections, affine-4 group 64
└── 1.0 GiB     engine reserve
```

- Board: ASUS ProArt Z890. CPU: Intel Core Ultra 7 265K, 8 P-cores and 12 E-cores.
- GPU driver: open kernel module 620.70.06. CUDA toolkit 13.5, nvcc V13.5.37 at `/usr/local/cuda-13.5`. Kernel `7.0.0-42-generic`.
- Engine: TensorFold v1.2.4, commit `267c5af6`, built with Zig 0.17.0 (`zig build -j10 -Dgpu=sm_120`). The binary is 50,235,784 bytes, sha256 `100ac2c1a0eb231d1ba1dbdac16613b6cf83892571eeed1ace1c3147c4e8b868`, and `--version` prints `tensorfold-native 1.2.4`. `ldd` shows libc only; the CUDA runtime is loaded when a server starts. Every CUDA fatbin marker in the binary reports `release 13.5` (110 occurrences of the string, and none for `release 13.4`), with `sm_120a` appearing once. The previous build, v1.2.3 at 49,464,584 bytes and sha256 `6384e0371f47572b919f3a06c70e573f18fd6ffad35ecae18bf807210c61bc19`, is kept beside it for rollback.
- Storage: the pack lives on `/mnt/models`, a Samsung 9100 PRO 2 TB linked at 32.0 GT/s PCIe x4, the Gen5 volume on this box. The copy on the 4 TB volume, a 9100 PRO at 16.0 GT/s, stayed as a backup. A cold read of 22.6 GB is quicker on the Gen5 link; decode after the weights are resident is not affected by which one it comes from.
- One process owns the card. Every engine unit on this host declares the others in `Conflicts=`, and only one can hold 32 GB.

The pack on this host is `Qwen3.8-27B-exl3-SC_6.00bpw_H6_V6`: 51 files, 22,624,940,803 bytes, EXL3 1.4.2, 6.0 bits per weight, 6-bit head, `mul1` codebook, architecture `Qwen3_5ForConditionalGeneration`, 64 layers of which 16 are full attention and 48 linear attention, 4 KV heads at head dimension 128, vocabulary 248,320, `max_position_embeddings` 262,144.

The drafter is `z-lab/Qwen3.8-27B-DFlash2` at revision `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`: 3,849,114,690 bytes, `DFlash2DraftModel`, five layers, hidden 5,120, `block_size` 8, target layers `[5, 19, 33, 47, 61]`, one `model.safetensors` of 3,848,817,896 bytes holding 81 BF16 tensors and no metadata.
