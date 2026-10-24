# PECO-MICRO2026

This repository contains the code for PECO-MICRO2026: token-clustering sparse attention for HunyuanVideo with CUDA kernels.

Requires Linux, Python 3.10+, a working NVIDIA driver, and a CUDA Toolkit matching your PyTorch installation. The build prefers CUDA 12.8 when available.

```bash
python -m pip install -e .
bash run.sh                          # Base-Offset PE enabled by default
BASE_OFFSET_PE=off bash run.sh        # Disable fake quantization for higher speed
BASE_OFFSET_PE=on bash run.sh         # Enable fake quantization
```

Edit `prompt.txt` for the prompt and `run.sh` for inference settings. Defaults are 480x720, 17 frames, 50 denoising steps, `cuda:0`, and sequential CPU offload. Environment overrides include `PYTHON`, `DEVICE`, `OFFLOAD`, `MODEL_ID`, `MODEL_REVISION`, and `PROMPT_FILE`. For local weights, set `MODEL_ID=/path/to/model MODEL_REVISION=''`.

Videos are saved directly to `results/YYYYMMDD_HHMMSS_microseconds_ModelName.mp4`, using Asia/Shanghai time. No run logs or additional output subdirectories are created. `run.sh` builds the required CUDA libraries; use `bash build.sh` to build them separately.

Base-Offset PE uses fake quantization: video Q/K centroid residuals are quantized to 4 bits and then dequantized before floating-point attention. These additional operations slow down this software implementation. Disable the feature with `BASE_OFFSET_PE=off` to skip that overhead and improve speed. This implementation does not use native INT4 attention.

The Base-Offset encoder, decoder, and fake-quantization entry point are in `peco/quantization.py`. `peco/processor.py` applies them to Q/K before token reordering and sparse attention.

## Limitations

- The CUDA implementation does not use warp specialization.
- PECO is not applicable to few-step diffusion models.
- GPU numerical correctness, video quality, and performance remain unverified in the current environment because its NVIDIA driver is unavailable.
