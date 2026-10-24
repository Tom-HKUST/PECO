"""Single-prompt HunyuanVideo entry used by run.sh."""

import argparse
from datetime import datetime
import math
from pathlib import Path
import random
import re
from zoneinfo import ZoneInfo


def output_video_path(model_id, started_at=None):
    """Keep generated videos in one results directory, named by time and model."""
    if started_at is None:
        started_at = datetime.now(ZoneInfo("Asia/Shanghai"))
    model_name = model_id.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    model_name = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("._-")[:96] or "model"
    filename = f"{started_at:%Y%m%d_%H%M%S_%f}_{model_name}.mp4"
    return Path(__file__).resolve().parent / "results" / filename


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default="tencent/HunyuanVideo")
    parser.add_argument("--model_revision", default="refs/pr/18")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative_prompt", default="Aerial view, overexposed, low quality, deformation, bad composition, bad hands, bad teeth, bad eyes, bad limbs, distortion")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--offload", choices=("sequential", "model", "none"), default="sequential")
    parser.add_argument("--first_times_fp", type=float, default=0.04)
    parser.add_argument("--first_layers_fp", type=float, default=0.0)
    parser.add_argument("--num_q_centroids", type=int, default=100)
    parser.add_argument("--num_k_centroids", type=int, default=500)
    parser.add_argument("--top_p_kmeans", type=float, default=0.9)
    parser.add_argument("--min_kc_ratio", type=float, default=0.1)
    parser.add_argument("--kmeans_iter_init", type=int, default=50)
    parser.add_argument("--kmeans_iter_step", type=int, default=2)
    parser.add_argument("--zero_step_kmeans_init", action="store_true")
    parser.add_argument("--base_offset_bits", type=int, choices=(0, 4, 8), default=4)
    args = parser.parse_args(argv)
    if min(args.height, args.width, args.num_frames, args.num_inference_steps) < 1:
        parser.error("Video dimensions and inference steps must be positive")
    if args.height % 16 or args.width % 16 or args.num_frames % 4 != 1:
        parser.error("Height/width must be divisible by 16; num_frames must be 4n+1")
    if not 0 <= args.first_times_fp <= 1 or not 0 <= args.first_layers_fp <= 1:
        parser.error("Warmup fractions must be in [0, 1]")
    return args


def main(argv=None):
    args = parse_args(argv)
    output = output_video_path(args.model_id)
    import numpy as np
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler, HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
    from diffusers.utils import export_to_video

    from peco.processor import PecoConfig
    from peco.model import install_attention, reset_state

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("HunyuanVideo inference requires an available NVIDIA CUDA device")
    torch.cuda.set_device(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    revision = {"revision": args.model_revision} if args.model_revision else {}
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        args.model_id, subfolder="transformer", torch_dtype=torch.bfloat16, **revision,
    )
    pipe = HunyuanVideoPipeline.from_pretrained(
        args.model_id, transformer=transformer,
        scheduler=FlowMatchEulerDiscreteScheduler(shift=7.0),
        torch_dtype=torch.bfloat16, **revision,
    )
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    total_layers = len(transformer.transformer_blocks) + len(transformer.single_transformer_blocks)
    config = PecoConfig(
        num_q_centroids=args.num_q_centroids, num_k_centroids=args.num_k_centroids,
        top_p=args.top_p_kmeans, min_kc_ratio=args.min_kc_ratio,
        kmeans_iter_init=args.kmeans_iter_init, kmeans_iter_step=args.kmeans_iter_step,
        warmup_steps=math.floor(args.first_times_fp * args.num_inference_steps),
        warmup_layers=math.floor(args.first_layers_fp * total_layers),
        zero_step_kmeans_init=args.zero_step_kmeans_init,
        base_offset_bits=args.base_offset_bits,
    )
    install_attention(pipe, config)
    reset_state(pipe)
    # Sequential offload reduces the amount of model weights resident on the GPU.
    if args.offload == "sequential":
        pipe.enable_sequential_cpu_offload(device=device)
    elif args.offload == "model":
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)
    with torch.inference_mode():
        frames = pipe(
            prompt=args.prompt, negative_prompt=args.negative_prompt,
            height=args.height, width=args.width, num_frames=args.num_frames,
            guidance_scale=6.0, num_inference_steps=args.num_inference_steps,
            generator=torch.Generator(device=device).manual_seed(args.seed),
        ).frames[0]
    output.parent.mkdir(exist_ok=True)
    export_to_video(frames, str(output), fps=24)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
