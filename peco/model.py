"""Attach processors and one context hook to a single transformer instance."""

import inspect

import torch

from .processor import PecoAttention, InferenceContext, PecoConfig


def install_attention(pipe, config: PecoConfig):
    transformer = pipe.transformer
    previous = getattr(transformer, "_peco_context_hook", None)
    if previous is not None:
        previous.remove()
    context = InferenceContext()
    blocks = list(transformer.transformer_blocks) + list(transformer.single_transformer_blocks)
    processors = [PecoAttention(index, config, context) for index in range(len(blocks))]
    for block, processor in zip(blocks, processors):
        block.attn.set_processor(processor)

    signature = inspect.signature(transformer.forward)

    def update_context(module, args, kwargs):
        arguments = signature.bind_partial(*args, **kwargs).arguments
        text_mask = arguments["encoder_attention_mask"]
        if text_mask.shape[0] != 1:
            raise ValueError("Use one prompt per inference call")
        mask = text_mask[0].detach().to(device="cpu", dtype=torch.bool)
        valid = int(mask.sum())
        if not bool(mask[:valid].all()) or bool(mask[valid:].any()):
            raise ValueError("Hunyuan text attention mask must use right padding")
        context.step += 1
        context.text_length = mask.numel()
        context.valid_text_length = valid

    transformer._peco_context_hook = transformer.register_forward_pre_hook(update_context, with_kwargs=True)
    transformer._peco_context = context
    transformer._peco_processors = processors
    return context


def reset_state(pipe):
    """Call before each new video, including repeated calls on one pipeline."""
    pipe.transformer._peco_context.step = -1
    for processor in pipe.transformer._peco_processors:
        processor.reset()
