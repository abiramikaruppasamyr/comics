import gc
import torch
from safetensors.torch import load_file
from peft import LoraConfig
from peft.tuners.lora import LoraLayer


def _compose_conv_lora_delta(down_weight: torch.Tensor, up_weight: torch.Tensor) -> torch.Tensor | None:
    if down_weight.dim() != 4 or up_weight.dim() != 4:
        return None

    if up_weight.shape[2:] != (1, 1):
        return None

    up_matrix = up_weight.squeeze(-1).squeeze(-1)

    if down_weight.shape[2:] == (1, 1):
        down_matrix = down_weight.squeeze(-1).squeeze(-1)
        delta = torch.einsum("or,ri->oi", up_matrix, down_matrix)
        return delta.unsqueeze(-1).unsqueeze(-1)

    return torch.einsum("or,rihw->oihw", up_matrix, down_weight)


def _remap_a1111_key(key: str) -> str | None:
    parts = key.split(".")
    suffix = ".".join(parts[1:])
    body = parts[0]

    if body.startswith("lora_te_"):
        body = body[len("lora_te_"):]
        body = body.replace("text_model_encoder_layers_", "text_model.encoder.layers.")
        body = body.replace("_self_attn_q_proj", ".self_attn.q_proj")
        body = body.replace("_self_attn_k_proj", ".self_attn.k_proj")
        body = body.replace("_self_attn_v_proj", ".self_attn.v_proj")
        body = body.replace("_self_attn_out_proj", ".self_attn.out_proj")
        body = body.replace("_mlp_fc1", ".mlp.fc1")
        body = body.replace("_mlp_fc2", ".mlp.fc2")
        body = body.strip(".")
        body = body.replace("..", ".")
        if suffix:
            return f"text_encoder.{body}.{suffix}"
        return f"text_encoder.{body}"

    if not body.startswith("lora_unet_"):
        return None

    body = body[len("lora_unet_"):]

    block_map = {
        "input_blocks_0_0": "down_blocks.0.resnets.0",
        "input_blocks_1_0": "down_blocks.0.resnets.0",
        "input_blocks_1_1": "down_blocks.0.attentions.0",
        "input_blocks_2_0": "down_blocks.0.resnets.1",
        "input_blocks_2_1": "down_blocks.0.attentions.1",
        "input_blocks_3_0": "down_blocks.0.downsamplers.0",
        "input_blocks_4_0": "down_blocks.1.resnets.0",
        "input_blocks_4_1": "down_blocks.1.attentions.0",
        "input_blocks_5_0": "down_blocks.1.resnets.1",
        "input_blocks_5_1": "down_blocks.1.attentions.1",
        "input_blocks_6_0": "down_blocks.1.downsamplers.0",
        "input_blocks_7_0": "down_blocks.2.resnets.0",
        "input_blocks_7_1": "down_blocks.2.attentions.0",
        "input_blocks_8_0": "down_blocks.2.resnets.1",
        "input_blocks_8_1": "down_blocks.2.attentions.1",
        "input_blocks_9_0": "down_blocks.2.downsamplers.0",
        "input_blocks_10_0": "down_blocks.3.resnets.0",
        "input_blocks_11_0": "down_blocks.3.resnets.1",
        "middle_block_0": "mid_block.resnets.0",
        "middle_block_1": "mid_block.attentions.0",
        "middle_block_2": "mid_block.resnets.1",
        "output_blocks_0_0": "up_blocks.0.resnets.0",
        "output_blocks_1_0": "up_blocks.0.resnets.1",
        "output_blocks_2_0": "up_blocks.0.resnets.2",
        "output_blocks_2_1": "up_blocks.0.upsamplers.0",
        "output_blocks_3_0": "up_blocks.1.resnets.0",
        "output_blocks_3_1": "up_blocks.1.attentions.0",
        "output_blocks_4_0": "up_blocks.1.resnets.1",
        "output_blocks_4_1": "up_blocks.1.attentions.1",
        "output_blocks_5_0": "up_blocks.1.resnets.2",
        "output_blocks_5_1": "up_blocks.1.attentions.2",
        "output_blocks_5_2": "up_blocks.1.upsamplers.0",
        "output_blocks_6_0": "up_blocks.2.resnets.0",
        "output_blocks_6_1": "up_blocks.2.attentions.0",
        "output_blocks_7_0": "up_blocks.2.resnets.1",
        "output_blocks_7_1": "up_blocks.2.attentions.1",
        "output_blocks_8_0": "up_blocks.2.resnets.2",
        "output_blocks_8_1": "up_blocks.2.attentions.2",
        "output_blocks_8_2": "up_blocks.2.upsamplers.0",
        "output_blocks_9_0": "up_blocks.3.resnets.0",
        "output_blocks_9_1": "up_blocks.3.attentions.0",
        "output_blocks_10_0": "up_blocks.3.resnets.1",
        "output_blocks_10_1": "up_blocks.3.attentions.1",
        "output_blocks_11_0": "up_blocks.3.resnets.2",
        "output_blocks_11_1": "up_blocks.3.attentions.2",
    }

    matched_prefix = None
    matched_diffusers = None
    for a1111_prefix, diffusers_prefix in block_map.items():
        if body.startswith(a1111_prefix):
            matched_prefix = a1111_prefix
            matched_diffusers = diffusers_prefix
            break

    if matched_prefix is None:
        return None

    remainder = body[len(matched_prefix):]
    remainder = remainder.replace("_transformer_blocks_", ".transformer_blocks.")
    remainder = remainder.replace("_attn1_to_q", ".attn1.to_q")
    remainder = remainder.replace("_attn1_to_k", ".attn1.to_k")
    remainder = remainder.replace("_attn1_to_v", ".attn1.to_v")
    remainder = remainder.replace("_attn1_to_out_0", ".attn1.to_out.0")
    remainder = remainder.replace("_attn2_to_q", ".attn2.to_q")
    remainder = remainder.replace("_attn2_to_k", ".attn2.to_k")
    remainder = remainder.replace("_attn2_to_v", ".attn2.to_v")
    remainder = remainder.replace("_attn2_to_out_0", ".attn2.to_out.0")
    remainder = remainder.replace("_ff_net_0_proj", ".ff.net.0.proj")
    remainder = remainder.replace("_ff_net_2", ".ff.net.2")
    remainder = remainder.replace("_proj_in", ".proj_in")
    remainder = remainder.replace("_proj_out", ".proj_out")
    remainder = remainder.replace("_norm1", ".norm1")
    remainder = remainder.replace("_norm2", ".norm2")
    remainder = remainder.replace("_norm3", ".norm3")
    remainder = remainder.replace("_emb_layers_1", ".time_emb_proj")
    remainder = remainder.replace("_in_layers_0", ".norm1")
    remainder = remainder.replace("_in_layers_2", ".conv1")
    remainder = remainder.replace("_out_layers_0", ".norm2")
    remainder = remainder.replace("_out_layers_3", ".conv2")
    remainder = remainder.replace("_skip_connection", ".conv_shortcut")
    if remainder.startswith("_"):
        remainder = "." + remainder[1:]

    remainder = remainder.strip(".")
    remainder = remainder.replace("..", ".")
    if remainder:
        mapped_key = f"unet.{matched_diffusers}.{remainder}"
    else:
        mapped_key = f"unet.{matched_diffusers}"

    if suffix:
        return f"{mapped_key}.{suffix}"
    return mapped_key


def load_a1111_lora_into_pipeline(pipeline, lora_path: str, lora_strength: float = 1.0):
    if lora_strength == 0.0:
        return pipeline  # LoRA disabled, return pipeline unchanged

    """
    Manually load A1111/kohya format LoRA into a diffusers pipeline
    by directly applying weights to the UNet and text encoder.
    Uses weight merging approach instead of PEFT adapter injection.
    """
    sd = load_file(lora_path)

    # Filter out diff_b keys and alpha keys, keep only lora_down and lora_up
    lora_down = {k: v for k, v in sd.items() if k.endswith(".lora_down.weight")}
    lora_up = {k: v for k, v in sd.items() if k.endswith(".lora_up.weight")}

    # Get alpha values (used to scale the LoRA contribution)
    alphas = {}
    for k, v in sd.items():
        if k.endswith(".alpha"):
            base = k[:-len(".alpha")]
            alphas[base] = v.item()

    unet = pipeline.unet
    text_encoder = pipeline.text_encoder

    applied = 0
    skipped = 0

    for down_key, down_weight in lora_down.items():
        base_key = down_key[:-len(".lora_down.weight")]
        up_key = base_key + ".lora_up.weight"

        if up_key not in lora_up:
            skipped += 1
            continue

        up_weight = lora_up[up_key]
        alpha = alphas.get(base_key, down_weight.shape[0])
        scale = lora_strength * (alpha / down_weight.shape[0])

        # Compute delta weight: up @ down (for linear layers)
        # For conv layers shape handling is different
        if down_weight.dim() == 2 and up_weight.dim() == 2:
            delta = up_weight @ down_weight
        elif down_weight.dim() == 4 and up_weight.dim() == 4:
            delta = _compose_conv_lora_delta(down_weight, up_weight)
            if delta is None:
                skipped += 1
                continue
        else:
            skipped += 1
            continue

        # Remap key to diffusers format
        diffusers_key = _remap_a1111_key(down_key.replace(".lora_down.weight", ""))
        if diffusers_key is None:
            skipped += 1
            continue

        # Find the target module
        key_parts = diffusers_key.split(".")
        model_name = key_parts[0]  # unet or text_encoder
        module_path = key_parts[1:]

        model = unet if model_name == "unet" else text_encoder

        try:
            module = model
            for part in module_path:
                module = getattr(module, part)
            if hasattr(module, "weight") and module.weight is not None:
                if module.weight.shape == delta.shape:
                    module.weight.data += scale * delta.to(module.weight.dtype)
                    applied += 1
                else:
                    skipped += 1
        except (AttributeError, StopIteration):
            skipped += 1
            continue

    print(f"LoRA applied: {applied} layers, skipped: {skipped} layers")
    return pipeline


def unload_lora_from_pipeline(pipeline, original_unet_state, original_te_state):
    """Restore original weights to unload LoRA effects."""
    pipeline.unet.load_state_dict(original_unet_state)
    pipeline.text_encoder.load_state_dict(original_te_state)
    gc.collect()
