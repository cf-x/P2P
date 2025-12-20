import os

from peft import load_peft_weights
from safetensors.torch import save_file

from hyper_llm_modulator.utils import generate_simplex_points, get_end_points


def mix_loras(lora_dirs, mixing_setup, num_simplex_points, save_dir):
    # linearly mix the loras
    loras = {k: load_peft_weights(v, device="cpu") for k, v in lora_dirs.items()}
    if mixing_setup == "lerp":
        lerp_vals = generate_simplex_points(num_simplex_points, len(lora_dirs))
    elif mixing_setup == "end_points":
        lerp_vals = get_end_points(len(lora_dirs))

    lora_keys = list(loras.values())[0].keys()
    assert all(lora_keys == lora.keys() for lora in loras.values()), f"All loras must have the same keys"

    def_dir = f"{list(lora_dirs.values())[0]}"
    out_dirs = [f"{save_dir}/lora_{i}" for i in range(num_simplex_points)]
    for i, lerp_val in enumerate(lerp_vals):
        mixed_lora = {k: sum(lerp_val[j] * lora[k] for j, lora in enumerate(loras.values())) for k in lora_keys}
        os.makedirs(out_dirs[i], exist_ok=True)

        os.system(f"cp {def_dir}/adapter_config.json {out_dirs[i]}/adapter_config.json")

        save_file(mixed_lora, f"{out_dirs[i]}/adapter_model.safetensors")

    return out_dirs, lerp_vals
