import os
import torch
from transformers import (
    TrainerCallback,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
)
from sentence_transformers import SentenceTransformer
from peft import LoraConfig, PeftModel
from src.llama.modeling_llama import LlamaForCausalLMWithBeacon
from src.qwen2.modeling_qwen2 import Qwen2ForCausalLMWithBeacon
from src.mistral.modeling_mistral import MistralForCausalLMWithBeacon
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.mistral.modeling_mistral import MistralForCausalLM


class StopTrainingCallback(TrainerCallback):
    def __init__(self, stop_after_n_steps: int):
        self.stop_after_n_steps = stop_after_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_after_n_steps:
            control.should_training_stop = True
MODEL_MAP = {
    "llama": LlamaForCausalLMWithBeacon,
    "qwen": Qwen2ForCausalLMWithBeacon,
    "mistral": MistralForCausalLMWithBeacon,
}
def load_model_and_tokenizer(
        model_args, 
        model_name,
        lm_model_device_map = None,
        **kwargs
    ):
    # TODO: add other model
    if lm_model_device_map is None:
        model = MODEL_MAP[model_name].from_pretrained(model_args.model_name_or_path, torch_dtype=model_args.torch_dtype, low_cpu_mem_usage=model_args.low_cpu_mem_usage, attn_implementation=model_args.attn_implementation)
    else:
        model = MODEL_MAP[model_name].from_pretrained(model_args.model_name_or_path, torch_dtype=model_args.torch_dtype, low_cpu_mem_usage=model_args.low_cpu_mem_usage, attn_implementation=model_args.attn_implementation, device_map=lm_model_device_map)
    for name, param in model.named_parameters():
        if not "beacon" in name:
            param.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer