import os
import json
import pandas as pd
from copy import deepcopy
from transformers import HfArgumentParser, AutoModelForCausalLM, AutoTokenizer
# from utils import load_model_and_tokenizer
# from dataprocessor import SamplePreprocessor, CHADataCollator
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm

@dataclass
class InferArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer to use."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    attn_implementation: Optional[str] = field(
        default="eager",
        metadata={
            "help": ("The attention implementation to use in the model."),
            "choices": ["eager", "sdpa", "flash_attention_2"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=True,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
    )
        },
    )
    dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to folder with train.json and val.json"},
    )
    save_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save the processed dataset"},
    )
    beacon_size: int = field(
        default=1, metadata={"help": "Beacon size"}
    )

if __name__ == "__main__":
    parser = HfArgumentParser(InferArguments)
    (args,) = parser.parse_args_into_dataclasses()

    if "llama" in args.model_name_or_path.lower():
        model_name = "llama"
    elif "qwen" in args.model_name_or_path.lower():
        model_name = "qwen"
    elif "mistral" in args.model_name_or_path.lower():
        model_name = "mistral"
    elif "deepseek" in args.model_name_or_path.lower():
        model_name = "deepseek"
    else:
        raise ValueError("Unsupported model name. Please use a model from Llama, Qwen, or Mistral.")

    print("Model name:", model_name)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    model = model.cuda()
    model.eval()

    fnames = [x for x in os.listdir(args.dataset_path) if x.endswith('.jsonl')]
    for filename in fnames:
        print(f"Processing {filename}")
        file_path = os.path.join(args.dataset_path, filename)
        lines = [json.loads(x) for x in open(file_path, encoding='utf-8').readlines() if x.strip()]

        # 保存路径
        save_path = os.path.join(args.save_path, args.model_name_or_path.split('/')[-1] + '_' + filename.split('.')[0] + '.jsonl')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 加载已有结果（resume 支持）
        if os.path.exists(save_path):
            with open(save_path, 'r', encoding='utf-8') as f:
                saved = {json.loads(l)['id']: json.loads(l) for l in f if '"prediction"' in l}
        else:
            saved = {}

        with open(save_path, 'w') as fout:
            for idx, sample in tqdm(enumerate(lines), total=len(lines)):
                sample_id = sample['id']
                if sample_id in saved:
                    fout.write(json.dumps(saved[sample_id]) + '\n')
                    continue

                # 构造输入
                chat = [{"role": "user", "content": sample["instruction"]}]
                prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt, return_tensors='pt').to("cuda")

                # 推理
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=8000,
                    do_sample=False,
                )
                generated_texts = tokenizer.decode(output_ids[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)

                print(f"{idx}: generated: {generated_texts}, answer: {sample['answer']}")

                sample['prediction'] = generated_texts
                sample['model_name'] = model_name

                fout.write(json.dumps(sample) + '\n')
