import os
import json
import pandas as pd
from copy import deepcopy
from transformers import HfArgumentParser
from utils import load_model_and_tokenizer
from dataprocessor import SamplePreprocessor, CHADataCollator
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
    (args, ) = parser.parse_args_into_dataclasses()
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
    model, tokenizer = load_model_and_tokenizer(
        model_args = args, 
        model_name = model_name
    )
    model = model.cuda()
    model.eval()
    preprocessor = SamplePreprocessor(tokenizer=tokenizer, beacon_size=args.beacon_size)
    data_collator = CHADataCollator()
    fnames = [x for x in os.listdir(args.dataset_path) if x.endswith('.jsonl')]
    for filename in fnames:
        print(f"Processing {filename}")
        if "DP" in filename:
            instruction_type = "DP"
        elif "PoT" in filename:
            instruction_type = "PoT"
        elif "SCoT" in filename:
            instruction_type = "SCoT"
        elif "TCoT" in filename:
            instruction_type = "TCoT"
        else:
            raise ValueError("Not support")
        lines = open(os.path.join(args.dataset_path, filename), encoding='utf-8').readlines()
        lines = [json.loads(x) for x in lines if x.strip()]
        samples = []
        for l in lines:
            sample = deepcopy(l)
            table_dict = sample['table']
            df = pd.DataFrame(table_dict["data"], columns=table_dict["columns"])
            sample['df'] = df
            sample['instruction_type'] = instruction_type
            processed_sample = preprocessor(sample)
            samples.append(processed_sample)
        samples = [data_collator([sample]) for sample in samples]
        for idx, sample in tqdm(enumerate(samples)):
            input_ids = sample['input_ids'].to('cuda')
            attention_mask = sample['attention_mask'].to('cuda')
            position_ids = sample['position_ids'].to('cuda')
            question_ids = sample['question_ids'].to('cuda')
            is_beacon = sample['is_beacon'].to('cuda')
            # print("input_ids shape: ", input_ids.shape)
            # print("attention_mask shape: ", attention_mask.shape)
            # print("position_ids shape: ", position_ids.shape)
            # print("is_beacon shape: ", is_beacon.shape)
            input_ids, attention_mask, position_ids, past_key_values = model.construct_inputs_for_generation(
                input_ids = input_ids,
                attention_mask = attention_mask,
                position_ids = position_ids,
                question_ids = question_ids,
                is_beacon = is_beacon
            )
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
            }
            output_ids = model.generate(
                **inputs,
                max_new_tokens = 8000,
            )
            generated_texts = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
            print(f"{idx}: generated: {generated_texts}, answer: {lines[idx]['answer']}")
            lines[idx]['prediction'] = generated_texts
            lines[idx]['model_name'] = model_name

            save_path = os.path.join(args.save_path, args.model_name_or_path.split('/')[-1]+'_'+filename.split('.')[0]+'.jsonl')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, 'w') as f:
                for item in lines:
                    f.write(json.dumps(item)+'\n')
