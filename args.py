import torch
from typing import Optional
from transformers import HfArgumentParser, TrainingArguments
from transformers.utils import cached_property, is_torch_npu_available
from dataclasses import dataclass, field

@dataclass
class ModelArguments:
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
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
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
        default="sdpa",
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


    def __post_init__(self):
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )

# add more arguments
@dataclass
class TrainingArgumentsAddDevice(TrainingArguments):
    """
    Arguments pertaining to the training of the model
    """

    device_main: str = field(
        default="cuda:0",
        metadata={"help": "Device to run the model on."},
    )
    device_aug: str = field(
        default=None,
        metadata={"help": "Device to run the model on."},
    )

    def __post_init__(self):
        super().__post_init__()
        print("Device: ", self.device)

    @cached_property
    def _setup_devices(self):
        device = super()._setup_devices

        if is_torch_npu_available():
            return torch.device(self.device_main)

        if self.use_cpu:
            return torch.device("cpu")

        torch.cuda.set_device(self.device_main)
        
        self._n_gpu = 1
        
        return torch.device(self.device_main)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to folder with train.json and val.json"},
    )
    eval_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to folder with eval.json"},
    )
    save_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save the processed dataset"},
    )
    overwrite_cache: bool = field(
        default=True,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_table_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum ONE input sequence length after tokenization. sequence longer "
                "than this will be truncated."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"},
    )
    line_by_line: bool = field(
        default=False,
        metadata={
            "help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    inference_on_training_precomputed_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to precomputed dataset for training"},
    )


    def __post_init__(self):
        if self.dataset_path is None:
            raise ValueError(
                "Need dataset_path"
            )

# add more arguments
@dataclass
class CustomArguments:
    """
    Custom arguments for the script
    """

    stop_after_n_steps: int = field(
        default=10000, metadata={"help": "Stop training after n steps"}
    )
    # param_dir: str = field(
    #     default=None, metadata={"help": "The directory to load the parameters from"}
    # )
    beacon_size: int = field(
        default=1, metadata={"help": "Beacon size"}
    )
    # sentence_encoder_lora_dropout: float = field(
    #     default=0.05, metadata={"help": "The dropout rate for lora"}
    # )
    # sentence_encoder_lora_r: int = field(default=16, metadata={"help": "The r value for lora"})
    # lm_model_lora_dropout: float = field(
    #     default=0.05, metadata={"help": "The dropout rate for lora"}
    # )
    # lm_model_lora_r: int = field(default=16, metadata={"help": "The r value for lora"})


def parse_args(is_device_specific = True):
    if not is_device_specific:
        parser = HfArgumentParser(
            (ModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments)
        )
    else:
        parser = HfArgumentParser(
            (ModelArguments, DataTrainingArguments, TrainingArgumentsAddDevice, CustomArguments)
        )

    model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()

    return model_args, data_args, training_args, custom_args