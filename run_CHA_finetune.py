import logging
import os
import sys
import math
import json
import datetime
import transformers
import pandas as pd
from tqdm import tqdm
from transformers.utils import send_example_telemetry
from transformers.trainer_utils import get_last_checkpoint
from transformers import set_seed
import datasets
from datasets import DatasetDict, Dataset, load_dataset, load_from_disk
from dataprocessor import SamplePreprocessorForFinetune, CHADataCollator
from trainer import CHATrainer
# from src.llama.modeling_llama import LlamaForCausalLMWithBeacon
from args import parse_args
from utils import load_model_and_tokenizer

logger = logging.getLogger(__name__)
if __name__ == "__main__":

    model_args, data_args, training_args, custom_args = parse_args(is_device_specific=False)

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    if "llama" in model_args.model_name_or_path.lower():
        model_name = "llama"
    elif "qwen" in model_args.model_name_or_path.lower():
        model_name = "qwen"
    elif "mistral" in model_args.model_name_or_path.lower():
        model_name = "mistral"
    elif "deepseek" in model_args.model_name_or_path.lower():
        model_name = "deepseek"
    else:
        raise ValueError("Unsupported model name. Please use a model from Llama, Qwen, or Mistral.")
    local_rank = str(os.environ.get("LOCAL_RANK", 0))
    if training_args.overwrite_output_dir or not os.path.exists(training_args.output_dir):
        training_args.output_dir = training_args.output_dir + f"-{model_name}" + "-gradient" + str(training_args.gradient_accumulation_steps) + "-epochs" + str(training_args.num_train_epochs) + "-time" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "-localrank" + local_rank
    training_args.run_name = training_args.output_dir.split('/')[-1]

    # import wandb
    # wandb.login(key = '2dbf72a244cd2141d63b7191be9de6f90ef4056b')
    # wandb.init(project = 'CHA_finetune', name = training_args.run_name)
    # import swanlab
    # swanlab.init(project='CHA_finetune', name=training_args.run_name, config=vars(training_args))
    from swanlab.integration.transformers import SwanLabCallback
    swanlab_callback = SwanLabCallback(project="CHA_finetune", experiment_name=training_args.run_name, config=vars(training_args))
    training_args.report_to = []

    # Logging stuff
    send_example_telemetry("run_CHA_finetune", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    if training_args.deepspeed:
        logger.info(f"✅ DeepSpeed config detected: {training_args.deepspeed}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # tokenizer = AutoTokenizer.from_finetuneed(args.model_name_or_path)
    model, tokenizer = load_model_and_tokenizer(
        model_args = model_args, 
        model_name = model_name
    )
    preprocessor = SamplePreprocessorForFinetune(tokenizer=tokenizer, beacon_size=custom_args.beacon_size, max_length=data_args.max_length)
    data_collator = CHADataCollator()
    if data_args.dataset_path is None:
        raise ValueError("Please provide the dataset path with --dataset_path")
    train_dataset = load_dataset("json", data_files={"train": data_args.dataset_path})['train']
    samples = []
    for sample in tqdm(train_dataset):
        try:
            table_dict = json.loads(sample['table'])
            df = pd.DataFrame(table_dict["data"], columns=table_dict["columns"])
        except:
            continue
        sample['df'] = df
        processed_sample = preprocessor(sample)
        samples.append(processed_sample)
    train_dataset = Dataset.from_list(samples)
    # train_dataset.save_to_disk(data_args.save_path)
    train_dataset.shuffle(seed=training_args.seed)
    eval_dataset = None
    # filter out training sample that token length larger than 6000
    train_dataset = train_dataset.filter(lambda x: len(x['input_ids']) <= 6000, num_proc=32)  

    trainer = CHATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    trainer.add_callback(swanlab_callback)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        
        metrics = trainer.evaluate()

        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)