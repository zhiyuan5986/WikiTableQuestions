#!/bin/bash
# PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# nohup python run_SHA_pretrain.py \
#     --model_name_or_path /home/qiaoan/data/meta-llama-2-7b-chat-hf \
#     --dataset_path /home/qiaoan/data/long-llm-data/redpajama \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --learning_rate 5e-5 \
#     --remove_unused_columns false \
#     --do_train true \
#     --do_eval false \
#     --seed 42 \
#     --bf16 true \
#     --warmup_ratio 0.1 \
#     --max_grad_norm 2.0 \
#     --max_seq_length 128 \
#     --output_dir output/meta-llama-2-7b-chat-hf/MTP \
#     --save_steps 10 \
#     --gradient_checkpointing true \
#     --torch_dtype bfloat16 \
#     --attn_implementation eager \
#     --logging_strategy steps \
#     --logging_steps 1 \
#     --num_train_epochs 1 \
#     --use_cpu false \
#     --low_cpu_mem_usage false \
#     --max_length 5000 \
#     > ./log/SHA_pretrain_llama_lr5e-5.log 2>&1 &

nohup python run_CHA_pretrain.py \
    --model_name_or_path /home/qiaoan/data/meta-llama-2-7b-chat-hf \
    --tokenizer_name_or_path /home/qiaoan/data/meta-llama-2-7b-chat-hf \
    --dataset_path ./data/training.tsv \
    --save_path ./data/CHA_pretrain_llama \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 5e-5 \
    --remove_unused_columns false \
    --do_train true \
    --do_eval false \
    --seed 42 \
    --bf16 true \
    --fp16 false \
    --output_dir output/meta-llama-2-7b-chat-hf/CHA-pretrain \
    --save_steps 100 \
    --gradient_checkpointing true \
    --torch_dtype bfloat16 \
    --attn_implementation eager \
    --logging_strategy steps \
    --logging_steps 1 \
    --num_train_epochs 5 \
    --use_cpu false \
    --low_cpu_mem_usage false \
    --max_table_length 3000 \
    > ./log/CHA_pretrain.log 2>&1 &