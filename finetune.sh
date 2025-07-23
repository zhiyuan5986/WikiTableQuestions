# nohup deepspeed run_CHA_finetune.py \
#     --model_name_or_path /home/qiaoan/Documents/WikiTableQuestions/output/Mistral-7B-Instruct-v0.2/CHA-pretrain-mistral-gradient32-time20250718100143-localrank0 \
#     --tokenizer_name_or_path /mnt/hdd-storage/storage_all_users/qiaoan/data/Mistral-7B-Instruct-v0.2 \
#     --dataset_path /mnt/hdd-storage/storage_all_users/qiaoan/data/TableInstruct/TableInstruct_instructions.jsonl \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 32 \
#     --learning_rate 1e-5 \
#     --remove_unused_columns false \
#     --do_train true \
#     --do_eval false \
#     --seed 42 \
#     --bf16 true \
#     --fp16 false \
#     --output_dir output/Mistral-7B-Instruct-v0.2/CHA-finetune \
#     --save_steps 10 \
#     --gradient_checkpointing true \
#     --torch_dtype bfloat16 \
#     --attn_implementation eager \
#     --logging_strategy steps \
#     --logging_steps 1 \
#     --num_train_epochs 1 \
#     --use_cpu false \
#     --low_cpu_mem_usage false \
#     --max_length 3000 \
#     --deepspeed configs/ds_config_zero2.json \
#     > ./log/CHA_finetune_mistral.log 2>&1 &

# while true; do
#     if ! nvidia-smi | grep python; then
#     nohup deepspeed run_CHA_finetune.py \
#         --model_name_or_path /home/qiaoan/Documents/WikiTableQuestions/output/Qwen2-7B-Instruct/CHA-pretrain-qwen-gradient32-time20250721105147-localrank0 \
#         --tokenizer_name_or_path /mnt/hdd-storage/storage_all_users/qiaoan/data/Qwen2-7B-Instruct \
#         --dataset_path /mnt/hdd-storage/storage_all_users/qiaoan/data/TableInstruct/TableInstruct_instructions.jsonl \
#         --per_device_train_batch_size 1 \
#         --per_device_eval_batch_size 1 \
#         --gradient_accumulation_steps 32 \
#         --learning_rate 1e-5 \
#         --remove_unused_columns false \
#         --do_train true \
#         --do_eval false \
#         --seed 42 \
#         --bf16 true \
#         --fp16 false \
#         --output_dir output/Qwen2-7B-Instruct/CHA-finetune \
#         --save_steps 10 \
#         --gradient_checkpointing true \
#         --torch_dtype bfloat16 \
#         --attn_implementation eager \
#         --logging_strategy steps \
#         --logging_steps 1 \
#         --num_train_epochs 1 \
#         --use_cpu false \
#         --low_cpu_mem_usage false \
#         --max_length 3000 \
#         --deepspeed configs/ds_config_zero2.json \
#         > ./log/CHA_finetune_qwen.log 2>&1 &
#         break
#     fi
#     echo "Waiting for GPU to be available..."
#     sleep 300  # Wait for 5 minutes
# done
nohup deepspeed run_CHA_finetune.py \
    --model_name_or_path /home/qiaoan/Documents/WikiTableQuestions/output/Qwen2-7B-Instruct/CHA-pretrain-qwen-gradient32-time20250721105147-localrank0 \
    --tokenizer_name_or_path /mnt/hdd-storage/storage_all_users/qiaoan/data/Qwen2-7B-Instruct \
    --dataset_path /mnt/hdd-storage/storage_all_users/qiaoan/data/TableInstruct/TableInstruct_instructions.jsonl \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-5 \
    --remove_unused_columns false \
    --do_train true \
    --do_eval false \
    --seed 42 \
    --bf16 true \
    --fp16 false \
    --output_dir output/Qwen2-7B-Instruct/CHA-finetune \
    --save_steps 10 \
    --gradient_checkpointing true \
    --torch_dtype bfloat16 \
    --attn_implementation eager \
    --logging_strategy steps \
    --logging_steps 1 \
    --num_train_epochs 1 \
    --use_cpu false \
    --low_cpu_mem_usage false \
    --max_length 3000 \
    --deepspeed configs/ds_config_zero2.json \
    > ./log/CHA_finetune_qwen.log 2>&1 &