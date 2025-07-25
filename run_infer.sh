DATA_PATH='/mnt/hdd-storage/storage_all_users/qiaoan/data/TableBench'
SAVE_PATH='/home/qiaoan/Documents/WikiTableQuestions/eval_examples/inference_results'
MODEL_DIR='/home/qiaoan/Documents/WikiTableQuestions/output/Qwen2-7B-Instruct/CHA-finetune-qwen-gradient32-epochs1.0-time20250723075312-localrank0'
TOKENIZER_DIR='/mnt/hdd-storage/storage_all_users/qiaoan/data/Qwen2-7B-Instruct'

CUDA_LAUNCH_BLOCKING=1 python infer.py \
    --model_name_or_path $MODEL_DIR \
    --tokenizer_name_or_path $TOKENIZER_DIR \
    --dataset_path $DATA_PATH \
    --save_path $SAVE_PATH \
    --torch_dtype bfloat16 \
    --attn_implementation eager \
    --low_cpu_mem_usage false


