output_dir=synthesised_data/qwen2.5-bakeneko-32b-instruct-v2-repeat/

nohup python Evol_Instruct_jp/main_repeat.py \
    --output_dir ${output_dir} \
    --seed_tasks_path ../ichikara_data/merged_output_1000.jsonl \
    --model_name "rinna/qwen2.5-bakeneko-32b-instruct-v2" \
    --cuda_device "4,5,6" \
    --batch_size 5 \
    >> logs/qwen2.5-bakeneko-32b-instruct-v2.log & 