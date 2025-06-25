output_dir=synthesised_data/qwen2.5-bakeneko-32b-instruct/

nohup python Evol_Instruct_jp/main.py \
    --output_dir ${output_dir} \
    --seed_tasks_path ../ichikara_data/merged_output.jsonl \
    --model_name "rinna/qwen2.5-bakeneko-32b-instruct" \
    >> logs/qwen2.5-bakeneko-32b-instruct.log & 