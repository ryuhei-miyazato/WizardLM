output_dir=synthesised_data/Gemma-2-Llama-Swallow-27b-it-v0.1-repeat/

nohup python Evol_Instruct_jp/main_repeat.py \
    --output_dir ${output_dir} \
    --seed_tasks_path ../ichikara_data/merged_output_1000.jsonl \
    --model_name "tokyotech-llm/Gemma-2-Llama-Swallow-27b-it-v0.1" \
    --cuda_device "4,5,6" \
    >> logs/Gemma-2-Llama-Swallow-27b-it-v0.1-repeat.log & 