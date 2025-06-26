output_dir=synthesised_data/calm3-22b-chat/

nohup python Evol_Instruct_jp/main_repeat.py \
    --output_dir ${output_dir} \
    --seed_tasks_path ../ichikara_data/merged_output_1000.jsonl \
    --model_name "cyberagent/calm3-22b-chat" \
    --cuda_device "0,1,2" \
    --batch_size 5 \
    >> logs/calm3-22b-chat.log & 
