import json
import random
import os

from call_llm import call_llm
from depth import createConstraintsPrompt, createDeepenPrompt, createConcretizingPrompt, createReasoningPrompt
from breadth import createBreadthPrompt

import argparse, torch, re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

EVOLUTIONS_PER_INSTRUCTION = 5

def initialize_models(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def encode_prompt(prompt, tokenizer, classification=False):

    messages = [
        # {"role": "system", "content": "あなたはタスク設計の専門家です。与えられた一連のタスクを参考に、形式を揃えて次に来るべきタスクを提案してください。"},
        {"role": "user", "content": prompt}
        # {"role": "system", "content": prompt}
    ]
    formatted_promt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True, 
        tokenize=False
    )
    return formatted_promt

def process_and_write_batch(batch_lines, model, tokenizer, output_file_path):
    if not batch_lines:
        return

    instructions = [json.loads(l)['instruction'].strip() for l in batch_lines]
    ids = [json.loads(l)['id'] for l in batch_lines]
    seed_ids = [id for id in ids for _ in range(EVOLUTIONS_PER_INSTRUCTION)]
    
    print("instructions: ", instructions)

    evol_prompts = []
    for instruction in instructions:
        evol_prompts.extend([
            encode_prompt(createConstraintsPrompt(instruction), tokenizer),
            encode_prompt(createDeepenPrompt(instruction), tokenizer),
            encode_prompt(createConcretizingPrompt(instruction), tokenizer),
            encode_prompt(createReasoningPrompt(instruction), tokenizer),
            encode_prompt(createBreadthPrompt(instruction), tokenizer),
        ])
    
    # LLMを呼び出してinstructionを生成
    evol_instructions = call_llm(evol_prompts, model, tokenizer)
    # print("Generated Instructions:", evol_instructions) # デバッグ用に生成内容を表示

    # バッチ処理ごとにファイルを開き、追記モードで書き込む
    with open(output_file_path, "a", encoding="utf-8") as f:
        for i, (evol_instruction, seed_id) in enumerate(zip(evol_instructions, seed_ids)):
            entry = {
                "seed_id": seed_id,
                "syn_id": f"{seed_id}_{i}", # このID生成ロジックは要件に応じて調整
                "instruction": evol_instruction,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The directory where the genearted data is stored.",
    )
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        required=True,
        default="data/seed_tasks.jsonl",
        help="The path to the human written data.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="The organization to use. If not specified, the default organization id will be used."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--cuda_device",
        type=str,
        default="1,2",
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # CUDAデバイスの設定 (必要に応じて)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    
    model, tokenizer = initialize_models(args.model_name)
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "machine_generated_instructions.jsonl")

    batch_lines = []
    with open(args.seed_tasks_path, 'r', encoding="utf-8") as fr:
        total_lines = sum(1 for _ in fr)
        fr.seek(0)

        for line in tqdm(fr, total=total_lines, desc="Generating prompts"):
            batch_lines.append(line)
            if len(batch_lines) == args.batch_size:
                process_and_write_batch(batch_lines, model, tokenizer, output_file)
                batch_lines = []
    
    # ループ終了後に残った最後のバッチを処理
    if batch_lines:
        process_and_write_batch(batch_lines, model, tokenizer, output_file)