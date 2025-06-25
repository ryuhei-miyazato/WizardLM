import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import json
import random
from pathlib import Path

from call_llm import call_llm
from depth import createConstraintsPrompt, createDeepenPrompt, createConcretizingPrompt, createReasoningPrompt
from breadth import createBreadthPrompt

import argparse, torch, re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

random.seed(42)
N_ROUNDS = 5

def initialize_models(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype=torch.float32,
        # attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def iter_jsonl(path):
    """1 行ずつ読み込んで辞書にして返すジェネレータ"""
    with path.open("r", encoding="utf-8") as fr:
        for ln in fr:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)
                

def write_jsonl(path, objs):
    """リストのオブジェクトを JSONL として書き出す"""
    with path.open("a", encoding="utf-8") as fw:
        for obj in objs:
            fw.write(json.dumps(obj, ensure_ascii=False) + "\n")
                
def encode_prompt(instr,tokenizer):
    instr = instr.strip()
    
    evol_prompts = [
        createConstraintsPrompt(instr),
        createDeepenPrompt(instr),
        createConcretizingPrompt(instr),
        createReasoningPrompt(instr),
        createBreadthPrompt(instr),
    ]
    
    chosed = random.choice(evol_prompts)
    
    messages = [
        {"role": "user", "content": chosed}
    ]
    
    formatted_promt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True, 
        tokenize=False
    )
    return formatted_promt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,required=True,)
    parser.add_argument("--seed_tasks_path", type=str, required=True, default="data/seed_tasks.jsonl",)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--cuda_device",type=str,default= "5,6")
    return parser.parse_args()
    

def main() -> None:
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    os.makedirs(args.output_dir, exist_ok=True)
        
    model, tokenizer = initialize_models(args.model_name)

    for rnd in tqdm(range(N_ROUNDS)):
        
        in_path = Path(args.seed_tasks_path) if rnd == 0 else Path(f"{args.output_dir}evol_instruct_round{rnd}.jsonl")
        out_path = Path(os.path.join(args.output_dir, f"evol_instruct_round{rnd+1}.jsonl"))
        
        batch_cnt = 0
        evol_prompts = []
        meta_buf = [] 

        for obj in tqdm(iter_jsonl(in_path)):
            src_instruction = obj["instruction"].strip()
            seed_id = obj["id"]
            
            meta_prompt = encode_prompt(src_instruction,tokenizer)

            evol_prompts.append(meta_prompt)
            meta_buf.append({"src_id": seed_id, "src_instruction": src_instruction})

            if len(evol_prompts) >= args.batch_size:
                batch_cnt += 1
                evol_instructions = call_llm(evol_prompts, model, tokenizer)
                
                records = [
                    {
                        "seed_id": buf["src_id"],
                        "syn_id": f"{buf['src_id']}_{rnd}",
                        "instruction": inst,
                    }
                for i, (buf, inst)  in enumerate(zip(meta_buf, evol_instructions))
                ]
                
                write_jsonl(out_path, records)
                print(f"【Round {rnd+1} | Batch {batch_cnt:03d}】→ {out_path}")

                evol_prompts.clear()
                meta_buf.clear()

        if evol_prompts:
            batch_cnt += 1
            evol_instructions = call_llm(evol_prompts, model, tokenizer)
                
            records = [
                {
                    "seed_id": buf["src_id"],
                    "syn_id": f"{buf['src_id']}_{rnd}",
                    "instruction": inst,
                }
            for i, (buf, inst)  in enumerate(zip(meta_buf, evol_instructions))
            ]
            
            write_jsonl(out_path, records)
            print(f"【Round {rnd+1} | Batch {batch_cnt:03d}】→ {out_path}")

if __name__ == "__main__":
    main()