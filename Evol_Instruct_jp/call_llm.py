import time
import torch

def instruction_generation(
    formatted_prompt, 
    model, 
    tokenizer, 
    max_generation_tokens=2048,
    temperature=1,
    top_p=0.95,
):

    with torch.inference_mode():
        # ProcessorまたはTokenizerでテキストのみをまとめて前処理
        inputs = tokenizer(
            formatted_prompt,       # List[str]
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_generation_tokens,
            do_sample=True,
            temperature=temperature,
            top_p = top_p,
            return_dict_in_generate=False,
        )
    
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, prompt_len:]
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    del inputs, outputs, generated_ids   # 明示的に削除
    torch.cuda.empty_cache()
    
    return responses

def call_llm(prompts, model, tokenizer):
    try:
        return instruction_generation(prompts, model, tokenizer)
    except RuntimeError as e:
        # device-side assert などでバッチ全体が落ちた場合
        if "device-side assert triggered" in str(e):
            print("device-side assert in batch! Try one by one.")
            results = []
            for idx, p in enumerate(prompts):
                try:
                    single_result = instruction_generation([p], model, tokenizer)
                    results.append(single_result[0])
                except Exception as ee:
                    print(f"Prompt {idx} failed and will be skipped. Error:", ee)
                    results.append("")  # または "SKIPPED"
            return results
        else:
            raise  # その他の例外はそのまま