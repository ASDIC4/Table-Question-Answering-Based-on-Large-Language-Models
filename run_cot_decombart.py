import os
import json

from typing import Optional
from tqdm import tqdm
from fire import Fire
from agent import Model
import tiktoken

from utils.data import construct_markdown_table
from utils.execute import markdown_to_df, remove_merged_suffixes
from utils.table import transpose, sort_dataframe

from run_helper import load_dataset, get_cot_prompt, query, check_transpose, check_sort, read_json_file

def main(
        model:Optional[str] = "gpt-3.5-turbo-0125", # base model of the agent (for short prompt to save money)
        long_model:Optional[str] = "gpt-3.5-turbo-0125", # long model of the agent (only used for long prompt)
        provider: str = "openai", # openai, huggingface, vllm
        dataset:str = "wtq", # wtq or tabfact
        perturbation: str = "none", # none, transpose, shuffle, transpose_shuffle
        norm: bool = True, # whether to NORM the table
        disable_resort: bool = True, # whether to disable the resort stage in NORM
        norm_cache: bool = True, # whether to cache the normalization results so that we can reuse them
        sub_sample: bool = True, # whether to only run on the subset sampled data points
        resume:int = 0, # resume from the i-th data point
        stop_at:int = 1e6, # stop at the i-th data point
        self_consistency:int = 10, # how many times to do self consistency
        temperature:float=0.8, # temperature for model
        log_dir: str = "output/wtq_dp", # directory to store the logs
        cache_dir: str = "cache", # directory to store the cache (normalization results)
):
    token_content = []
    #### create log & cache dir and save config ####
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    tokens_path = os.path.join(log_dir, "token.json")
    # store the config
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({key: value for key, value in locals().items() if key != 'f'}, f, indent=4)
    
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo-0125")
    input_tokens = 0
    output_tokens = 0

    #### load dataset and cot prompt ####
    # with open("./data/wtq.json.bak.4k+.18.add-new-decom-1-bart-adjust.json", "r") as f:
    #     data = json.load(f)
    
    # no-use-answer
    # with open("./data/wtq.json.bak.4k+.18.add-new-decom-1-bart-no-use-answer-adjust.json", "r") as f:
    #     data = json.load(f)
    with open("/home/zjc/zjc-grad/data/wtq_4k+_sel_10.json","r") as f:
        data = json.load(f)
    cot_prompt = get_cot_prompt(dataset)

    #### load the model ####
    if model:
        model = Model(model, provider=provider)
    if long_model:
        long_model = Model(long_model, provider=provider)
    
    #### load the cache ####
    transpose_cache = read_json_file(os.path.join(cache_dir, "transpose.json"))
    resort_cache = read_json_file(os.path.join(cache_dir, "resort.json"))
    
    #### prepare the iterator ####
    global_i = 0
    bre
    # total = sum([len(d['sampled_indices']) for d in data]) if sub_sample else sum([len(d['questions']) for d in data])
    pbar = tqdm(total=stop_at if stop_at < total else total)
    
    #### start the loop ####
    for idx, d in enumerate(data):
        
        title = d["title"]
        question_id = d["question_id"]

        table = construct_markdown_table(**d["table"])    
        df = markdown_to_df(table)

        # transpose and sort if necessary
        transpose_flag = False
        resort_list = []
        
        # # reset the table
        # table = df.to_markdown()

        question = d["question"]
        answer = d["answer"]

        prompt = cot_prompt.replace("[TABLE]", table)\
            .replace("[QUESTION]", question)\
            .replace("[TITLE]", title)\
            .strip()

        text, response = query(model, long_model, prompt, temperature, self_consistency)

        tmp_count1 = len(tokenizer.encode(str(prompt)))
        tmp_count2 = len(tokenizer.encode(str(text))) 
        token_content.append({
            "idx": global_i,
            "type": "dp",
            "input tokens": tmp_count1,
            "output tokens": tmp_count2 
        })
        input_tokens += tmp_count1
        output_tokens += tmp_count2      
           
        # print("第idx: {} 个问题结束")
        log_path = os.path.join(log_dir, "log", f"{global_i}.txt")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            f.write("===================Title===================\n")
            f.write(title + "\n")
            f.write("===================Table===================\n")
            f.write(table + "\n")
            f.write("===================Question===================\n")
            f.write(question + "\n")
            f.write("===================Text===================\n")
            f.write(text if isinstance(text, str) else "\n".join(text))
            f.write("\n")
            f.write("===================Answer===================\n")
            f.write(",".join(answer) if isinstance(answer, list) else str(answer))
            f.write("\n")

        res = {
            "idx": global_i,
            "answer": [answer],
            "text": text,
            "title": title,
            "question": question,
            "question_id": question_id,
            "table": table,
        }

        with open(os.path.join(log_dir, "result.jsonl"), "a") as f:
            json.dump(res, f)
            f.write("\n")

        global_i += 1
        pbar.update(1)
    
    token_content.append({
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    })

    with open(tokens_path, "w") as f:
        json.dump(token_content, f, indent=4)

if __name__ == "__main__":
    Fire(main)