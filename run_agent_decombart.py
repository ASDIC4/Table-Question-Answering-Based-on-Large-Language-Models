import os
import json
from typing import Optional
from tqdm import tqdm
from fire import Fire
from agent import TableAgent, Model
from utils.data import construct_markdown_table
from utils.execute import markdown_to_df, remove_merged_suffixes, convert_cells_to_numbers
from utils.table import transpose, sort_dataframe
from run_helper import load_dataset, check_transpose, check_sort, read_json_file
import tiktoken

def main(
        model:Optional[str] = "gpt-3.5-turbo-0125", # base model of the agent (for short prompt to save money)
        long_model:Optional[str] = "gpt-3.5-turbo-0125", # long model of the agent (only used for long prompt)
        provider: str = "openai", # openai, huggingface, vllm
        dataset:str = "wtq", # wtq, tabfact
        perturbation: str = "none", # none, transpose, shuffle, transpose_shuffle
        use_full_table: bool = True, # whether to use the full table or only the partial table
        norm: bool = True, # whether to NORM the table
        disable_resort: bool = True, # whether to disable the resort stage in NORM
        norm_cache: bool = True, # whether to cache the normalization results so that we can reuse them
        sub_sample: bool = True, # whether to only run on the subset sampled data points
        resume:int = 0, # resume from the i-th data point
        stop_at:int = 1e6, # stop at the i-th data point
        self_consistency:int = 1, # how many times to do self consistency
        temperature:float=0.8, # temperature for model
        log_dir: str = "output/tabfact_agent", # directory to store the logs
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

    # with open("./data/wtq.json.bak.4k+.18.add-new-decom-1-bart-adjust.json", "r") as f:
    #     data = json.load(f)

    # no-use-answer
    # with open("./data/wtq.json.bak.4k+.18.add-new-decom-1-bart-no-use-answer-adjust.json", "r") as f:
    #     data = json.load(f)
    with open("/home/zjc/zjc-grad/data/wtq_4k+_sel_10.json","r") as f:
        data = json.load(f)

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
    break_flag = False
    total = len(data)
    # total = sum([len(d['sampled_indices']) for d in data]) if sub_sample else sum([len(d['questions']) for d in data])
    
    pbar = tqdm(total=stop_at if stop_at < total else total)

    # # read the results from output/wtq_cot_wo_norm
    # with open("output/wtq_agent_wo_norm/result.jsonl", "r") as f:
    #     temp = [json.loads(line) for line in f.readlines()]
    # temp = []
    
    #### start the loop ####
    for idx, d in enumerate(data):
        if break_flag:
            break

        title = d["title"]
        question_id = d["question_id"]
        table = construct_markdown_table(**d["table"])
        df = markdown_to_df(table)

        # transpose and sort if necessary
        transpose_flag = False
        resort_list = []
    
        df = convert_cells_to_numbers(df)
        # reset the table
        table = df.to_markdown()


        question = d["question"]
        answer = d["answer"]
        # question_id = d["ids"][idx]
            
        log_path = os.path.join(log_dir, "log", f"{global_i}.txt")
        # create the file
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        texts = []
        
        for _ in range(self_consistency):  
            # create the table agent
            agent = TableAgent(
                table=df,
                prompt_type=dataset,
                model=model,
                long_model=long_model,
                temperature=temperature,
                log_dir=log_path,
                use_full_table=use_full_table,
            )

            text, response = agent.run(question=question, title=title)
            print("text", text)
            texts.append(text)

            tmp_count1 = len(tokenizer.encode(str(agent.prompt)))
            tmp_count2 = len(tokenizer.encode(str(text))) 
            token_content.append({
                "idx": global_i,
                "type": "agent",
                "input tokens": tmp_count1,
                "output tokens": tmp_count2 
            })
            input_tokens += tmp_count1
            output_tokens += tmp_count2     


        res = {
            "idx": global_i,
            "answer": [answer],
            "text": texts if self_consistency > 1 else texts[0],
            "question_id": question_id,
            "title": title,
            "table": table,
            "question": question,
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