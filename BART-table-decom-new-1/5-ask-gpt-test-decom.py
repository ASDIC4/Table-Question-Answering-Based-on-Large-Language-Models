import json
from openai import OpenAI
import tiktoken
import pandas as pd
import argparse 

import sys
sys.path.append('../')
from run_helper import query
from agent import Model
from utils.data import construct_markdown_table

if __name__ == "__main__":

    
   
    # parser = argparse.ArgumentParser(description='Process start and end parameters.')
    
    # parser.add_argument('--start', type=int, help='start index', required=True)
    # parser.add_argument('--end', type=int, help='end index', required=True)
    # args = parser.parse_args()

    # start = args.start
    # end = args.end

    output_file = f"./data/test_gpt_new_decom_1_wtq4k+153.json"

     # train集 14152条 # 14149
    with open("./data/wtq.json.bak.4k+.18", "r") as f:
        data = json.load(f)

    print(f"len of dataset: {len(data)}")


    model = "gpt-3.5-turbo-0125"
    long_model = "gpt-3.5-turbo-0125"
    provider = "openai"

    model = Model(model, provider=provider)
    long_model = Model(long_model, provider=provider)
    self_consistency = 1, # how many times to do self consistency
    temperature = 0.8

    pre_prompt_origin = """ 
    You are an advanced AI capable of analyzing and understanding information within tables.
    Here is a table, and a question related to table needed to be answered.
    Table header:
    [HEADER]

    Question:
    [QUESTION]

    Answer:
    [ANSWER]
    
    I need you to tell me the columns that is needed to answer the question.
    Sometimes the question is about a whole row, which means the question still needs all the columns.

    Ensure the final answer format is only "Final Answer: columns: ['column1', 'column2', ...]" 
    """

    result = []
    cnt = 0
    #### start the loop ####
    for table_idx, d in enumerate(data):

        table = construct_markdown_table(**d["table"])

        for idx in range(len(d["questions"])):
            cnt += 1
            question = d['questions'][idx]
            answer = d['answers'][idx][0]

            # header_df = pd.DataFrame(d["table"]["header"])
            # df = pd.DataFrame(table)
            # markdown_header = header_df.to_markdown(index=False)\
            print(",".join(d["table"]["header"]))

            prompt = pre_prompt_origin.replace("[HEADER]", ",".join(d["table"]["header"]))\
                .replace("[QUESTION]", question)\
                .replace("[ANSWER]", answer) \
                .strip()
            
            text, response = query(model, long_model, prompt, temperature, self_consistency)
            
            result.append({
                "idx": cnt,
                "response": text,
                "question": question,
                "answer": answer,
                "table": table
            })
            # cost += tmp_cost
            print(f"now: {cnt}")
            # , start::{start}, end: {end}

    with open(output_file, "w") as f:
        json.dump(result, f, indent=4)
    
    # print(f"cost: {cost}")
