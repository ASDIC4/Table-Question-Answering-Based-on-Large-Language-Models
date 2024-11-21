import json
from openai import OpenAI
import tiktoken
import pandas as pd
import argparse 

import sys
sys.path.append('../')
from run_helper import query
from agent import Model

def query_with_timeout(prompt):
    # print("query in")
    tmp_cost = 0
    client = OpenAI(api_key = API_KEY)

    tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo-0125')
    
    input_tokens = len(tokenizer.encode(prompt))

    content = ""
    if input_tokens >= 16000:
        response = None
        output_tokens = 0

        tmp_cost += input_tokens * 0.0005 / 1000
    # tmp_cost += input_tokens * 0.0005 / 1000
    else:
    # response = None
        response =  client.chat.completions.create(
            model = 'gpt-3.5-turbo-0125',
            messages = [{"role": "user", "content": prompt}]
        )   
        content = response.choices[0].message.content
        output_tokens = response.usage.total_tokens

        tmp_cost += output_tokens * 0.0005 / 1000

    # # output_tokens = input_tokens
    # tmp_cost += output_tokens * 0.0005 / 1000
    # # print(f"output: {output_tokens} tokens, {output_tokens * 0.0015 / 1000} prices")
    

    return content, tmp_cost

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process start and end parameters.')
    
    parser.add_argument('--start', type=int, help='start index', required=True)
    parser.add_argument('--end', type=int, help='end index', required=True)
    args = parser.parse_args()

    start = args.start
    end = args.end
    output_file = f"./data/train_gpt_new_decom_1_{start}_{end}.json"

    # train集 14152条 # 14149
    with open("./data/train.json", "r") as f:
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
    for line in data:
        cnt += 1
        
        # print("why not")
        if cnt >= start and cnt <= end:
            table = line["table"]
            question = line['question']
            answer = line['answer']

            header_df = pd.DataFrame(table[0])
            df = pd.DataFrame(table)
            markdown_header = header_df.to_markdown(index=False)
            markdown_table = df.to_markdown(index=False)
            # print(markdown_table)

            # print("markdown_header", ",".join(table[0]))

            prompt = pre_prompt_origin.replace("[HEADER]", ",".join(table[0]))\
                .replace("[QUESTION]", question)\
                .replace("[ANSWER]", answer) \
                .strip()
            
            text, response = query(model, long_model, prompt, temperature, self_consistency)
            
            
            result.append({
                "idx": cnt,
                "response": text,
                "question": question,
                "answer": answer,
                "table": markdown_table
            })
            # cost += tmp_cost
            print(f"now: {cnt}, start::{start}, end: {end}")

    with open(output_file, "w") as f:
        json.dump(result, f, indent=4)
    
    # print(f"cost: {cost}")
