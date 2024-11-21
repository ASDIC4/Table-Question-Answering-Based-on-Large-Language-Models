import json
from openai import OpenAI
import tiktoken
import pandas as pd



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

    cost = 0

    pre_prompt_give_column_name = """ 
    You are an advanced AI capable of analyzing and understanding information within tables. Read the column name of this table below.
    [TABLE]
    There is a question:
    [QUESTION]
    Please Give me some help. Think which columns or rows holds the information with higher probability. If the question has to answer use the whole table information to answer, just give me all the name of row and column.
    Ensure the final answer format is only "Final Answer: row: ['row1', 'row2', ...], column: ['column1', 'column2', ...]" form, no other form. And ensure the final answer is a number or entity names, as short as possible, without any explanation.
    """

    # train集 14152条

    with open("./data/train.json", "r") as f:
        data = json.load(f)

    print(f"len of dataset: {len(data)}")

    start = 501
    end = 2000
    output_file = f"./data/train_gpt_decompose_{start}_{end}.json"

    result = []
    cnt = 0
    for line in data:
        cnt += 1
        
        # print("why not")
        if cnt >= start and cnt <= end:
            table = line["table"]
            question = line['question']

            df = pd.DataFrame(table)
            markdown_table = df.to_markdown(index=False)
            # print(markdown_table)

            prompt = pre_prompt_give_column_name.replace("[TABLE]", markdown_table)\
                .replace("[QUESTION]", question)\
                .strip()
            response, tmp_cost = query_with_timeout(prompt)
            result.append({
                "idx": cnt,
                "response": response
            })
            cost += tmp_cost
            print(cnt)
        # sprint("why not2")

    with open(output_file, "w") as f:
        json.dump(result, f, indent=4)
    
    print(f"cost: {cost}")
